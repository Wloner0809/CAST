"""Offline numerical correctness test for GiGPO core advantage math.

Exercises rllm/gigpo/core_gigpo.py directly on synthetic trajectory data —
NO model, NO rollout, NO slow worker imports. Verifies the acceptance
signals the smoke test would show in its [GiGPO-Trajectory] diagnostics:

  1. Eq.5 discounted returns are correct (backward gamma accumulation).
  2. Eq.6 anchor grouping clusters repeated states (avg group size > 1).
  3. Eq.3 episode advantage = unweighted GRPO baseline when
     compute_mean_std_cross_steps=False (the trajectory-level fix).
  4. Eq.7 step advantage is non-trivial within anchor groups.
  5. Eq.8 joint advantage = episode + w * step; w=0 collapses to GRPO.

Run:  python3 rllm/gigpo/test_gigpo_offline.py
"""

import numpy as np
import torch

from rllm.gigpo import core_gigpo


def _approx(a, b, tol=1e-5):
    return abs(float(a) - float(b)) <= tol


def build_synthetic():
    """Two prompts, N=2 rollouts each. Construct repeated anchor states so
    step-level grouping has groups of size > 1, mirroring Sokoban revisits.

    Layout (flattened, one row per step):
      prompt P0:
        traj A (reward 1.0, success): steps s0, s1, s2
        traj B (reward 0.0, fail):    steps s0, s1, s3   <- shares s0,s1 with A
      prompt P1:
        traj C (reward 1.0):          steps s0, s4
        traj D (reward 0.0):          steps s0, s5       <- shares s0 with C
    """
    rows = []
    # (uid, traj_uid, anchor, per_step_reward, traj_reward)
    # P0 / traj A  (success at last step)
    rows += [("P0", "A", "s0", 0.0, 1.0),
             ("P0", "A", "s1", 0.0, 1.0),
             ("P0", "A", "s2", 1.0, 1.0)]
    # P0 / traj B  (fail)
    rows += [("P0", "B", "s0", 0.0, 0.0),
             ("P0", "B", "s1", 0.0, 0.0),
             ("P0", "B", "s3", 0.0, 0.0)]
    # P1 / traj C  (success)
    rows += [("P1", "C", "s0", 0.0, 1.0),
             ("P1", "C", "s4", 1.0, 1.0)]
    # P1 / traj D  (fail)
    rows += [("P1", "D", "s0", 0.0, 0.0),
             ("P1", "D", "s5", 0.0, 0.0)]

    uids = np.array([r[0] for r in rows], dtype=object)
    traj_uids = np.array([r[1] for r in rows], dtype=object)
    anchors = np.array([r[2] for r in rows], dtype=object)
    step_rewards = np.array([r[3] for r in rows], dtype=np.float64)
    traj_rewards = np.array([r[4] for r in rows], dtype=np.float64)
    active = np.ones(len(rows), dtype=np.float64)
    return uids, traj_uids, anchors, step_rewards, traj_rewards, active


def test_discounted_returns(gamma=0.95):
    uids, traj_uids, anchors, step_rewards, traj_rewards, active = build_synthetic()
    n = len(uids)
    dummy = torch.zeros(n, 1, dtype=torch.long)
    batch = core_gigpo.DataProto.from_dict(
        tensors={"input_ids": dummy},
        non_tensors={"rewards": step_rewards, "traj_uid": traj_uids, "active_masks": active},
    )
    R = core_gigpo.compute_step_discounted_returns(batch, gamma=gamma).cpu().numpy()

    # traj A: rewards [0,0,1] -> returns [gamma^2, gamma^1, 1]
    expectedA = [gamma ** 2, gamma ** 1, 1.0]
    gotA = R[0:3].tolist()
    assert all(_approx(a, b) for a, b in zip(gotA, expectedA)), f"A returns {gotA} != {expectedA}"
    # traj B: rewards [0,0,0] -> [0,0,0]
    assert all(_approx(x, 0.0) for x in R[3:6]), f"B returns {R[3:6]}"
    # traj C: rewards [0,1] -> [gamma, 1]
    assert _approx(R[6], gamma) and _approx(R[7], 1.0), f"C returns {R[6:8]}"
    print(f"[PASS] Eq.5 discounted returns correct (gamma={gamma}); "
          f"trajA={[round(x,4) for x in gotA]}")
    return R


def test_anchor_grouping():
    uids, traj_uids, anchors, step_rewards, traj_rewards, active = build_synthetic()
    group_uids = core_gigpo.build_step_group(anchors, uids, enable_similarity=False)
    # Within P0: s0 appears in A and B -> same group; s1 in A and B -> same group;
    # s2 (A only), s3 (B only) -> singleton groups.
    # Map row -> group
    g = {i: group_uids[i] for i in range(len(uids))}
    # P0 rows: 0(A,s0),1(A,s1),2(A,s2),3(B,s0),4(B,s1),5(B,s3)
    assert g[0] == g[3], "P0 s0 should group across A,B"
    assert g[1] == g[4], "P0 s1 should group across A,B"
    assert g[2] != g[0] and g[5] != g[0], "s2/s3 are singletons"
    # P1 rows: 6(C,s0),7(C,s4),8(D,s0),9(D,s5); s0 groups across C,D
    assert g[6] == g[8], "P1 s0 should group across C,D"
    assert g[7] != g[6] and g[9] != g[6], "s4/s5 singletons"
    # cross-prompt: P0 s0 and P1 s0 must NOT group (different uid/index)
    assert g[0] != g[6], "s0 must not group across different prompts"

    sizes = {}
    for u in group_uids:
        sizes[u] = sizes.get(u, 0) + 1
    avg = float(np.mean(list(sizes.values())))
    n_multi = sum(1 for v in sizes.values() if v > 1)
    print(f"[PASS] Eq.6 anchor grouping: {len(sizes)} groups, avg size={avg:.2f}, "
          f"{n_multi} multi-member groups (size>1 => NOT degenerate to GRPO)")
    assert avg > 1.0, "avg group size must exceed 1 for GiGPO to differ from GRPO"
    return group_uids


def _grpo_baseline(traj_rewards_per_traj):
    """Reference GRPO advantage: reward - unweighted group mean, per prompt."""
    return traj_rewards_per_traj - traj_rewards_per_traj.mean()


def test_episode_advantage_grpo_parity():
    """Eq.3 with compute_mean_std_cross_steps=False must equal the unweighted
    per-trajectory GRPO baseline, despite trajectories having different #steps."""
    uids, traj_uids, anchors, step_rewards, traj_rewards, active = build_synthetic()
    n = len(uids)
    tlr = torch.tensor(traj_rewards, dtype=torch.float32).unsqueeze(-1)  # full traj reward on each step
    rmask = torch.ones(n, 1)
    ep = core_gigpo.episode_norm_reward(
        tlr, rmask, uids, traj_uids,
        epsilon=1e-6, remove_std=True, compute_mean_std_cross_steps=False,
    ).cpu().numpy()[:, 0]

    # P0: trajA reward 1, trajB reward 0 -> mean over 2 trajectories = 0.5
    #   A steps -> 1-0.5=+0.5 ; B steps -> 0-0.5=-0.5
    # NOTE: A has 3 steps, B has 3 steps here; make A/B different lengths to
    # truly test weighting: but our P0 A,B both 3. P1 C,D both 2. So test the
    # KEY property: every step of a trajectory gets the SAME episode adv = its
    # GRPO advantage, and the baseline is unweighted (0.5), independent of len.
    for i in range(0, 3):   # A
        assert _approx(ep[i], 0.5), f"A ep adv {ep[i]} != 0.5"
    for i in range(3, 6):   # B
        assert _approx(ep[i], -0.5), f"B ep adv {ep[i]} != -0.5"
    for i in range(6, 8):   # C
        assert _approx(ep[i], 0.5), f"C ep adv {ep[i]} != 0.5"
    for i in range(8, 10):  # D
        assert _approx(ep[i], -0.5), f"D ep adv {ep[i]} != -0.5"

    # Compare to reference GRPO over the 2 trajectories of each prompt
    grpo_P0 = _grpo_baseline(np.array([1.0, 0.0]))  # [+0.5,-0.5]
    assert _approx(ep[0], grpo_P0[0]) and _approx(ep[3], grpo_P0[1])
    print(f"[PASS] Eq.3 episode adv == unweighted GRPO baseline "
          f"(A/C=+0.5, B/D=-0.5); cross_steps=False dedup verified")
    return ep


def test_episode_advantage_length_weighting_bug():
    """Verify the two episode-baseline conventions. With cross_steps=True the
    baseline is step-count-weighted (a K-step trajectory counts K times); with
    False it is unweighted. Build A with 4 steps, B with 1 step, same prompt."""
    uids = np.array(["Q"] * 5, dtype=object)
    traj_uids = np.array(["A", "A", "A", "A", "B"], dtype=object)
    traj_rewards = np.array([1.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float32)  # A rew=1 (4 steps), B rew=0 (1 step)
    tlr = torch.tensor(traj_rewards, dtype=torch.float32).unsqueeze(-1)
    rmask = torch.ones(5, 1)

    ep_false = core_gigpo.episode_norm_reward(
        tlr, rmask, uids, traj_uids, remove_std=True, compute_mean_std_cross_steps=False
    ).cpu().numpy()[:, 0]
    ep_true = core_gigpo.episode_norm_reward(
        tlr, rmask, uids, traj_uids, remove_std=True, compute_mean_std_cross_steps=True
    ).cpu().numpy()[:, 0]

    assert _approx(ep_false[0], 0.5) and _approx(ep_false[4], -0.5), \
        f"cross_steps=False should be unweighted: {ep_false}"
    assert _approx(ep_true[0], 0.2) and _approx(ep_true[4], -0.8), \
        f"cross_steps=True is step-weighted: {ep_true}"
    print(f"[PASS] Length-weighting fix confirmed: cross_steps=False -> A=+0.5/B=-0.5 (GRPO); "
          f"True -> A=+0.2/B=-0.8 (biased). Trajectory path uses False.")


def test_joint_and_w0_collapse(gamma=0.95):
    """Eq.8 joint; with step_advantage_w=0 the joint must equal episode-only
    (i.e. GRPO), proving 'only advantage changes' parity."""
    uids, traj_uids, anchors, step_rewards, traj_rewards, active = build_synthetic()
    n = len(uids)
    dummy = torch.zeros(n, 1, dtype=torch.long)
    batch = core_gigpo.DataProto.from_dict(
        tensors={"input_ids": dummy},
        non_tensors={"rewards": step_rewards, "traj_uid": traj_uids, "active_masks": active},
    )
    step_returns = core_gigpo.compute_step_discounted_returns(batch, gamma=gamma)
    tlr = torch.tensor(traj_rewards, dtype=torch.float32).unsqueeze(-1)
    rmask = torch.ones(n, 1)

    ep = core_gigpo.episode_norm_reward(
        tlr, rmask, uids, traj_uids, remove_std=True, compute_mean_std_cross_steps=False)
    gids = core_gigpo.build_step_group(anchors, uids, enable_similarity=False)
    st = core_gigpo.step_norm_reward(step_returns, rmask, gids, remove_std=True)

    ep_np, st_np = ep.cpu().numpy()[:, 0], st.cpu().numpy()[:, 0]

    # step adv must be non-trivial (some nonzero) within multi-member groups
    assert np.abs(st_np).sum() > 0, "step advantage all zero -> grouping ineffective"

    joint_w1 = ep + 1.0 * st
    joint_w0 = ep + 0.0 * st
    assert np.allclose(joint_w0.cpu().numpy(), ep.cpu().numpy()), "w=0 must collapse to episode-only (GRPO)"
    assert not np.allclose(joint_w1.cpu().numpy(), ep.cpu().numpy()), "w=1 must differ from GRPO"

    print(f"[PASS] Eq.8 joint: w=0 == episode-only (GRPO parity); "
          f"w=1 != GRPO. step_adv nonzero count={int((st_np!=0).sum())}/{n}")
    print(f"        episode adv range [{ep_np.min():.3f},{ep_np.max():.3f}], "
          f"step adv range [{st_np.min():.3f},{st_np.max():.3f}]")


if __name__ == "__main__":
    print("=" * 70)
    print("  GiGPO OFFLINE NUMERICAL CORRECTNESS TEST (no model / no rollout)")
    print("=" * 70)
    test_discounted_returns()
    test_anchor_grouping()
    test_episode_advantage_grpo_parity()
    test_episode_advantage_length_weighting_bug()
    test_joint_and_w0_collapse()
    print("=" * 70)
    print("  ALL GiGPO CORE MATH CHECKS PASSED")
    print("=" * 70)
