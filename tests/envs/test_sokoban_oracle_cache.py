"""Correctness + performance test for Sokoban oracle_advantage_cache.

The cache reuses N(s_{t+1}) from the previous step as N(s_t) for the next step,
halving oracle solver calls (2/step -> 1/step). This is only valid if it is
NUMERICALLY IDENTICAL to the uncached path. These tests run REAL environments
(no mocks) and assert:

  1. Byte-identical oracle_advantage sequences (cache on vs off), across many
     seeds and both single- and multi-action steps.
  2. reset() invalidates the cache (no cross-episode leakage).
  3. The cache actually reduces get_min_steps call count (the perf win is real).
  4. Default config keeps the feature OFF (zero impact on existing runs).
"""

import pytest

pytest.importorskip("gym_sokoban")
pytest.importorskip("gym")

from rllm.environments.sokoban import sokoban as sokoban_mod
from rllm.environments.sokoban.sokoban import SokobanEnv


# A fixed action script exercised by every env so both cache modes see the
# exact same transitions. Mixes single moves, text moves, an invalid action,
# and a multi-action turn (the "||" path in step()).
ACTION_SCRIPT = [1, 2, "Up", 3, 0, "Right", "2||3", 4, "Left", 1]


def _run_episode(seed: int, cache: bool) -> list[float]:
    """Run the fixed script on a fresh env and collect oracle_advantage/step."""
    env = SokobanEnv(
        seed=seed,
        dim_room=(6, 6),
        num_boxes=2,
        max_steps=50,
        compute_oracle_advantage=True,
        oracle_advantage_cache=cache,
        # Keep deadlock detection at its default ("solver"); it uses weight=2.5
        # and must NOT interact with the weight=1.0 oracle cache.
    )
    env.reset(working_seed=seed)
    advs = []
    for action in ACTION_SCRIPT:
        _, _, done, info = env.step(action)
        advs.append(info.get("oracle_advantage"))
        if done:
            break
    return advs


class TestOracleAdvantageCache:
    def test_default_config_is_off(self):
        """Existing runs must be unaffected: feature defaults to disabled."""
        env = SokobanEnv(seed=1)
        assert env.oracle_advantage_cache is False

    @pytest.mark.parametrize("seed", [42, 123, 777, 2024, 31337, 9, 555, 88])
    def test_cached_equals_uncached(self, seed):
        """The cache path must be numerically identical, step-for-step."""
        uncached = _run_episode(seed, cache=False)
        cached = _run_episode(seed, cache=True)
        assert cached == uncached, (
            f"seed={seed}: oracle_advantage diverged.\n"
            f"  uncached={uncached}\n  cached  ={cached}"
        )
        # Sanity: we actually produced advantages (not an all-None no-op).
        assert any(a is not None for a in uncached)

    def test_cache_reduces_solver_calls(self, monkeypatch):
        """Perf claim: cache turns 2 solver calls/step into 1 (after step 1)."""
        calls = {"n": 0}
        real_get_min_steps = sokoban_mod.get_min_steps

        def counting_get_min_steps(*args, **kwargs):
            calls["n"] += 1
            return real_get_min_steps(*args, **kwargs)

        # Patch the name used inside sokoban.step() (imported symbol).
        monkeypatch.setattr(sokoban_mod, "get_min_steps", counting_get_min_steps)

        # Uncached: 2 oracle calls per executed step.
        calls["n"] = 0
        uncached_advs = _run_episode(42, cache=False)
        uncached_calls = calls["n"]

        # Cached: 1 oracle call per executed step, +1 for the very first prev_min.
        calls["n"] = 0
        cached_advs = _run_episode(42, cache=True)
        cached_calls = calls["n"]

        # Same behavior (guards against a broken monkeypatch run).
        assert cached_advs == uncached_advs
        # The cache must strictly reduce solver calls.
        assert cached_calls < uncached_calls, (
            f"cache did not reduce solver calls: "
            f"uncached={uncached_calls}, cached={cached_calls}"
        )

    def test_reset_invalidates_cache(self):
        """A second episode must not reuse the previous episode's N(s)."""
        env = SokobanEnv(
            seed=42,
            dim_room=(6, 6),
            num_boxes=2,
            max_steps=50,
            compute_oracle_advantage=True,
            oracle_advantage_cache=True,
        )
        # Episode 1
        env.reset(working_seed=42)
        for action in ACTION_SCRIPT:
            _, _, done, _ = env.step(action)
            if done:
                break
        # Fresh episode with a DIFFERENT board; cache must be cleared on reset.
        env.reset(working_seed=999)
        assert env._cached_oracle_min is None

        # And the first step of episode 2 must match a from-scratch (uncached) run
        # on the same board — proving no stale value leaked in as prev_min.
        _, _, _, info_cached = env.step(ACTION_SCRIPT[0])
        expected = _run_episode(999, cache=False)[0]
        assert info_cached.get("oracle_advantage") == expected
