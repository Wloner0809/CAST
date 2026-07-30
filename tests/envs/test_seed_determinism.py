"""
Comprehensive seed determinism tests for all rllm environments.

Motivation
----------
In ``agent_ppo_trainer.py``, ``batch.repeat(repeat_times=rollout_n, interleave=True)``
duplicates the same ``extra_info`` dict N times.  Then ``init_envs_and_agents``
creates N independent env instances via ``from_dict`` using
``ThreadPoolExecutor(max_workers=64)``.  All N instances **must** produce
identical puzzles so GRPO can compare N different solution attempts on the same
problem.

This test file provides regression coverage against a critical bug already
found and fixed: ``Sokoban``'s ``seed_context()`` used global
``random``/``numpy.random`` state, so under ``ThreadPoolExecutor`` threads
overwrote each other's global state, producing different rooms from the same
seed.

The 6 test scenarios are:
  1. Rollout N determinism  (N=16 envs from same from_dict config)
  2. Same instance multiple resets  (K=5 resets on one env)
  3. from_dict vs constructor  (only for 4 core envs)
  4. Concurrent creation via ThreadPoolExecutor  (mirrors init_envs_and_agents)
  5. Different seeds/tasks produce different observations
  6. Post-action determinism  (same seed, same action sequence)

Run::

    python3 -m pytest tests/envs/test_seed_determinism.py -v -s
    python3 -m pytest tests/envs/test_seed_determinism.py -v -s -k "sokoban"
    python3 -m pytest tests/envs/test_seed_determinism.py -v -s -k "concurrent"
"""

from __future__ import annotations

import dataclasses
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import pytest


# ---------------------------------------------------------------------------
# EnvSpec: frozen descriptor for one environment
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class EnvSpec:
    """Description of an environment for seed determinism testing."""

    name: str
    env_class: type
    from_dict_config: dict[str, Any]
    alt_config: dict[str, Any]           # Must produce a *different* puzzle
    needs_close: bool = False
    pool_shutdown: Callable | None = None  # e.g. ALFWorldEnv.shutdown_pool
    first_action: str | int | None = None  # Single action for post-action test
    action_sequence: list[str | int] | None = None  # For multi-step post-action
    deferred_rng: bool = False            # True if RNG only fires on first action (Minesweeper)
    # Constructor kwargs (for from_dict vs constructor test)
    constructor_kwargs: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Registry: try-import each env; skip if deps missing
# ---------------------------------------------------------------------------

_ENV_SPECS: list[EnvSpec] = []


def _register(spec: EnvSpec) -> None:
    _ENV_SPECS.append(spec)


# ---- Sokoban ----
try:
    import gym_sokoban  # noqa: F401
    import gym  # noqa: F401
    from rllm.environments.sokoban.sokoban import SokobanEnv

    _sokoban_cfg = {"seed": 42, "dim_x": 6, "dim_y": 6, "num_boxes": 1, "max_steps": 50}
    _sokoban_alt = {"seed": 999, "dim_x": 6, "dim_y": 6, "num_boxes": 1, "max_steps": 50}
    _register(EnvSpec(
        name="sokoban",
        env_class=SokobanEnv,
        from_dict_config=_sokoban_cfg,
        alt_config=_sokoban_alt,
        needs_close=False,
        first_action=1,  # "Up"
        action_sequence=[1, 2, 3, 4, 1],  # Up, Down, Left, Right, Up
        constructor_kwargs={"seed": 42, "dim_room": (6, 6), "num_boxes": 1, "max_steps": 50},
    ))
except ImportError:
    pass


# ---- Minesweeper ----
try:
    from rllm.environments.minesweeper.minesweeper_env import MinesweeperEnv

    _ms_cfg = {"seed": 42, "rows": 5, "cols": 5, "num_mines": 3, "max_steps": 50}
    _ms_alt = {"seed": 999, "rows": 5, "cols": 5, "num_mines": 3, "max_steps": 50}
    _register(EnvSpec(
        name="minesweeper",
        env_class=MinesweeperEnv,
        from_dict_config=_ms_cfg,
        alt_config=_ms_alt,
        needs_close=False,
        first_action="reveal 2 2",
        action_sequence=["reveal 2 2", "reveal 0 0", "reveal 4 4"],
        deferred_rng=True,  # Mines placed on first reveal, not on reset
        constructor_kwargs={"seed": 42, "rows": 5, "cols": 5, "num_mines": 3, "max_steps": 50},
    ))
except ImportError:
    pass


# ---- ALFWorld ----
try:
    import ray  # noqa: F401
    import textworld  # noqa: F401
    from rllm.environments.alfworld.alfworld_env import ALFWorldEnv
    import pandas as pd
    from pathlib import Path

    _alf_data = Path(__file__).resolve().parents[2] / "data" / "datasets" / "alfworld" / "eval_id.parquet"
    _alf_game_files: list[str] = []
    if _alf_data.exists():
        _alf_df = pd.read_parquet(_alf_data)
        if len(_alf_df) >= 2:
            _alf_game_files = [_alf_df.iloc[0]["game_file"], _alf_df.iloc[1]["game_file"]]

    if len(_alf_game_files) == 2:
        _alf_cfg = {"game_file": _alf_game_files[0], "max_steps": 20}
        _alf_alt = {"game_file": _alf_game_files[1], "max_steps": 20}
        _register(EnvSpec(
            name="alfworld",
            env_class=ALFWorldEnv,
            from_dict_config=_alf_cfg,
            alt_config=_alf_alt,
            needs_close=True,
            pool_shutdown=ALFWorldEnv.shutdown_pool,
            first_action="look",
        ))
except ImportError:
    pass


# ---- WebShop ----
try:
    import ray  # noqa: F401
    import glob as _glob
    import os as _os
    # pyserini needs Java 11; probe the usual locations unless JAVA_HOME is preset.
    _java11 = next(
        (p for p in sorted(_glob.glob("/usr/lib/jvm/java-11-openjdk*"))
         + ["/usr/lib/jvm/java-11-openjdk-amd64", "/usr/lib/jvm/java-11"]
         if _os.path.isdir(p)),
        "",
    )
    if _java11:
        _os.environ.setdefault("JAVA_HOME", _java11)
        _os.environ.setdefault("JVM_PATH", _os.path.join(_java11, "lib", "server", "libjvm.so"))
    if _os.environ.get("JAVA_HOME"):
        _os.environ["PATH"] = _os.environ["JAVA_HOME"] + "/bin:" + _os.environ.get("PATH", "")
    from rllm.environments.webshop.webshop_env import WebShopEnv

    _ws_cfg = {"session_id": 0, "seed": 42, "max_steps": 20}
    _ws_alt = {"session_id": 1, "seed": 42, "max_steps": 20}
    _register(EnvSpec(
        name="webshop",
        env_class=WebShopEnv,
        from_dict_config=_ws_cfg,
        alt_config=_ws_alt,
        needs_close=True,
        pool_shutdown=WebShopEnv.shutdown_pool,
        first_action="search[laptop]",
    ))
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_obs(obs: Any) -> str:
    """Normalize observation to a comparable string.

    Handles dict observations by sorting keys and joining.
    """
    if isinstance(obs, dict):
        parts = []
        for key in sorted(obs.keys()):
            parts.append(f"{key}={obs[key]}")
        return "\n".join(parts)
    return str(obs)


def _close_env(env: Any, spec: EnvSpec) -> None:
    """Safely close a single environment."""
    if env is None:
        return
    if spec.needs_close and hasattr(env, "close"):
        try:
            env.close()
        except Exception:
            pass


def _close_all(envs: list, spec: EnvSpec) -> None:
    """Safely close a list of environments."""
    for env in envs:
        _close_env(env, spec)


def _get_action_sequence(spec: EnvSpec, env: Any) -> list[str | int]:
    """Get an action sequence for post-action testing."""
    if spec.action_sequence is not None:
        return spec.action_sequence

    if spec.first_action is not None:
        return [spec.first_action]

    return []


def _get_initial_obs(spec: EnvSpec, env: Any) -> str:
    """Get the initial observation after reset, handling deferred_rng envs.

    For Minesweeper (deferred_rng=True), take the first action before
    comparing, since mines are placed on first reveal.  The step observation
    text is just action feedback (e.g. "you revealed cell (2,2)") — the
    actual board state lives in ``info['suffix']`` / ``info['board']``, so
    we incorporate that to detect real differences.
    """
    obs, info = env.reset()
    if spec.deferred_rng and spec.first_action is not None:
        obs, _, _, step_info = env.step(spec.first_action)
        # Append board state from info for a more complete comparison.
        # Minesweeper puts the rendered board in info['suffix'] or info['board'].
        suffix = step_info.get("suffix", "") or step_info.get("board", "")
        if suffix:
            return normalize_obs(obs) + "\n" + str(suffix)
    return normalize_obs(obs)


# ---------------------------------------------------------------------------
# Parametrization helpers
# ---------------------------------------------------------------------------

_spec_ids = [s.name for s in _ENV_SPECS]
_spec_params = [pytest.param(s, id=s.name) for s in _ENV_SPECS]

# Specs that have constructor_kwargs for from_dict vs constructor test
_constructor_specs = [s for s in _ENV_SPECS if s.constructor_kwargs is not None]
_constructor_params = [pytest.param(s, id=s.name) for s in _constructor_specs]


# ---------------------------------------------------------------------------
# Module-scoped cleanup
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True, scope="module")
def _cleanup_pools():
    """Shutdown worker pools (ALFWorld, WebShop) after all tests in module."""
    yield
    for spec in _ENV_SPECS:
        if spec.pool_shutdown is not None:
            try:
                spec.pool_shutdown()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Skip if no environments registered
# ---------------------------------------------------------------------------

if not _ENV_SPECS:
    pytest.skip("No environment dependencies available", allow_module_level=True)


# ===========================================================================
# TEST 1: Rollout N Determinism
# ===========================================================================

_ROLLOUT_N = 16
_ROLLOUT_SEEDS = [42, 123, 7]


@pytest.mark.parametrize("spec", _spec_params)
@pytest.mark.parametrize("seed_offset", [0, 1, 2], ids=["seed42", "seed123", "seed7"])
def test_rollout_n_determinism(spec: EnvSpec, seed_offset: int):
    """N envs from the same from_dict config must produce identical initial obs.

    This is the core invariant that GRPO relies on: all N rollouts start from
    the exact same puzzle state.
    """
    seed_map = {0: 42, 1: 123, 2: 7}
    test_seed = seed_map[seed_offset]

    # Build config with the test seed
    cfg = dict(spec.from_dict_config)
    if "seed" in cfg:
        cfg["seed"] = test_seed
    elif "game_file" in cfg:
        pass  # ALFWorld determinism is by game_file, not seed
    elif "session_id" in cfg:
        pass  # WebShop determinism is by session_id

    envs = []
    try:
        for _ in range(_ROLLOUT_N):
            envs.append(spec.env_class.from_dict(cfg))

        observations = [_get_initial_obs(spec, env) for env in envs]

        # All observations must be identical
        first_obs = observations[0]
        assert first_obs, f"Empty observation for {spec.name}"
        for i, obs in enumerate(observations[1:], start=1):
            assert obs == first_obs, (
                f"{spec.name} seed={test_seed}: env[{i}] differs from env[0].\n"
                f"--- env[0] (first 300 chars) ---\n{first_obs[:300]}\n"
                f"--- env[{i}] (first 300 chars) ---\n{obs[:300]}"
            )
    finally:
        _close_all(envs, spec)


# ===========================================================================
# TEST 2: Same Instance Multiple Resets
# ===========================================================================

_NUM_RESETS = 5


@pytest.mark.parametrize("spec", _spec_params)
def test_same_instance_multiple_resets(spec: EnvSpec):
    """Resetting the same env K times must produce the same observation each time."""
    env = spec.env_class.from_dict(spec.from_dict_config)
    try:
        observations = [_get_initial_obs(spec, env) for _ in range(_NUM_RESETS)]

        first_obs = observations[0]
        assert first_obs, f"Empty observation for {spec.name}"
        for i, obs in enumerate(observations[1:], start=1):
            assert obs == first_obs, (
                f"{spec.name}: reset #{i+1} differs from reset #1.\n"
                f"--- reset #1 (first 300 chars) ---\n{first_obs[:300]}\n"
                f"--- reset #{i+1} (first 300 chars) ---\n{obs[:300]}"
            )
    finally:
        _close_env(env, spec)


# ===========================================================================
# TEST 3: from_dict vs Constructor
# ===========================================================================

@pytest.mark.parametrize("spec", _constructor_params)
def test_from_dict_vs_constructor(spec: EnvSpec):
    """from_dict and direct constructor with same params must produce same obs."""
    env_fd = spec.env_class.from_dict(spec.from_dict_config)
    env_ctor = spec.env_class(**spec.constructor_kwargs)
    try:
        obs_fd = _get_initial_obs(spec, env_fd)
        obs_ctor = _get_initial_obs(spec, env_ctor)

        assert obs_fd == obs_ctor, (
            f"{spec.name}: from_dict obs differs from constructor obs.\n"
            f"--- from_dict (first 300 chars) ---\n{obs_fd[:300]}\n"
            f"--- constructor (first 300 chars) ---\n{obs_ctor[:300]}"
        )
    finally:
        _close_env(env_fd, spec)
        _close_env(env_ctor, spec)


# ===========================================================================
# TEST 4: Concurrent Creation via ThreadPoolExecutor
# ===========================================================================

_CONCURRENT_N = 16
_CONCURRENT_SEEDS = [42, 123]


@pytest.mark.parametrize("spec", _spec_params)
@pytest.mark.parametrize("seed_idx", [0, 1], ids=["cseed42", "cseed123"])
def test_concurrent_creation_determinism(spec: EnvSpec, seed_idx: int):
    """Create N envs concurrently via ThreadPoolExecutor — mirrors init_envs_and_agents.

    This test is the most critical: it catches race conditions in global RNG
    state (the original Sokoban bug).
    """
    test_seed = _CONCURRENT_SEEDS[seed_idx]

    cfg = dict(spec.from_dict_config)
    if "seed" in cfg:
        cfg["seed"] = test_seed

    def _create_and_observe(idx: int) -> tuple[int, str]:
        env = spec.env_class.from_dict(cfg)
        try:
            obs = _get_initial_obs(spec, env)
            return idx, obs
        finally:
            _close_env(env, spec)

    results: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=min(_CONCURRENT_N, 16)) as executor:
        futures = [executor.submit(_create_and_observe, i) for i in range(_CONCURRENT_N)]
        for future in as_completed(futures):
            idx, obs = future.result()
            results[idx] = obs

    observations = [results[i] for i in range(_CONCURRENT_N)]
    first_obs = observations[0]
    assert first_obs, f"Empty observation for {spec.name}"

    mismatches = [i for i, obs in enumerate(observations) if obs != first_obs]
    assert not mismatches, (
        f"{spec.name} seed={test_seed}: {len(mismatches)}/{_CONCURRENT_N} concurrent envs "
        f"differ from env[0]. Indices: {mismatches[:10]}...\n"
        f"--- env[0] (first 300 chars) ---\n{first_obs[:300]}"
    )


# ===========================================================================
# TEST 5: Different Seeds/Tasks Produce Different Observations
# ===========================================================================

@pytest.mark.parametrize("spec", _spec_params)
def test_different_seeds_different_observations(spec: EnvSpec):
    """Sanity check: different configs must produce different initial observations."""
    env1 = spec.env_class.from_dict(spec.from_dict_config)
    env2 = spec.env_class.from_dict(spec.alt_config)
    try:
        obs1 = _get_initial_obs(spec, env1)
        obs2 = _get_initial_obs(spec, env2)

        assert obs1, f"Empty observation for {spec.name} (config 1)"
        assert obs2, f"Empty observation for {spec.name} (config 2)"
        assert obs1 != obs2, (
            f"{spec.name}: Different configs produced identical observations!\n"
            f"Config 1: {spec.from_dict_config}\n"
            f"Config 2: {spec.alt_config}\n"
            f"Obs (first 300 chars): {obs1[:300]}"
        )
    finally:
        _close_env(env1, spec)
        _close_env(env2, spec)


# ===========================================================================
# TEST 6: Post-Action Determinism
# ===========================================================================

@pytest.mark.parametrize("spec", _spec_params)
def test_post_action_determinism(spec: EnvSpec):
    """Two identically-seeded envs with the same action sequence must produce
    identical obs/reward/done at every step.

    Critical for Minesweeper where mines are placed on first reveal (deferred RNG).
    """
    env1 = spec.env_class.from_dict(spec.from_dict_config)
    env2 = spec.env_class.from_dict(spec.from_dict_config)
    try:
        obs1, _ = env1.reset()
        obs2, _ = env2.reset()

        actions = _get_action_sequence(spec, env1)
        if not actions:
            pytest.skip(f"No action sequence defined for {spec.name}")

        # Compare initial obs (before any action)
        obs1_norm = normalize_obs(obs1)
        obs2_norm = normalize_obs(obs2)
        assert obs1_norm == obs2_norm, (
            f"{spec.name}: Initial obs differ before any action.\n"
            f"--- env1 ---\n{obs1_norm[:300]}\n"
            f"--- env2 ---\n{obs2_norm[:300]}"
        )

        # Execute action sequence and compare at each step
        for step_idx, action in enumerate(actions):
            result1 = env1.step(action)
            result2 = env2.step(action)

            o1, r1, d1 = result1[0], result1[1], result1[2]
            o2, r2, d2 = result2[0], result2[1], result2[2]

            o1_norm = normalize_obs(o1)
            o2_norm = normalize_obs(o2)

            assert o1_norm == o2_norm, (
                f"{spec.name} step {step_idx} (action={action}): observations differ.\n"
                f"--- env1 ---\n{o1_norm[:300]}\n"
                f"--- env2 ---\n{o2_norm[:300]}"
            )
            assert r1 == r2, (
                f"{spec.name} step {step_idx} (action={action}): rewards differ. "
                f"env1={r1}, env2={r2}"
            )
            assert d1 == d2, (
                f"{spec.name} step {step_idx} (action={action}): done flags differ. "
                f"env1={d1}, env2={d2}"
            )

            if d1:
                break
    finally:
        _close_env(env1, spec)
        _close_env(env2, spec)
