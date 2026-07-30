"""Registry and mode-string contract tests for the CAST oracle advantage mapping.

These are guard-rail tests, not behavioral coverage of the mapping math. They
exist because ``oracle_diagnostics`` branches on the mode string, and a rename
that updates the registry but not the diagnostics branch would fail *silently*:
training keeps running, only the CAST-specific metrics quietly disappear or get
computed against the wrong transform. Nothing else in the repo catches that.
"""

import numpy as np
import pytest
import torch

from rllm.trainer.verl import oracle_diagnostics
from rllm.trainer.verl.oracle_advantage_mapping import (
    CAST_MODES,
    MODE_CAST,
    MODE_CAST_ABLATE,
    REGISTRY,
    apply_mapping,
)


def test_registry_exposes_exactly_the_cast_modes():
    """The public mode strings are part of the CLI/docs contract."""
    assert sorted(REGISTRY.keys()) == ["cast", "cast_ablate"]


def test_constants_agree_with_registry_keys():
    """Fails if a @register(...) call reverts to a literal that drifts."""
    assert MODE_CAST in REGISTRY
    assert MODE_CAST_ABLATE in REGISTRY
    assert set(CAST_MODES) == set(REGISTRY.keys())


def test_diagnostics_shares_the_mode_constants():
    """Identity, not equality: diagnostics must import the constants rather than
    keep its own copy, otherwise the two can drift apart again."""
    assert oracle_diagnostics.CAST_MODES is CAST_MODES
    assert oracle_diagnostics.MODE_CAST is MODE_CAST


def test_unknown_mode_raises_with_available_modes_listed():
    """A stale config value must fail loudly and say what is valid."""
    with pytest.raises(ValueError) as excinfo:
        apply_mapping(
            mode="v0",
            oracle_advs=torch.zeros(1, 2),
            response_mask=torch.ones(1, 2),
        )
    message = str(excinfo.value)
    assert "v0" in message
    assert "cast" in message


def _tiny_batch():
    """Two trajectories in one uid group, two turns each, 3 tokens per turn.

    response_mask uses a 0 gap between the two turns because the mapping
    recovers step boundaries from contiguous runs of 1s.
    """
    response_mask = torch.tensor(
        [
            [1, 1, 1, 0, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1],
        ],
        dtype=torch.float32,
    )
    token_level_scores = torch.zeros(2, 7)
    token_level_scores[0, -1] = 1.0  # trajectory 0 succeeds
    token_level_scores[1, -1] = 0.0  # trajectory 1 fails -> nonzero GRPO contrast
    turn_advantages = np.empty(2, dtype=object)
    turn_advantages[0] = np.array([1.0, -4.0])
    turn_advantages[1] = np.array([0.0, 1.0])
    return {
        "oracle_advs": torch.zeros(2, 7),
        "response_mask": response_mask,
        "uids": np.array(["g0", "g0"]),
        "token_level_scores": token_level_scores,
        "turn_advantages": turn_advantages,
        "rollout_n": 2,
    }


def test_cast_ablate_with_asinh_and_rms_matches_cast():
    """``mapping_cast_ablate`` is a deliberate copy of ``mapping_cast``.

    Its docstring promises numerical equivalence at the default knobs, and that
    promise is the one invariant the copy can silently break.
    """
    batch = _tiny_batch()

    production = apply_mapping(
        mode=MODE_CAST, config={"cast_alpha": 0.1, "cast_eps": 1e-8}, **batch
    )
    ablate = apply_mapping(
        mode=MODE_CAST_ABLATE,
        config={
            "cast_alpha": 0.1,
            "cast_eps": 1e-8,
            "cast_transform": "asinh",
            "cast_norm": "rms",
        },
        **batch,
    )

    assert torch.allclose(production, ablate, atol=1e-6)
    # Guard against the whole thing being trivially zero.
    assert production.abs().sum() > 0


def test_cast_ablate_identity_transform_differs_from_cast():
    """Sanity check the ablation knob actually does something."""
    batch = _tiny_batch()

    production = apply_mapping(
        mode=MODE_CAST, config={"cast_alpha": 0.1, "cast_eps": 1e-8}, **batch
    )
    identity = apply_mapping(
        mode=MODE_CAST_ABLATE,
        config={
            "cast_alpha": 0.1,
            "cast_eps": 1e-8,
            "cast_transform": "identity",
            "cast_norm": "rms",
        },
        **batch,
    )

    assert not torch.allclose(production, identity, atol=1e-6)


def test_cast_alpha_config_key_is_actually_read():
    """Pins the *name* of the alpha config key, not just its effect.

    Renaming the key without updating the shell scripts would not raise: both
    call sites would quietly fall back to the built-in default, and every run
    would train at the wrong alpha with no error anywhere. Two different alphas
    must produce two different outputs.
    """
    batch = _tiny_batch()

    low = apply_mapping(mode=MODE_CAST, config={"cast_alpha": 0.1}, **batch)
    high = apply_mapping(mode=MODE_CAST, config={"cast_alpha": 0.9}, **batch)

    assert not torch.allclose(low, high, atol=1e-6), (
        "cast_alpha had no effect — the config key is probably misspelled or "
        "renamed, so both calls silently used the default"
    )


def test_unknown_cast_norm_is_rejected():
    batch = _tiny_batch()
    with pytest.raises(ValueError, match="cast_norm"):
        apply_mapping(
            mode=MODE_CAST_ABLATE,
            config={"cast_transform": "asinh", "cast_norm": "bogus"},
            **batch,
        )
