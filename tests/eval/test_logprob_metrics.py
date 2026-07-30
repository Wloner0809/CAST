"""Test logprob-based evaluation metrics.

This module tests the logprob-based metrics for trajectory quality evaluation:
- Sequence-Level Perplexity (SLP)
- Token-Level Perplexity (TLP)
- Sequence-Level Entropy (SLE)
- Token-Level Entropy (TLE)
- Variance of Acc@K
- Step-Level metrics
- Confidence metrics
- Response length metrics
"""

import math

import numpy as np
import pytest

from rllm.agents.agent import Step, Trajectory
from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm.eval.metrics import (
    aggregate_token_metrics_by_text,
    compare_success_vs_failure_tokens,
    compute_all_logprob_metrics,
    compute_detailed_trajectory_metrics,
    compute_metrics,
    compute_response_length_metrics,
    compute_sequence_level_entropy,
    compute_sequence_level_perplexity,
    compute_step_level_metrics,
    compute_token_level_entropy,
    compute_token_level_perplexity,
    compute_trajectory_confidence_metrics,
    compute_variance_of_acc_at_k,
    get_high_entropy_tokens,
    get_low_confidence_tokens,
    print_metrics,
    search_tokens_by_text,
)


def create_mock_trajectory(
    logprobs: list[list[float]] | None = None,
    reward: float = 0.0,
    task: dict | None = None,
) -> Trajectory:
    """Create a mock trajectory with specified logprobs.

    Args:
        logprobs: List of logprob lists, one per step. Each list contains
            token-level logprobs for that step.
        reward: Trajectory-level reward.
        task: Task dictionary for grouping by problem.

    Returns:
        Mock Trajectory object with steps containing the specified logprobs.
    """
    if logprobs is None:
        logprobs = []

    steps = []
    for step_logprobs in logprobs:
        step = Step(
            logprobs=step_logprobs,
            model_output=ModelOutput(
                text="test response",
                logprobs=step_logprobs,
            ),
        )
        steps.append(step)

    return Trajectory(
        task=task or {"problem_id": "test"},
        steps=steps,
        reward=reward,
    )


class TestSequenceLevelPerplexity:
    """Test class for Sequence-Level Perplexity (SLP) metrics."""

    def test_empty_results(self):
        """Test SLP with empty results."""
        metrics = compute_sequence_level_perplexity([])
        assert math.isnan(metrics["slp_mean"])
        assert metrics["trajectories_with_logprobs"] == 0

    def test_single_trajectory(self):
        """Test SLP with a single trajectory."""
        # Logprobs: -1.0, -2.0, -0.5 -> avg = -1.167 -> ppl = exp(1.167) ≈ 3.21
        trajectory = create_mock_trajectory(logprobs=[[-1.0, -2.0, -0.5]])
        metrics = compute_sequence_level_perplexity([trajectory])

        expected_avg = (-1.0 - 2.0 - 0.5) / 3
        expected_ppl = math.exp(-expected_avg)

        assert abs(metrics["slp_mean"] - expected_ppl) < 0.01
        assert metrics["trajectories_with_logprobs"] == 1

    def test_multiple_trajectories(self):
        """Test SLP with multiple trajectories."""
        traj1 = create_mock_trajectory(logprobs=[[-1.0, -1.0]])  # avg=-1.0, ppl=e^1
        traj2 = create_mock_trajectory(logprobs=[[-2.0, -2.0]])  # avg=-2.0, ppl=e^2

        metrics = compute_sequence_level_perplexity([traj1, traj2])

        ppl1 = math.exp(1.0)
        ppl2 = math.exp(2.0)
        expected_mean = (ppl1 + ppl2) / 2

        assert abs(metrics["slp_mean"] - expected_mean) < 0.01
        assert metrics["trajectories_with_logprobs"] == 2

    def test_multi_step_trajectory(self):
        """Test SLP with a trajectory having multiple steps."""
        # Step 1: [-1.0, -1.0], Step 2: [-2.0, -2.0]
        # All logprobs: [-1.0, -1.0, -2.0, -2.0] -> avg = -1.5 -> ppl = e^1.5
        trajectory = create_mock_trajectory(logprobs=[[-1.0, -1.0], [-2.0, -2.0]])
        metrics = compute_sequence_level_perplexity([trajectory])

        expected_ppl = math.exp(1.5)
        assert abs(metrics["slp_mean"] - expected_ppl) < 0.01

    def test_trajectory_without_logprobs(self):
        """Test SLP with trajectory having no logprobs."""
        trajectory = create_mock_trajectory(logprobs=[[]])
        metrics = compute_sequence_level_perplexity([trajectory])
        assert math.isnan(metrics["slp_mean"])


class TestTokenLevelPerplexity:
    """Test class for Token-Level Perplexity (TLP) metrics."""

    def test_empty_results(self):
        """Test TLP with empty results."""
        metrics = compute_token_level_perplexity([])
        assert math.isnan(metrics["tlp_mean"])
        assert metrics["total_tokens"] == 0

    def test_single_trajectory(self):
        """Test TLP with a single trajectory."""
        # Logprobs: -1.0, -2.0 -> per-token ppls: e^1, e^2
        trajectory = create_mock_trajectory(logprobs=[[-1.0, -2.0]])
        metrics = compute_token_level_perplexity([trajectory])

        ppl1 = math.exp(1.0)
        ppl2 = math.exp(2.0)
        expected_mean = (ppl1 + ppl2) / 2

        assert abs(metrics["tlp_mean"] - expected_mean) < 0.01
        assert metrics["total_tokens"] == 2

    def test_percentiles(self):
        """Test TLP percentile calculations."""
        # Create trajectory with known distribution
        logprobs = [[-i * 0.1 for i in range(100)]]
        trajectory = create_mock_trajectory(logprobs=logprobs)
        metrics = compute_token_level_perplexity([trajectory])

        assert metrics["tlp_p50"] > metrics["tlp_p25"]
        assert metrics["tlp_p75"] > metrics["tlp_p50"]
        assert metrics["tlp_p95"] > metrics["tlp_p75"]


class TestSequenceLevelEntropy:
    """Test class for Sequence-Level Entropy (SLE) metrics."""

    def test_empty_results(self):
        """Test SLE with empty results."""
        metrics = compute_sequence_level_entropy([])
        assert math.isnan(metrics["sle_mean"])

    def test_single_trajectory(self):
        """Test SLE with a single trajectory."""
        # Logprobs: -1.0, -2.0 -> avg entropy = (1.0 + 2.0) / 2 = 1.5
        trajectory = create_mock_trajectory(logprobs=[[-1.0, -2.0]])
        metrics = compute_sequence_level_entropy([trajectory])

        expected_entropy = 1.5
        assert abs(metrics["sle_mean"] - expected_entropy) < 0.01

    def test_entropy_perplexity_relationship(self):
        """Test that entropy = log(perplexity)."""
        trajectory = create_mock_trajectory(logprobs=[[-1.5, -2.5, -0.5]])

        sle_metrics = compute_sequence_level_entropy([trajectory])
        slp_metrics = compute_sequence_level_perplexity([trajectory])

        # log(perplexity) should equal entropy
        expected_entropy = math.log(slp_metrics["slp_mean"])
        assert abs(sle_metrics["sle_mean"] - expected_entropy) < 0.01


class TestTokenLevelEntropy:
    """Test class for Token-Level Entropy (TLE) metrics."""

    def test_empty_results(self):
        """Test TLE with empty results."""
        metrics = compute_token_level_entropy([])
        assert math.isnan(metrics["tle_mean"])
        assert metrics["total_tokens"] == 0

    def test_single_trajectory(self):
        """Test TLE with a single trajectory."""
        # Logprobs: -1.0, -2.0 -> entropies: 1.0, 2.0 -> avg = 1.5
        trajectory = create_mock_trajectory(logprobs=[[-1.0, -2.0]])
        metrics = compute_token_level_entropy([trajectory])

        expected_mean = 1.5
        assert abs(metrics["tle_mean"] - expected_mean) < 0.01
        assert metrics["total_tokens"] == 2


class TestVarianceOfAccAtK:
    """Test class for Variance of Acc@K metrics."""

    def test_empty_results(self):
        """Test variance with empty results."""
        metrics = compute_variance_of_acc_at_k([])
        assert math.isnan(metrics["acc_at_k_variance"])

    def test_single_rollout_per_problem(self):
        """Test variance with single rollout per problem."""
        traj1 = create_mock_trajectory(task={"id": "1"}, reward=1.0)
        traj2 = create_mock_trajectory(task={"id": "2"}, reward=0.0)
        traj3 = create_mock_trajectory(task={"id": "3"}, reward=1.0)

        metrics = compute_variance_of_acc_at_k([traj1, traj2, traj3])

        # Per-problem accuracies: [1.0, 0.0, 1.0]
        # Mean = 2/3, Variance = ((1-2/3)^2 + (0-2/3)^2 + (1-2/3)^2) / 3
        assert metrics["total_problems"] == 3
        assert metrics["total_rollouts"] == 3
        assert abs(metrics["per_problem_accuracy_mean"] - 2 / 3) < 0.01

    def test_multiple_rollouts_per_problem(self):
        """Test variance with multiple rollouts per problem."""
        # Problem 1: 2 successes out of 3
        traj1a = create_mock_trajectory(task={"id": "1"}, reward=1.0)
        traj1b = create_mock_trajectory(task={"id": "1"}, reward=1.0)
        traj1c = create_mock_trajectory(task={"id": "1"}, reward=0.0)

        # Problem 2: 1 success out of 3
        traj2a = create_mock_trajectory(task={"id": "2"}, reward=0.0)
        traj2b = create_mock_trajectory(task={"id": "2"}, reward=1.0)
        traj2c = create_mock_trajectory(task={"id": "2"}, reward=0.0)

        trajectories = [traj1a, traj1b, traj1c, traj2a, traj2b, traj2c]
        metrics = compute_variance_of_acc_at_k(trajectories)

        assert metrics["total_problems"] == 2
        assert metrics["total_rollouts"] == 6
        assert abs(metrics["avg_rollouts_per_problem"] - 3.0) < 0.01

        # Per-problem accuracies: [2/3, 1/3]
        expected_mean = (2 / 3 + 1 / 3) / 2
        assert abs(metrics["per_problem_accuracy_mean"] - expected_mean) < 0.01

    def test_perfect_consistency(self):
        """Test variance when all problems have same accuracy."""
        # All problems succeed
        traj1 = create_mock_trajectory(task={"id": "1"}, reward=1.0)
        traj2 = create_mock_trajectory(task={"id": "2"}, reward=1.0)

        metrics = compute_variance_of_acc_at_k([traj1, traj2])

        # All accuracies are 1.0, variance should be 0
        assert abs(metrics["acc_at_k_variance"]) < 0.01


class TestStepLevelMetrics:
    """Test class for Step-Level metrics."""

    def test_empty_results(self):
        """Test step metrics with empty results."""
        metrics = compute_step_level_metrics([])
        assert math.isnan(metrics["step_ppl_mean"])
        assert metrics["total_steps"] == 0

    def test_single_step(self):
        """Test step metrics with single step."""
        trajectory = create_mock_trajectory(logprobs=[[-1.0, -1.0]])
        metrics = compute_step_level_metrics([trajectory])

        expected_entropy = 1.0
        expected_ppl = math.exp(1.0)

        assert abs(metrics["step_entropy_mean"] - expected_entropy) < 0.01
        assert abs(metrics["step_ppl_mean"] - expected_ppl) < 0.01
        assert metrics["total_steps"] == 1

    def test_confidence_progression(self):
        """Test confidence progression across steps."""
        # Create trajectory where later steps have higher entropy (lower confidence)
        trajectory = create_mock_trajectory(
            logprobs=[
                [-0.5, -0.5],  # Step 0: low entropy
                [-2.0, -2.0],  # Step 1: high entropy
            ]
        )
        metrics = compute_step_level_metrics([trajectory])

        progression = metrics["confidence_progression"]
        assert len(progression) == 2

        # Step 0 should have higher confidence than Step 1
        assert progression[0]["avg_confidence"] > progression[1]["avg_confidence"]


class TestTrajectoryConfidenceMetrics:
    """Test class for trajectory confidence metrics."""

    def test_empty_results(self):
        """Test confidence with empty results."""
        metrics = compute_trajectory_confidence_metrics([])
        assert math.isnan(metrics["avg_confidence_successful"])

    def test_successful_vs_failed(self):
        """Test confidence difference between successful and failed trajectories."""
        # Successful trajectory with high confidence (low entropy)
        traj_success = create_mock_trajectory(
            logprobs=[[-0.5, -0.5]],  # low entropy = high confidence
            reward=1.0,
        )

        # Failed trajectory with low confidence (high entropy)
        traj_failed = create_mock_trajectory(
            logprobs=[[-3.0, -3.0]],  # high entropy = low confidence
            reward=0.0,
        )

        metrics = compute_trajectory_confidence_metrics([traj_success, traj_failed])

        assert metrics["avg_confidence_successful"] > metrics["avg_confidence_failed"]
        assert metrics["confidence_gap"] > 0

    def test_high_low_confidence_accuracy(self):
        """Test accuracy for high vs low confidence trajectories."""
        trajectories = [
            create_mock_trajectory(logprobs=[[-0.5]], reward=1.0),  # high conf, success
            create_mock_trajectory(logprobs=[[-0.6]], reward=1.0),  # high conf, success
            create_mock_trajectory(logprobs=[[-3.0]], reward=0.0),  # low conf, fail
            create_mock_trajectory(logprobs=[[-3.5]], reward=0.0),  # low conf, fail
        ]

        metrics = compute_trajectory_confidence_metrics(trajectories)

        # High confidence trajectories are successful, low confidence ones fail
        assert metrics["high_confidence_accuracy"] == 1.0
        assert metrics["low_confidence_accuracy"] == 0.0


class TestResponseLengthMetrics:
    """Test class for response length metrics."""

    def test_empty_results(self):
        """Test length metrics with empty results."""
        metrics = compute_response_length_metrics([])
        assert math.isnan(metrics["avg_trajectory_tokens"])

    def test_token_counting(self):
        """Test token counting across trajectories."""
        traj1 = create_mock_trajectory(logprobs=[[-1.0, -1.0], [-1.0]])  # 3 tokens
        traj2 = create_mock_trajectory(logprobs=[[-1.0, -1.0, -1.0, -1.0]])  # 4 tokens

        metrics = compute_response_length_metrics([traj1, traj2])

        assert metrics["total_tokens"] == 7
        assert abs(metrics["avg_trajectory_tokens"] - 3.5) < 0.01
        assert metrics["total_steps"] == 3  # 2 + 1 steps


class TestComputeAllLogprobMetrics:
    """Test class for the aggregate logprob metrics function."""

    def test_empty_results(self):
        """Test all metrics with empty results."""
        metrics = compute_all_logprob_metrics([])

        # Should contain all metric groups
        assert any(k.startswith("slp/") for k in metrics)
        assert any(k.startswith("tlp/") for k in metrics)
        assert any(k.startswith("sle/") for k in metrics)
        assert any(k.startswith("tle/") for k in metrics)
        assert any(k.startswith("acc_variance/") for k in metrics)
        assert any(k.startswith("step/") for k in metrics)
        assert any(k.startswith("confidence/") for k in metrics)
        assert any(k.startswith("length/") for k in metrics)

    def test_with_trajectories(self):
        """Test all metrics with actual trajectories."""
        trajectories = [
            create_mock_trajectory(
                logprobs=[[-1.0, -1.5], [-2.0]],
                reward=1.0,
                task={"id": "1"},
            ),
            create_mock_trajectory(
                logprobs=[[-0.5, -0.8]],
                reward=0.0,
                task={"id": "2"},
            ),
        ]

        metrics = compute_all_logprob_metrics(trajectories)

        # Check that metrics are computed
        assert not math.isnan(metrics["slp/slp_mean"])
        assert not math.isnan(metrics["tlp/tlp_mean"])
        assert not math.isnan(metrics["sle/sle_mean"])
        assert not math.isnan(metrics["tle/tle_mean"])


class TestComputeMetricsWithLogprobs:
    """Test compute_metrics with logprob metrics enabled."""

    def test_include_logprob_metrics(self):
        """Test compute_metrics with include_logprob_metrics flag."""
        trajectories = [
            create_mock_trajectory(
                logprobs=[[-1.0, -1.0]],
                reward=1.0,
            ),
        ]

        metrics = compute_metrics(
            trajectories,
            metrics_list=["success_rate"],
            include_logprob_metrics=True,
        )

        assert "success_rate" in metrics
        assert any(k.startswith("slp/") for k in metrics)

    def test_explicit_logprob_metrics(self):
        """Test compute_metrics with explicit logprob metric requests."""
        trajectories = [
            create_mock_trajectory(logprobs=[[-1.0, -1.0]], reward=1.0),
        ]

        metrics = compute_metrics(
            trajectories,
            metrics_list=["sequence_level_perplexity", "token_level_entropy"],
        )

        assert "sequence_level_perplexity/slp_mean" in metrics
        assert "token_level_entropy/tle_mean" in metrics


class TestPrintMetrics:
    """Test print_metrics function."""

    def test_print_basic_metrics(self, capsys):
        """Test printing basic metrics."""
        metrics = {
            "success_rate": 0.75,
            "avg_reward": 0.5,
        }
        print_metrics(metrics, title="TEST RESULTS")

        captured = capsys.readouterr()
        assert "TEST RESULTS" in captured.out
        assert "success_rate" in captured.out
        assert "0.7500" in captured.out

    def test_print_grouped_metrics(self, capsys):
        """Test printing grouped metrics with prefixes."""
        metrics = {
            "slp/slp_mean": 2.5,
            "slp/slp_std": 0.3,
            "tle/tle_mean": 1.5,
        }
        print_metrics(metrics, title="LOGPROB METRICS")

        captured = capsys.readouterr()
        assert "[SLP]" in captured.out
        assert "[TLE]" in captured.out
        assert "2.5000" in captured.out

    def test_print_nan_values(self, capsys):
        """Test printing NaN values."""
        metrics = {
            "test_metric": float("nan"),
        }
        print_metrics(metrics)

        captured = capsys.readouterr()
        assert "N/A" in captured.out


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_nan_logprobs_filtered(self):
        """Test that NaN logprobs are properly filtered."""
        trajectory = create_mock_trajectory(logprobs=[[float("nan"), -1.0, float("nan")]])
        metrics = compute_sequence_level_perplexity([trajectory])

        # Should only use the valid logprob (-1.0)
        expected_ppl = math.exp(1.0)
        assert abs(metrics["slp_mean"] - expected_ppl) < 0.01

    def test_very_large_logprobs(self):
        """Test handling of very large negative logprobs."""
        trajectory = create_mock_trajectory(logprobs=[[-100.0, -100.0]])
        metrics = compute_sequence_level_perplexity([trajectory])

        # Should handle large values without overflow
        assert not math.isinf(metrics["slp_mean"])
        assert metrics["slp_mean"] > 0

    def test_zero_logprobs(self):
        """Test handling of zero logprobs (probability 1)."""
        trajectory = create_mock_trajectory(logprobs=[[0.0, 0.0]])
        metrics = compute_sequence_level_perplexity([trajectory])

        # ppl = exp(0) = 1
        assert abs(metrics["slp_mean"] - 1.0) < 0.01

    def test_mixed_trajectory_types(self):
        """Test with mix of trajectories with and without logprobs."""
        traj_with_logprobs = create_mock_trajectory(logprobs=[[-1.0, -1.0]])
        traj_without_logprobs = create_mock_trajectory(logprobs=[[]])

        metrics = compute_sequence_level_perplexity([traj_with_logprobs, traj_without_logprobs])

        # Should only compute for trajectory with logprobs
        assert metrics["trajectories_with_logprobs"] == 1


# =============================================================================
# Tests for Detailed Token/Step/Trajectory Level Metrics
# =============================================================================


class TestDetailedTrajectoryMetrics:
    """Test class for detailed trajectory metrics."""

    def test_empty_results(self):
        """Test detailed metrics with empty results."""
        detailed = compute_detailed_trajectory_metrics([])
        assert detailed == []

    def test_basic_structure(self):
        """Test that detailed metrics have correct structure."""
        trajectory = create_mock_trajectory(
            logprobs=[[-1.0, -2.0]],
            reward=1.0,
            task={"id": "test_task"},
        )
        detailed = compute_detailed_trajectory_metrics([trajectory])

        assert len(detailed) == 1
        traj_data = detailed[0]

        # Check top-level keys
        assert "trajectory_id" in traj_data
        assert "trajectory_idx" in traj_data
        assert "task" in traj_data
        assert "reward" in traj_data
        assert "is_success" in traj_data
        assert "sequence_metrics" in traj_data
        assert "steps" in traj_data

        # Check sequence metrics
        seq_metrics = traj_data["sequence_metrics"]
        assert "perplexity" in seq_metrics
        assert "entropy_mean" in seq_metrics
        assert "num_tokens" in seq_metrics

    def test_step_structure(self):
        """Test that step data has correct structure."""
        trajectory = create_mock_trajectory(
            logprobs=[[-1.0, -2.0]],
            reward=1.0,
        )
        detailed = compute_detailed_trajectory_metrics([trajectory])

        step_data = detailed[0]["steps"][0]

        assert "step_idx" in step_data
        assert "step_metrics" in step_data
        assert "tokens" in step_data

    def test_token_metrics(self):
        """Test that token-level metrics are computed correctly."""
        trajectory = create_mock_trajectory(logprobs=[[-1.0, -2.0]])
        detailed = compute_detailed_trajectory_metrics([trajectory])

        tokens = detailed[0]["steps"][0]["tokens"]
        assert len(tokens) == 2

        # Check first token
        token1 = tokens[0]
        assert token1["logprob"] == -1.0
        assert token1["entropy"] == 1.0  # -logprob
        assert abs(token1["perplexity"] - math.exp(1.0)) < 0.01
        assert "confidence" in token1

    def test_multiple_steps(self):
        """Test detailed metrics with multiple steps."""
        trajectory = create_mock_trajectory(
            logprobs=[[-1.0, -1.0], [-2.0, -2.0]],
            reward=1.0,
        )
        detailed = compute_detailed_trajectory_metrics([trajectory])

        assert len(detailed[0]["steps"]) == 2

        # Check sequence metrics include all tokens
        assert detailed[0]["sequence_metrics"]["num_tokens"] == 4


class TestSearchTokensByText:
    """Test class for token search functionality."""

    def test_search_with_mock_tokenizer(self):
        """Test token search with matching tokens."""
        # Create detailed results with token text
        detailed_results = [{
            "trajectory_id": "traj_1",
            "reward": 1.0,
            "is_success": True,
            "steps": [{
                "step_idx": 0,
                "tokens": [
                    {"token_idx": 0, "token_text": "Hello", "logprob": -1.0, "entropy": 1.0, "perplexity": 2.7, "confidence": 0.5},
                    {"token_idx": 1, "token_text": " Wait", "logprob": -2.0, "entropy": 2.0, "perplexity": 7.4, "confidence": 0.33},
                    {"token_idx": 2, "token_text": " for", "logprob": -0.5, "entropy": 0.5, "perplexity": 1.6, "confidence": 0.67},
                ],
            }],
        }]

        matches = search_tokens_by_text(detailed_results, "Wait")
        assert len(matches) == 1
        assert matches[0]["token_text"] == " Wait"
        assert matches[0]["entropy"] == 2.0

    def test_case_insensitive_search(self):
        """Test case-insensitive token search."""
        detailed_results = [{
            "trajectory_id": "traj_1",
            "reward": 1.0,
            "is_success": True,
            "steps": [{
                "step_idx": 0,
                "tokens": [
                    {"token_idx": 0, "token_text": "WAIT", "logprob": -1.0, "entropy": 1.0, "perplexity": 2.7, "confidence": 0.5},
                ],
            }],
        }]

        matches = search_tokens_by_text(detailed_results, "wait", case_sensitive=False)
        assert len(matches) == 1

    def test_no_matches(self):
        """Test search with no matches."""
        detailed_results = [{
            "trajectory_id": "traj_1",
            "reward": 1.0,
            "is_success": True,
            "steps": [{
                "step_idx": 0,
                "tokens": [
                    {"token_idx": 0, "token_text": "Hello", "logprob": -1.0, "entropy": 1.0, "perplexity": 2.7, "confidence": 0.5},
                ],
            }],
        }]

        matches = search_tokens_by_text(detailed_results, "NotFound")
        assert len(matches) == 0


class TestAggregateTokenMetrics:
    """Test class for token aggregation functionality."""

    def test_aggregate_by_text(self):
        """Test token aggregation by text."""
        detailed_results = [{
            "trajectory_id": "traj_1",
            "reward": 1.0,
            "is_success": True,
            "steps": [{
                "step_idx": 0,
                "tokens": [
                    {"token_idx": 0, "token_text": "Hello", "logprob": -1.0, "entropy": 1.0, "perplexity": 2.7, "confidence": 0.5},
                    {"token_idx": 1, "token_text": "Hello", "logprob": -2.0, "entropy": 2.0, "perplexity": 7.4, "confidence": 0.33},
                ],
            }],
        }]

        aggregated = aggregate_token_metrics_by_text(detailed_results)

        assert "Hello" in aggregated
        assert aggregated["Hello"]["count"] == 2
        assert abs(aggregated["Hello"]["entropy_mean"] - 1.5) < 0.01  # (1.0 + 2.0) / 2

    def test_success_rate_tracking(self):
        """Test that success rate is tracked correctly."""
        detailed_results = [
            {
                "trajectory_id": "traj_1",
                "reward": 1.0,
                "is_success": True,
                "steps": [{"step_idx": 0, "tokens": [
                    {"token_idx": 0, "token_text": "test", "logprob": -1.0, "entropy": 1.0, "perplexity": 2.7, "confidence": 0.5},
                ]}],
            },
            {
                "trajectory_id": "traj_2",
                "reward": 0.0,
                "is_success": False,
                "steps": [{"step_idx": 0, "tokens": [
                    {"token_idx": 0, "token_text": "test", "logprob": -1.0, "entropy": 1.0, "perplexity": 2.7, "confidence": 0.5},
                ]}],
            },
        ]

        aggregated = aggregate_token_metrics_by_text(detailed_results)

        assert aggregated["test"]["count"] == 2
        assert aggregated["test"]["success_rate"] == 0.5


class TestHighEntropyTokens:
    """Test class for high entropy token detection."""

    def test_get_high_entropy_tokens(self):
        """Test getting high entropy tokens."""
        detailed_results = [{
            "trajectory_id": "traj_1",
            "reward": 1.0,
            "is_success": True,
            "steps": [{
                "step_idx": 0,
                "tokens": [
                    {"token_idx": 0, "token_text": "low", "logprob": -0.1, "entropy": 0.1, "perplexity": 1.1, "confidence": 0.9},
                    {"token_idx": 1, "token_text": "high", "logprob": -5.0, "entropy": 5.0, "perplexity": 148.4, "confidence": 0.17},
                ],
            }],
        }]

        high_entropy = get_high_entropy_tokens(detailed_results, top_k=1)
        assert len(high_entropy) == 1
        assert high_entropy[0]["token_text"] == "high"
        assert high_entropy[0]["entropy"] == 5.0

    def test_threshold_filtering(self):
        """Test threshold-based filtering."""
        detailed_results = [{
            "trajectory_id": "traj_1",
            "reward": 1.0,
            "is_success": True,
            "steps": [{
                "step_idx": 0,
                "tokens": [
                    {"token_idx": 0, "logprob": -0.1, "entropy": 0.1, "perplexity": 1.1, "confidence": 0.9},
                    {"token_idx": 1, "logprob": -3.0, "entropy": 3.0, "perplexity": 20.1, "confidence": 0.25},
                    {"token_idx": 2, "logprob": -5.0, "entropy": 5.0, "perplexity": 148.4, "confidence": 0.17},
                ],
            }],
        }]

        high_entropy = get_high_entropy_tokens(detailed_results, threshold=2.5)
        assert len(high_entropy) == 2  # entropy >= 2.5


class TestLowConfidenceTokens:
    """Test class for low confidence token detection."""

    def test_get_low_confidence_tokens(self):
        """Test getting low confidence tokens."""
        detailed_results = [{
            "trajectory_id": "traj_1",
            "reward": 1.0,
            "is_success": True,
            "steps": [{
                "step_idx": 0,
                "tokens": [
                    {"token_idx": 0, "logprob": -0.1, "entropy": 0.1, "perplexity": 1.1, "confidence": 0.9},
                    {"token_idx": 1, "logprob": -5.0, "entropy": 5.0, "perplexity": 148.4, "confidence": 0.17},
                ],
            }],
        }]

        low_conf = get_low_confidence_tokens(detailed_results, threshold=0.5)
        assert len(low_conf) == 1
        assert low_conf[0]["confidence"] == 0.17


class TestSuccessFailureComparison:
    """Test class for success vs failure comparison."""

    def test_comparison(self):
        """Test success vs failure comparison."""
        detailed_results = [
            {
                "trajectory_id": "traj_1",
                "reward": 1.0,
                "is_success": True,
                "steps": [{"step_idx": 0, "tokens": [
                    {"logprob": -0.5, "entropy": 0.5, "perplexity": 1.6, "confidence": 0.67},
                ]}],
            },
            {
                "trajectory_id": "traj_2",
                "reward": 0.0,
                "is_success": False,
                "steps": [{"step_idx": 0, "tokens": [
                    {"logprob": -2.0, "entropy": 2.0, "perplexity": 7.4, "confidence": 0.33},
                ]}],
            },
        ]

        comparison = compare_success_vs_failure_tokens(detailed_results)

        assert comparison["success"]["count"] == 1
        assert comparison["failure"]["count"] == 1
        assert comparison["success"]["entropy_mean"] < comparison["failure"]["entropy_mean"]
        assert comparison["entropy_diff"] < 0  # success has lower entropy
