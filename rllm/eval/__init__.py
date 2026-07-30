"""Unified evaluation portal for rLLM framework.

This module provides a unified interface for evaluating all environments
(sokoban, minesweeper, rush_hour, alfworld, webshop)
with a single command, reusing the existing AgentExecutionEngine infrastructure.

Includes logprob-based metrics for trajectory quality evaluation:
- Sequence-Level Perplexity (SLP)
- Token-Level Perplexity (TLP)
- Sequence-Level Entropy (SLE)
- Token-Level Entropy (TLE)
- Variance of Acc@K
- Step/Turn/Sequence level trajectory quality metrics

Supports multiple parallel strategies for offline logprob computation:
- naive: Single GPU with basic batching (default)
- accelerate: HuggingFace Accelerate for distributed inference
- data_parallel: PyTorch DataParallel for multi-GPU batch processing
- fsdp: Fully Sharded Data Parallel for memory-efficient large model inference
- vllm: vLLM for high-throughput inference
- deepspeed: DeepSpeed ZeRO for memory-efficient inference

Example usage:
    # CLI usage
    python -m rllm.eval.cli --env webshop --limit 100

    # Programmatic usage
    from rllm.eval import EvalPortal
    portal = EvalPortal({"env": "webshop", "data": {"limit": 100}})
    results = await portal.run()
    metrics = portal.compute_metrics()

    # Compute logprob-based metrics
    from rllm.eval import compute_all_logprob_metrics
    logprob_metrics = compute_all_logprob_metrics(results)

    # Use offline logprob computation with parallel strategies
    from rllm.eval.offline_logprob import (
        populate_offline_logprobs,
        OfflineLogprobConfig,
        ParallelStrategy,
        get_available_strategies,
        get_recommended_strategy,
        estimate_memory_requirements,
    )
"""

from rllm.eval.portal import EvalPortal
from rllm.eval.registry import EnvConfigRegistry, EnvConfig
from rllm.eval.metrics import (
    # Core metrics
    compute_metrics,
    compute_pass_at_k,
    compute_success_rate,
    compute_avg_reward,
    compute_score,
    compute_score_stats,
    compute_avg_steps,
    compute_completion_rate,
    compute_f1_score,
    compute_exact_match,
    compute_reward_distribution,
    compute_exploration_coverage,
    compute_world_modeling_completeness,
    # Logprob-based metrics (aggregated)
    compute_sequence_level_perplexity,
    compute_token_level_perplexity,
    compute_sequence_level_entropy,
    compute_token_level_entropy,
    compute_variance_of_acc_at_k,
    compute_step_level_metrics,
    compute_trajectory_confidence_metrics,
    compute_response_length_metrics,
    compute_all_logprob_metrics,
    # Detailed token/step/trajectory level metrics
    compute_detailed_trajectory_metrics,
    search_tokens_by_text,
    aggregate_token_metrics_by_text,
    get_high_entropy_tokens,
    get_low_confidence_tokens,
    compare_success_vs_failure_tokens,
)
from rllm.eval.offline_logprob import (
    populate_offline_logprobs,
    populate_offline_logprobs_with_config,
    OfflineLogprobConfig,
    ParallelStrategy,
    get_available_strategies,
    get_recommended_strategy,
    estimate_memory_requirements,
)

__all__ = [
    # Portal and Registry
    "EvalPortal",
    "EnvConfigRegistry",
    "EnvConfig",
    # Core metrics
    "compute_metrics",
    "compute_pass_at_k",
    "compute_success_rate",
    "compute_avg_reward",
    "compute_score",
    "compute_score_stats",
    "compute_avg_steps",
    "compute_completion_rate",
    "compute_f1_score",
    "compute_exact_match",
    "compute_reward_distribution",
    "compute_exploration_coverage",
    "compute_world_modeling_completeness",
    # Logprob-based metrics (aggregated)
    "compute_sequence_level_perplexity",
    "compute_token_level_perplexity",
    "compute_sequence_level_entropy",
    "compute_token_level_entropy",
    "compute_variance_of_acc_at_k",
    "compute_step_level_metrics",
    "compute_trajectory_confidence_metrics",
    "compute_response_length_metrics",
    "compute_all_logprob_metrics",
    # Detailed token/step/trajectory level metrics
    "compute_detailed_trajectory_metrics",
    "search_tokens_by_text",
    "aggregate_token_metrics_by_text",
    "get_high_entropy_tokens",
    "get_low_confidence_tokens",
    "compare_success_vs_failure_tokens",
    # Offline logprob computation
    "populate_offline_logprobs",
    "populate_offline_logprobs_with_config",
    "OfflineLogprobConfig",
    "ParallelStrategy",
    "get_available_strategies",
    "get_recommended_strategy",
    "estimate_memory_requirements",
]
