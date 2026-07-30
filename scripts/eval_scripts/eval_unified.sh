#!/usr/bin/env bash
# Single-environment evaluation script for rllm.
#
# Runs ONE (model, env) evaluation by calling `python3 -m rllm.eval.cli`
# directly (no intermediate shell layer).
#
# For batch runs over multiple (model x env) combinations, use
# eval_batch.sh, which calls this script once per combination.
#
# Usage:
#   bash eval_unified.sh <model_path> <config_yaml> [output_dir] [-- extra_cli_args...]
#
# Environment variables:
#   MODEL_PATH        Model path      (alternative to 1st positional arg)
#   EVAL_CONFIG       Config YAML     (alternative to 2nd positional arg)
#   BASE_OUTPUT_DIR   Output root     (alternative to 3rd positional arg)
#   N_ROLLOUTS        Rollouts per prompt        (default: 16)
#   N_PARALLEL        Parallel agents            (default: 64)
#   SPLIT             Data split                 (default: test)
#   LIMIT             Max examples (empty = all)
#   LOGPROB_METRICS   Enable logprob metrics 1/0 (default: 0)
#   DETAILED_METRICS  Enable detailed metrics 1/0(default: 0)
#   DIFFICULTY        Dataset difficulty label for output path (auto-detected from data.data_file if unset)
#   EXTRA_OVERRIDES   Space-separated KEY=VALUE --set pairs
#   EXTRA_CLI_ARGS    Extra CLI flags forwarded verbatim
#
# Examples:
#   # Single evaluation
#   bash eval_unified.sh /path/to/model rllm/eval/config/sokoban.yaml
#
#   # With output dir and overrides
#   EXTRA_OVERRIDES="sampling.temperature=0.6" \
#   bash eval_unified.sh /path/to/model rllm/eval/config/sokoban.yaml /path/to/output
#
#   # Pass arbitrary CLI args after --
#   bash eval_unified.sh /path/to/model rllm/eval/config/sokoban.yaml /path/to/output -- --max-steps 200

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../launch_scripts/lib/agentrl_launch.sh"
agentrl::maybe_enable_xtrace

export TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d%H%M%S)}"
export RLLM_RUN_ID="${RLLM_RUN_ID:-eval-${TIMESTAMP}}"

##############################
# Part 1) Parse positional args
##############################
# arg1: model_path, arg2: config_yaml, arg3: output_dir (optional)
if [[ $# -gt 0 && -n "${1:-}" && "${1:-}" != "--" ]]; then
    MODEL_PATH="${1}"; shift
else
    MODEL_PATH="${MODEL_PATH:-}"
fi

if [[ $# -gt 0 && -n "${1:-}" && "${1:-}" != "--" ]]; then
    EVAL_CONFIG="${1}"; shift
else
    EVAL_CONFIG="${EVAL_CONFIG:-}"
fi

if [[ $# -gt 0 && -n "${1:-}" && "${1:-}" != "--" ]]; then
    BASE_OUTPUT_DIR="${1}"; shift
else
    BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-}"
fi

# Everything after "--" is forwarded to cli.py verbatim
PASSTHROUGH_ARGS=()
if [[ "${1:-}" == "--" ]]; then
    shift
    PASSTHROUGH_ARGS=("$@")
fi

[[ -n "${MODEL_PATH}" ]] || agentrl::die "MODEL_PATH is required (1st arg or env)"
[[ -n "${EVAL_CONFIG}" ]] || agentrl::die "EVAL_CONFIG is required (2nd arg or env)"

MODEL_BASENAME="$(basename "${MODEL_PATH}")"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-eval-${MODEL_BASENAME}}"
export MODEL_NAME="${MODEL_NAME:-${MODEL_BASENAME}}"

##############################
# Part 2) Environment setup (reusable)
##############################
agentrl::detect_gpus
agentrl::conda_activate
agentrl::pip_install_editable_if_requested
agentrl::setup_python_and_system_env
agentrl::ray_cleanup
agentrl::ray_defaults

RLLM_DIR="$(agentrl::resolve_repo_root)"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-${RLLM_DIR}/outputs/eval}"

agentrl::log "RLLM_DIR: %s" "${RLLM_DIR}"
agentrl::log "MODEL_PATH: %s" "${MODEL_PATH}"
agentrl::log "MODEL_BASENAME: %s" "${MODEL_BASENAME}"
agentrl::log "EVAL_CONFIG: %s" "${EVAL_CONFIG}"
agentrl::log "BASE_OUTPUT_DIR: %s" "${BASE_OUTPUT_DIR}"

##############################
# Part 3) Evaluation parameters
##############################
N_ROLLOUTS="${N_ROLLOUTS:-16}"
N_PARALLEL="${N_PARALLEL:-64}"
SPLIT="${SPLIT:-test}"
LIMIT="${LIMIT:-}"
LOGPROB_METRICS="${LOGPROB_METRICS:-}"
DETAILED_METRICS="${DETAILED_METRICS:-}"

# Arbitrary YAML overrides via --set (space-separated KEY=VALUE pairs)
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-sglang.tp_size=1 sglang.dp_size=8}"
# Extra CLI args forwarded verbatim (space-separated)
EXTRA_CLI_ARGS="${EXTRA_CLI_ARGS:-}"

# --- Difficulty detection for output path ---
# Priority: DIFFICULTY env var > auto-extract from data.data_file in EXTRA_OVERRIDES > "default"
if [[ -z "${DIFFICULTY:-}" ]]; then
    # Try to extract difficulty from data.data_file=<difficulty>/<split> in EXTRA_OVERRIDES
    # e.g. data.data_file=id/test -> DIFFICULTY=id
    DIFFICULTY=""
    for override in ${EXTRA_OVERRIDES}; do
        if [[ "${override}" == data.data_file=* ]]; then
            _data_file_val="${override#data.data_file=}"
            # Extract the first path component as difficulty (e.g. "id/test" -> "id")
            if [[ "${_data_file_val}" == */* ]]; then
                DIFFICULTY="${_data_file_val%%/*}"
            fi
            break
        fi
    done
    DIFFICULTY="${DIFFICULTY:-default}"
fi
agentrl::log "DIFFICULTY: %s" "${DIFFICULTY}"

##############################
# Part 4) Environment-specific setup
##############################

# Detect env name from config path (used for output paths and logging)
CONFIG_BASENAME="$(basename "${EVAL_CONFIG}" .yaml)"

##############################
# Part 5) Build and run evaluation command
##############################
cd "${RLLM_DIR}"

# Derive output subdir: {base}/{config}/{difficulty}/{model}
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${CONFIG_BASENAME}/${DIFFICULTY}/${MODEL_BASENAME}"

EVAL_CMD=(
    python3 -m rllm.eval.cli
    --config "${EVAL_CONFIG}"
    --model "${MODEL_PATH}"
    --output-dir "${OUTPUT_DIR}"
    --n-rollouts "${N_ROLLOUTS}"
    --n-parallel "${N_PARALLEL}"
    --split "${SPLIT}"
)

[[ -n "${LIMIT}" ]] && EVAL_CMD+=(--limit "${LIMIT}")
[[ "${LOGPROB_METRICS}" == "1" ]] && EVAL_CMD+=(--logprob-metrics)
[[ "${DETAILED_METRICS}" == "1" ]] && EVAL_CMD+=(--detailed-metrics)

# Apply arbitrary YAML overrides via --set
if [[ -n "${EXTRA_OVERRIDES}" ]]; then
    for override in ${EXTRA_OVERRIDES}; do
        EVAL_CMD+=(--set "${override}")
    done
fi

# Forward EXTRA_CLI_ARGS
if [[ -n "${EXTRA_CLI_ARGS}" ]]; then
    # shellcheck disable=SC2206
    EVAL_CMD+=(${EXTRA_CLI_ARGS})
fi

# Forward passthrough args (everything after --)
if [[ ${#PASSTHROUGH_ARGS[@]} -gt 0 ]]; then
    EVAL_CMD+=("${PASSTHROUGH_ARGS[@]}")
fi

agentrl::log "=========================================="
agentrl::log "Evaluating: model=%s  config=%s" "${MODEL_BASENAME}" "${CONFIG_BASENAME}"
agentrl::log "Output: %s" "${OUTPUT_DIR}"
agentrl::log "Running: %s" "${EVAL_CMD[*]}"
agentrl::log "=========================================="

"${EVAL_CMD[@]}"
exit_code=$?

agentrl::log "=========================================="
if [[ ${exit_code} -eq 0 ]]; then
    agentrl::log "Evaluation completed successfully: model=%s config=%s" "${MODEL_BASENAME}" "${CONFIG_BASENAME}"
else
    agentrl::log "Evaluation FAILED (exit %d): model=%s config=%s" "${exit_code}" "${MODEL_BASENAME}" "${CONFIG_BASENAME}"
fi
agentrl::log "=========================================="

echo "------ Disk Usage for ~ ------"
df -h ~
echo
echo "------ Disk Usage for /tmp ------"
df -h /tmp
echo

exit ${exit_code}