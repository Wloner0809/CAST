#!/usr/bin/env bash
# =============================================================================
# eval_closed.sh — closed-source model evaluation, run in parallel
# =============================================================================
# Evaluates closed-source models (Claude / GPT / Gemini) through an
# OpenAI-compatible gateway across the CAST game environments and the OOD
# benchmarks. Concurrency is handled by xargs -P: one (env, model) pair per job,
# each job independent, a failure never blocks the queue.
#
# For open-source models (local weights via sglang) use eval_batch.sh instead.
#
# API keys and the logical-model -> variant mapping live in lib/key_pool.sh.
# Fill in your keys there first.
#
# Resume: a combo that already finished with rc=0 under the same config is
# skipped, so re-running the script only fills the gaps.
#
# Usage:
#   bash scripts/eval_scripts/eval_closed.sh
#
#   PARALLEL=3 bash .../eval_closed.sh                # override concurrency
#   MODELS="opus-4.5" ENVS="sokoban" bash .../...      # run a subset
#   LIMIT=2 bash .../eval_closed.sh                    # smoke test
#   DRY_RUN=1 bash .../eval_closed.sh                  # print the task list only
#   RLLM_EVAL_FORCE_RERUN=1 bash .../...               # ignore resume, redo all
#
#   # long run in the background
#   nohup bash scripts/eval_scripts/eval_closed.sh \
#     > scripts/eval_scripts/logs/_orchestrator.log 2>&1 &
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RLLM_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SELF_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
mkdir -p "${LOG_DIR}"
cd "${RLLM_DIR}"

# key pool + failover (run_eval_with_failover / MODEL_VARIANTS live here)
source scripts/eval_scripts/lib/key_pool.sh

# --- models / envs to run (logical model names, see MODEL_VARIANTS in key_pool.sh) ---
MODELS="${MODELS:-sonnet-4.5 sonnet-4.6 gemini-2.5-pro gemini-2.5-flash opus-4.6 opus-4.5}"
ENVS="${ENVS:-sokoban minesweeper rush_hour}"

# --- concurrency ---
# PARALLEL   max simultaneous (env, model) jobs
# N_PARALLEL agent trajectories in flight inside one job
# N_ROLLOUTS samples per task; 4 gives Avg@4
PARALLEL="${PARALLEL:-2}"
N_PARALLEL="${N_PARALLEL:-5}"
N_ROLLOUTS="${N_ROLLOUTS:-4}"

# --- per-env config / output root ---
declare -A ENV_CFG=(
    [sokoban]="rllm/eval/config/closed_config/sokoban.yaml"
    [minesweeper]="rllm/eval/config/closed_config/minesweeper.yaml"
    [rush_hour]="rllm/eval/config/closed_config/rush_hour.yaml"
    [alfworld]="rllm/eval/config/closed_config/alfworld.yaml"
    [webshop]="rllm/eval/config/closed_config/webshop.yaml"
)
declare -A ENV_OUT=(
    [sokoban]="${RLLM_DIR}/outputs/sokoban"
    [minesweeper]="${RLLM_DIR}/outputs/minesweeper"
    [rush_hour]="${RLLM_DIR}/outputs/rush_hour"
    [alfworld]="${RLLM_DIR}/outputs/alfworld"
    [webshop]="${RLLM_DIR}/outputs/webshop"
)

# --- difficulty / data split / step budget per env ---
# DIFFICULTY selects the middle output path segment; DATA_FILE selects the data.
# Switch between id/ and unseen/ here to evaluate the other split.
declare -A ENV_DIFFICULTY=(
    [sokoban]="unseen"
    [minesweeper]="unseen"
    [rush_hour]="unseen"
    [alfworld]="eval_id"
    [webshop]="default"
)
declare -A ENV_DATA_FILE=(
    [sokoban]="unseen/test"
    [minesweeper]="unseen/test"
    [rush_hour]="unseen/test"
    [alfworld]="eval_id/test"
    [webshop]="test"
)
declare -A ENV_MAX_STEPS=(
    [sokoban]="40"
    [minesweeper]="40"
    [rush_hour]="20"
    [alfworld]="30"
    [webshop]="30"
)

# extra per-env --set pairs appended after the shared ones (empty for the games).
# alfworld/webshop select data via data.split; their loaders ignore data.data_file.
declare -A ENV_EXTRA=(
    [alfworld]="data.split=eval_id agent_args.use_admissible_commands=true agent_args.use_accumulate_history=true"
    [webshop]="data.split=test agent_args.use_available_actions=true agent_args.use_accumulate_history=true"
)

# shared sampling / length params
SAMPLING_TEMPERATURE="1.0"
SAMPLING_TOP_P="1.0"
MAX_RESPONSE_LENGTH="16384"
MAX_PROMPT_LENGTH="2048"

# concurrency-safe eval script; failover reads EVAL_UNIFIED_BIN
EVAL_UNIFIED_BIN="${RLLM_DIR}/scripts/eval_scripts/lib/eval_unified_concurrent.sh"
export EVAL_UNIFIED_BIN
# closed-source evals hit an API and need no Ray; skipping ray.init() saves memory
export RLLM_EVAL_SKIP_RAY=1

DRY_RUN="${DRY_RUN:-0}"

# --- per-env override string: the single source of the config that reaches the eval ---
# api_keys are not set here; key_pool.sh injects them during failover.
_overrides_for() {
    local env="$1"
    local max_steps="${ENV_MAX_STEPS[$env]}"
    local extra="${ENV_EXTRA[$env]:-}"
    echo "sampling.temperature=${SAMPLING_TEMPERATURE} sampling.top_p=${SAMPLING_TOP_P} execution.max_steps=${max_steps} execution.max_response_length=${MAX_RESPONSE_LENGTH} execution.max_prompt_length=${MAX_PROMPT_LENGTH} data.data_file=${ENV_DATA_FILE[$env]} env_args.max_steps=${max_steps} agent_args.max_steps=${max_steps} execution.resume=true${extra:+ ${extra}}"
}

# --- run_tag: readable prefix + hash of EVERYTHING that affects the result ---
# The hash covers the full override string plus rollouts/parallelism/difficulty/LIMIT,
# so any config change yields a new tag and resume can never reuse stale results.
_run_tag() {
    local env="$1"
    local base="ms${ENV_MAX_STEPS[$env]}_mrl${MAX_RESPONSE_LENGTH}_${ENV_DATA_FILE[$env]//\//-}"
    [[ "${N_ROLLOUTS}" != "1" ]] && base="${base}_n${N_ROLLOUTS}"
    local fingerprint
    fingerprint="$(printf '%s|%s|%s|%s|%s' \
        "$(_overrides_for "${env}")" "${N_ROLLOUTS}" "${N_PARALLEL}" \
        "${ENV_DIFFICULTY[$env]}" "${LIMIT:-all}" | md5sum | cut -c1-8)"
    echo "${base}_${fingerprint}"
}

# --- has this (env, model) already succeeded under the current config? ---
_already_done_ok() {
    local env="$1" model="$2"
    local log_file="${LOG_DIR}/${env}_${model}.log"
    local tag
    tag="$(_run_tag "${env}")"
    [[ "${RLLM_EVAL_FORCE_RERUN:-0}" == "1" ]] && return 1
    [[ -f "${log_file}" ]] || return 1
    grep -qF "run_tag=${tag} " "${log_file}" || return 1
    grep -qE "DONE \(rc=0\)" "${log_file}"
}

# --- run one (env, model): own log, own ray tmp dir, returns the real rc ---
launch_one() {
    local pair="$1"
    local env="${pair%%|*}"
    local model="${pair##*|}"

    if _already_done_ok "${env}" "${model}"; then
        echo "[parallel] SKIP env=${env} model=${model} (already DONE rc=0, same config)"
        return 0
    fi

    local cfg="${ENV_CFG[$env]}"
    local out="${ENV_OUT[$env]}"
    local difficulty="${ENV_DIFFICULTY[$env]}"
    local run_tag
    run_tag="$(_run_tag "${env}")"
    local log_file="${LOG_DIR}/${env}_${model}.log"
    local overrides
    overrides="$(_overrides_for "${env}")"

    local ray_tmp="/tmp/ray_peval_${env}_${model}_${BASHPID}"
    mkdir -p "${ray_tmp}"

    # LIMIT must be exported, not used as a command prefix: across line
    # continuations a ${LIMIT:+LIMIT=..} prefix parses as a command name.
    [[ -n "${LIMIT:-}" ]] && export LIMIT

    echo "[parallel] START env=${env} model=${model} -> ${log_file}"

    local rc
    # subshell so the redirection applies to the whole block and the inner exit
    # does not kill this --run-one process before the FINISH line below
    (
        echo "========== ${env} x ${model} =========="
        echo "config=${cfg} difficulty=${difficulty} run_tag=${run_tag} resume=true N_ROLLOUTS=${N_ROLLOUTS}"
        DIFFICULTY="${difficulty}" \
        N_ROLLOUTS="${N_ROLLOUTS}" N_PARALLEL="${N_PARALLEL}" \
        RAY_EVAL_TEMP_DIR="${ray_tmp}" \
        EXTRA_OVERRIDES="${overrides}" \
        run_eval_with_failover "${model}" "${cfg}" "${out}" "${run_tag}"
        _inner_rc=$?
        echo "========== ${env} x ${model} DONE (rc=${_inner_rc}) =========="
        exit ${_inner_rc}
    ) >"${log_file}" 2>&1
    rc=$?

    rm -rf "${ray_tmp}" 2>/dev/null || true

    if [[ "${rc}" -eq 0 ]]; then
        echo "[parallel] FINISH env=${env} model=${model} OK"
    else
        echo "[parallel] FINISH env=${env} model=${model} FAILED (rc=${rc}, see ${log_file})"
    fi
    return "${rc}"
}

# child mode: invoked by xargs, runs one combo and exits. Placed before the
# pre-install block so children never re-run pip.
if [[ "${1:-}" == "--run-one" ]]; then
    launch_one "$2"
    exit $?
fi

# main process installs deps once; children skip it to avoid a pip race
if [[ "${RLLM_SKIP_PREINSTALL:-0}" != "1" ]]; then
    echo "[parallel] Pre-installing dependencies once (serial)..."
    source "${RLLM_DIR}/launch_scripts/lib/agentrl_launch.sh"
    agentrl::conda_activate
    agentrl::pip_install_editable_if_requested
    echo "[parallel] Pre-install done; children will skip reinstall."
fi
export RLLM_SKIP_PIP_INSTALL=1

echo "=========================================="
echo "  Closed-Source Parallel Evaluation"
echo "=========================================="
echo "  Envs:       ${ENVS}"
echo "  Models:     ${MODELS}"
echo "  Parallel:   ${PARALLEL} (concurrent env,model jobs)"
echo "  N_PARALLEL: ${N_PARALLEL} (agent trajectories per job)"
echo "  N_ROLLOUTS: ${N_ROLLOUTS}"
echo "  Logs:       ${LOG_DIR}/{env}_{model}.log"
echo "  Resume:     enabled (skip DONE rc=0 with the same run_tag)"
[[ -n "${LIMIT:-}" ]] && echo "  LIMIT:      ${LIMIT}"
echo "=========================================="

for env in ${ENVS}; do
    if [[ -z "${ENV_CFG[$env]+_}" ]]; then
        echo "[parallel] ERROR: unknown env '${env}' (known: ${!ENV_CFG[*]})" >&2
        exit 1
    fi
done

for model in ${MODELS}; do
    if [[ -z "${MODEL_VARIANTS[$model]+_}" ]]; then
        echo "[parallel] ERROR: unknown model '${model}' (known: ${!MODEL_VARIANTS[*]})" >&2
        exit 1
    fi
done

TASK_LIST=()
for env in ${ENVS}; do
    for model in ${MODELS}; do
        TASK_LIST+=("${env}|${model}")
    done
done
echo "[parallel] Total tasks: ${#TASK_LIST[@]}"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[parallel] DRY_RUN: task list ->"
    printf '  %s\n' "${TASK_LIST[@]}"
    exit 0
fi

# xargs -P gates concurrency: one env|model per line, re-entering this script.
printf '%s\n' "${TASK_LIST[@]}" | \
    RLLM_SKIP_PREINSTALL=1 RLLM_SKIP_PIP_INSTALL=1 \
    xargs -P "${PARALLEL}" -I {} bash "${SELF_PATH}" --run-one {}

echo "=========================================="
echo "  All jobs finished. Results:"
echo "=========================================="
FAILED=0
for pair in "${TASK_LIST[@]}"; do
    env="${pair%%|*}"; model="${pair##*|}"
    if grep -qE "DONE \(rc=0\)" "${LOG_DIR}/${env}_${model}.log" 2>/dev/null; then
        echo "  OK      ${env} x ${model}"
    else
        echo "  FAILED  ${env} x ${model}  (see ${LOG_DIR}/${env}_${model}.log)"
        FAILED=$((FAILED + 1))
    fi
done
echo "=========================================="
echo "  Total: ${#TASK_LIST[@]}  Failed: ${FAILED}"
echo "=========================================="

[[ "${FAILED}" -gt 0 ]] && exit 1
exit 0
