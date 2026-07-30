#!/usr/bin/env bash
# =============================================================================
# eval_batch.sh — local batch evaluation runner
# =============================================================================
# Runs many evaluations locally with a bounded concurrency pool (xargs -P).
# No cluster scheduler involved: every job is a plain local process.
#
# JOB_MODE selects how the job list is built:
#   - per_env: one job per (model x env) combo, each job runs eval_unified.sh.
#   - ckpt:    one job per checkpoint dir, each job sweeps (step x env) serially.
#
# Concurrency is controlled by PARALLEL (default 1 = fully serial).
#
# Usage:
#   bash scripts/eval_scripts/eval_batch.sh                        # per_env (default)
#   JOB_MODE=ckpt STEPS="100 200 300" bash scripts/eval_scripts/eval_batch.sh
#   JOB_MODE=ckpt STEPS="100-500:100" bash scripts/eval_scripts/eval_batch.sh
#   PARALLEL=4 bash scripts/eval_scripts/eval_batch.sh             # 4 jobs at a time
#   DRY_RUN=1 bash scripts/eval_scripts/eval_batch.sh              # print the plan only
#
# For closed-source API models use eval_closed.sh instead.
#
# Logs: one file per job under scripts/eval_scripts/logs/<job_name>.log
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SELF_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
RLLM_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"

# Per-eval script the jobs call. With PARALLEL>1 the parent switches this to the
# concurrent variant, because plain eval_unified.sh runs `ray stop --force` and
# would kill sibling jobs' Ray clusters. Exported so worker branches inherit it.
EVAL_BIN="${EVAL_BIN:-${SCRIPT_DIR}/eval_unified.sh}"

##############################
# --run-one worker mode: runs a single job, invoked by xargs.
# Takes the job name and the full command line as two separate argv entries, so
# nothing has to be re-parsed out of a delimited string. Not for manual use.
##############################
if [[ "${1:-}" == "--run-one" ]]; then
    shift
    job_name="$1"; job_cmd="$2"
    mkdir -p "${LOG_DIR}"
    log_file="${LOG_DIR}/${job_name}.log"

    echo "[batch] START ${job_name} -> ${log_file}"
    (
        echo "========== ${job_name} =========="
        echo "cmd: ${job_cmd}"
        echo "=========================================="
        eval "${job_cmd}"
        _rc=$?
        echo "========== ${job_name} DONE (rc=${_rc}) =========="
        exit ${_rc}
    ) >"${log_file}" 2>&1
    rc=$?

    if [[ ${rc} -eq 0 ]]; then
        echo "[batch] FINISH ${job_name} OK"
    else
        echo "[batch] FINISH ${job_name} FAILED (rc=${rc}, see ${log_file})"
    fi
    exit ${rc}
fi

##############################
# --loop worker: evaluate several checkpoint steps of one env, serially.
# usage: eval_batch.sh --loop <ckpt_base_dir> <config_yaml> <output_dir> <step1,step2,...>
##############################
if [[ "${1:-}" == "--loop" ]]; then
    shift
    CKPT_BASE_DIR="$1"; CONFIG_YAML="$2"; OUTPUT_DIR="$3"; STEPS_CSV="$4"

    IFS=',' read -ra STEP_LIST <<< "${STEPS_CSV}"
    total_steps=${#STEP_LIST[@]}

    echo "=========================================="
    echo "Sequential Checkpoint Evaluation"
    echo "=========================================="
    echo "Checkpoint dir: ${CKPT_BASE_DIR}"
    echo "Config:         ${CONFIG_YAML}"
    echo "Output dir:     ${OUTPUT_DIR}"
    echo "Steps (${total_steps}):   ${STEPS_CSV}"
    echo "=========================================="

    declare -a SUCCESS_STEPS=()
    declare -a FAILED_STEPS=()

    for i in "${!STEP_LIST[@]}"; do
        step="${STEP_LIST[$i]}"
        idx=$((i + 1))
        model_path="${CKPT_BASE_DIR}/global_step_${step}_hf_model"

        echo ""
        echo "  [${idx}/${total_steps}] Evaluating step ${step} -> ${model_path}"

        if [[ ! -d "${model_path}" ]]; then
            echo "  ERROR: Model path does not exist, skipping: ${model_path}" >&2
            FAILED_STEPS+=("${step}")
            continue
        fi

        if bash "${EVAL_BIN}" "${model_path}" "${CONFIG_YAML}" "${OUTPUT_DIR}"; then
            echo "  Step ${step} completed successfully"
            SUCCESS_STEPS+=("${step}")
        else
            echo "  Step ${step} FAILED (exit code $?)" >&2
            FAILED_STEPS+=("${step}")
        fi
    done

    echo ""
    echo "=========================================="
    echo "Summary: total=${total_steps} success=${#SUCCESS_STEPS[@]} failed=${#FAILED_STEPS[@]}"
    [[ ${#FAILED_STEPS[@]} -gt 0 ]] && echo "Failed steps: ${FAILED_STEPS[*]}"
    echo "=========================================="

    [[ ${#FAILED_STEPS[@]} -gt 0 ]] && exit 1
    exit 0
fi

##############################
# --loop-multi worker: evaluate (step x env) serially for one checkpoint dir.
# usage: eval_batch.sh --loop-multi <ckpt_base_dir> <output_dir> <env1,env2,...> <step1,step2,...>
# Per-env yaml/overrides/rollouts arrive as LOOP_<ENV_UPPER>_YAML/_OVR/_NR env vars.
# A failed (step,env) does not abort the rest; exit code is non-zero if any failed.
##############################
if [[ "${1:-}" == "--loop-multi" ]]; then
    shift
    CKPT_BASE_DIR="$1"; OUTPUT_DIR="$2"; ENV_KEYS_CSV="$3"; STEPS_CSV="$4"

    IFS=',' read -ra STEP_LIST <<< "${STEPS_CSV}"
    IFS=',' read -ra ENV_LIST <<< "${ENV_KEYS_CSV}"
    total_steps=${#STEP_LIST[@]}
    total_envs=${#ENV_LIST[@]}
    total_tasks=$((total_steps * total_envs))

    echo "=========================================="
    echo "Sequential Multi-Env Checkpoint Evaluation"
    echo "=========================================="
    echo "Checkpoint dir: ${CKPT_BASE_DIR}"
    echo "Output dir:     ${OUTPUT_DIR}"
    echo "Steps (${total_steps}):  ${STEPS_CSV}"
    echo "Envs (${total_envs}):   ${ENV_KEYS_CSV}"
    echo "Total tasks:    ${total_tasks} (steps x envs)"
    echo "=========================================="

    declare -a SUCCESS_TASKS=()
    declare -a FAILED_TASKS=()
    task_idx=0

    for step in "${STEP_LIST[@]}"; do
        model_path="${CKPT_BASE_DIR}/global_step_${step}_hf_model"

        echo ""
        echo "  Step ${step}  ->  ${model_path}"

        if [[ ! -d "${model_path}" ]]; then
            echo "  ERROR: Model path does not exist, skipping all envs for this step: ${model_path}" >&2
            for env_key in "${ENV_LIST[@]}"; do
                FAILED_TASKS+=("step${step}/${env_key}")
                task_idx=$((task_idx + 1))
            done
            continue
        fi

        for env_key in "${ENV_LIST[@]}"; do
            task_idx=$((task_idx + 1))
            env_upper="$(echo "${env_key}" | tr '[:lower:]' '[:upper:]')"
            # indirect expansion of the injected per-env config (set -u safe)
            yaml_var="LOOP_${env_upper}_YAML"; env_yaml="${!yaml_var:-}"
            ovr_var="LOOP_${env_upper}_OVR";   env_ovr="${!ovr_var:-}"
            nr_var="LOOP_${env_upper}_NR";     env_nr="${!nr_var:-16}"

            echo ""
            echo "  --- [${task_idx}/${total_tasks}] step ${step} x env ${env_key} ---"
            if [[ -z "${env_yaml}" ]]; then
                echo "    ERROR: missing ${yaml_var} (env config not injected), skipping" >&2
                FAILED_TASKS+=("step${step}/${env_key}")
                continue
            fi

            if EXTRA_OVERRIDES="${env_ovr}" N_ROLLOUTS="${env_nr}" \
                bash "${EVAL_BIN}" "${model_path}" "${env_yaml}" "${OUTPUT_DIR}"; then
                echo "    step ${step} x ${env_key} completed successfully"
                SUCCESS_TASKS+=("step${step}/${env_key}")
            else
                echo "    step ${step} x ${env_key} FAILED (exit code $?)" >&2
                FAILED_TASKS+=("step${step}/${env_key}")
            fi
        done
    done

    echo ""
    echo "=========================================="
    echo "Summary: total=${total_tasks} success=${#SUCCESS_TASKS[@]} failed=${#FAILED_TASKS[@]}"
    [[ ${#FAILED_TASKS[@]} -gt 0 ]] && echo "Failed tasks: ${FAILED_TASKS[*]}"
    echo "=========================================="

    [[ ${#FAILED_TASKS[@]} -gt 0 ]] && exit 1
    exit 0
fi

##############################
# master switch
##############################

# job granularity: per_env | ckpt
JOB_MODE="${JOB_MODE:-per_env}"

# max concurrent jobs. 1 = serial. Raise it only if the box has the GPUs/RAM for it.
PARALLEL="${PARALLEL:-1}"

DRY_RUN="${DRY_RUN:-0}"

# global sample cap (empty = all), handy for smoke tests
LIMIT="${LIMIT:-}"

# Which per-eval script the jobs call. When PARALLEL>1 we must use the concurrent
# variant: plain eval_unified.sh runs `ray stop --force`, which would kill sibling
# jobs' Ray clusters. Each job also gets its own RAY_EVAL_TEMP_DIR (set per job below).
if [[ "${PARALLEL}" -gt 1 ]]; then
    EVAL_BIN="${SCRIPT_DIR}/lib/eval_unified_concurrent.sh"
fi
export EVAL_BIN

##############################
# shared config (all modes)
##############################

# models to evaluate
MODEL_PATHS=(
    "<PATH_TO_MODEL>/Qwen3-4B-Instruct-2507"
)

##############################
# per-env config: single source of truth for all modes
##############################
# One entry per "logical env": yaml path / EXTRA_OVERRIDES (--set pairs) / n_rollouts.
# Every mode names logical envs only and looks the rest up in these three tables.
#
# Notes:
# - ALFWorld is evaluated on its seen split (data.split=eval_id). The loader only
#   honors data.split; data.data_file just supplies the output-path label.
# - Rollouts come from ENV_NROLLOUTS. Keep _COMMON_OVERRIDES free of
#   execution.n_rollouts_per_prompt: --set is applied AFTER --n-rollouts in cli.py,
#   so a value there would silently override the table.

# shared --set overrides (tp/dp, concurrency, lengths, rollouts)
_COMMON_OVERRIDES="sglang.tp_size=1 sglang.dp_size=8 execution.n_parallel_agents=64 execution.max_workers=64 execution.max_response_length=16384 execution.max_prompt_length=2048"

# logical env -> yaml config
declare -A ENV_YAML=(
    [alfworld]="rllm/eval/config/alfworld.yaml"
    [webshop]="rllm/eval/config/webshop.yaml"
    [sokoban]="rllm/eval/config/sokoban.yaml"
    [minesweeper]="rllm/eval/config/minesweeper.yaml"
    [rush_hour]="rllm/eval/config/rush_hour.yaml"
    # ID/OOD variants share the yaml; difficulty comes from data.data_file in ENV_OVERRIDES.
    [sokoban_id]="rllm/eval/config/sokoban.yaml"
    [sokoban_ood]="rllm/eval/config/sokoban.yaml"
    [minesweeper_id]="rllm/eval/config/minesweeper.yaml"
    [minesweeper_ood]="rllm/eval/config/minesweeper.yaml"
    [rush_hour_id]="rllm/eval/config/rush_hour.yaml"
    [rush_hour_ood]="rllm/eval/config/rush_hour.yaml"
)

# logical env -> EXTRA_OVERRIDES (expanded into multiple --set key=value)
declare -A ENV_OVERRIDES=(
    [alfworld]="${_COMMON_OVERRIDES} data.split=eval_id data.data_file=eval_id/test execution.max_steps=30 env_args.max_steps=30 agent_args.max_steps=30 agent_args.use_admissible_commands=true agent_args.use_accumulate_history=true"
    [webshop]="${_COMMON_OVERRIDES} data.split=test execution.max_steps=30 env_args.max_steps=30 agent_args.max_steps=30 agent_args.use_available_actions=true agent_args.use_accumulate_history=true"
    [sokoban]="${_COMMON_OVERRIDES} data.data_file=unseen/test execution.max_steps=40 env_args.max_steps=40 agent_args.max_steps=40 agent_args.use_accumulate_history=true"
    [minesweeper]="${_COMMON_OVERRIDES} data.data_file=unseen/test execution.max_steps=40 env_args.max_steps=40 agent_args.max_steps=40 agent_args.use_accumulate_history=true"
    [rush_hour]="${_COMMON_OVERRIDES} data.data_file=unseen/test execution.max_steps=20 env_args.max_steps=20 agent_args.max_steps=20 agent_args.use_accumulate_history=true"
    # ID/OOD split: data.data_file and max_steps change together.
    [sokoban_id]="${_COMMON_OVERRIDES} data.data_file=id/test execution.max_steps=30 env_args.max_steps=30 agent_args.max_steps=30 agent_args.use_accumulate_history=true"
    [sokoban_ood]="${_COMMON_OVERRIDES} data.data_file=unseen/test execution.max_steps=40 env_args.max_steps=40 agent_args.max_steps=40 agent_args.use_accumulate_history=true"
    [minesweeper_id]="${_COMMON_OVERRIDES} data.data_file=id/test execution.max_steps=30 env_args.max_steps=30 agent_args.max_steps=30 agent_args.use_accumulate_history=true"
    [minesweeper_ood]="${_COMMON_OVERRIDES} data.data_file=unseen/test execution.max_steps=40 env_args.max_steps=40 agent_args.max_steps=40 agent_args.use_accumulate_history=true"
    [rush_hour_id]="${_COMMON_OVERRIDES} data.data_file=id/test execution.max_steps=20 env_args.max_steps=20 agent_args.max_steps=20 agent_args.use_accumulate_history=true"
    [rush_hour_ood]="${_COMMON_OVERRIDES} data.data_file=unseen/test execution.max_steps=20 env_args.max_steps=20 agent_args.max_steps=20 agent_args.use_accumulate_history=true"
)

# logical env -> n_rollouts (exported as N_ROLLOUTS, i.e. --n-rollouts)
declare -A ENV_NROLLOUTS=(
    [alfworld]=4
    [webshop]=4
    [sokoban]=4
    [minesweeper]=4
    [rush_hour]=4
    [sokoban_id]=4
    [sokoban_ood]=4
    [minesweeper_id]=4
    [minesweeper_ood]=4
    [rush_hour_id]=4
    [rush_hour_ood]=4
)

##############################
# per_env mode config
##############################

# logical envs to evaluate (must exist in the tables above).
# The OOD benchmarks alfworld / webshop can be listed here too.
EVAL_ENV_KEYS=(sokoban_id sokoban_ood minesweeper_id minesweeper_ood rush_hour_id rush_hour_ood)

# output root - final path: {PER_ENV_OUTPUT_DIR}/{config}/{difficulty}/{model}
PER_ENV_OUTPUT_DIR="${PER_ENV_OUTPUT_DIR:-./outputs/ood_eval}"

# extra env vars prepended to each job command, e.g. "LOGPROB_METRICS=1 DETAILED_METRICS=1"
EXTRA_EVAL_ENVVARS="${EXTRA_EVAL_ENVVARS:-}"

##############################
# ckpt mode config
##############################

# training checkpoint dirs, each holding global_step_*_hf_model subdirs.
# entry format:
#   "dir_path"            -> use the global STEPS spec below
#   "dir_path::step_spec" -> per-dir spec, e.g. "/path/run2::100-500:100"
CKPT_BASE_DIRS=(
    # "<PATH_TO_CHECKPOINT_DIR>"
)

# default step spec when an entry carries no :: override. Three forms:
#   "all"            evaluate every global_step_*_hf_model found
#   "100 200 300"    explicit steps
#   "100-500:100"    range with stride
STEPS="${STEPS:-all}"

# logical envs for ckpt mode
CKPT_ENV_KEYS=(sokoban_id sokoban_ood minesweeper_id minesweeper_ood rush_hour_id rush_hour_ood)

# ckpt job granularity:
#   per_dir     (default) one job per dir, worker runs --loop-multi (all envs x all steps).
#   per_env_dir           one job per (dir x env), worker runs --loop (one env, all steps).
CKPT_JOB_GRANULARITY="${CKPT_JOB_GRANULARITY:-per_dir}"

# output root - final path:
#   {CKPT_OUTPUT_DIR}/{run_dir_name}/{config}/{difficulty}/global_step_{N}_hf_model/
CKPT_OUTPUT_DIR="${CKPT_OUTPUT_DIR:-./outputs/ood_eval_ckpt}"

##############################
# job list
##############################
# Parallel arrays, one index per job:
#   JOB_NAMES[i]    - short name, also the log filename
#   JOB_CMDS[i]     - shell command line to run
#   JOB_LABELS[i]   - label shown in the summary
#   JOB_PATHS[i]    - model/ckpt path, checked for existence before running
#   JOB_INFO[i]     - multi-line description printed in the plan
declare -a JOB_NAMES=()
declare -a JOB_CMDS=()
declare -a JOB_LABELS=()
declare -a JOB_PATHS=()
declare -a JOB_INFO=()

# replace characters that are awkward in filenames
sanitize_name() {
    local raw="$1"
    echo "${raw//[^a-zA-Z0-9_-]/_}"
}

# short hash of a path, so jobs whose dirs share a basename get distinct names
path_hash() {
    printf '%s' "$1" | md5sum | cut -c1-6
}

# discover checkpoint steps from a dir (ckpt mode). Prints them space-separated.
discover_steps() {
    local ckpt_dir="$1"
    local step_spec="$2"
    local steps=()

    if [[ "${step_spec}" == "all" ]]; then
        while IFS= read -r dir; do
            local basename
            basename="$(basename "${dir}")"
            local step_num
            step_num="$(echo "${basename}" | sed -n 's/^global_step_\([0-9]*\)_hf_model$/\1/p')"
            if [[ -n "${step_num}" ]]; then
                steps+=("${step_num}")
            fi
        done < <(find "${ckpt_dir}" -maxdepth 1 -type d -name 'global_step_*_hf_model' | sort -t_ -k3 -n)
    elif [[ "${step_spec}" == *-*:* ]]; then
        local range_start range_end stride rest
        range_start="$(echo "${step_spec}" | cut -d'-' -f1)"
        rest="$(echo "${step_spec}" | cut -d'-' -f2)"
        range_end="$(echo "${rest}" | cut -d':' -f1)"
        stride="$(echo "${rest}" | cut -d':' -f2)"
        for ((s = range_start; s <= range_end; s += stride)); do
            if [[ -d "${ckpt_dir}/global_step_${s}_hf_model" ]]; then
                steps+=("${s}")
            else
                echo "  Warning: step ${s} not found in ${ckpt_dir}, skipping" >&2
            fi
        done
    else
        for s in ${step_spec}; do
            if [[ -d "${ckpt_dir}/global_step_${s}_hf_model" ]]; then
                steps+=("${s}")
            else
                echo "  Warning: step ${s} not found in ${ckpt_dir}, skipping" >&2
            fi
        done
    fi

    echo "${steps[@]}"
}

##############################
# per_env mode: build the job list
##############################
# One job per (model x logical env), each running eval_unified.sh.
build_jobs_per_env() {
    local model_path env_key config_path overrides n_rollouts
    local model_basename clean_basename job_name cmd info env_prefix

    for model_path in "${MODEL_PATHS[@]}"; do
        for env_key in "${EVAL_ENV_KEYS[@]}"; do
            if [[ -z "${ENV_YAML[${env_key}]+_}" ]]; then
                echo "Error: Unknown env key '${env_key}' (not in ENV_YAML). Available: ${!ENV_YAML[*]}" >&2
                exit 1
            fi
            config_path="${ENV_YAML[${env_key}]}"
            overrides="${ENV_OVERRIDES[${env_key}]}"
            n_rollouts="${ENV_NROLLOUTS[${env_key}]}"

            model_basename="$(basename "${model_path}")"
            clean_basename="$(sanitize_name "${model_basename}")"
            # include a hash of the full path: two runs can share a basename
            # (e.g. runA/global_step_100_hf_model and runB/...) and must not
            # collide onto one log file
            job_name="${clean_basename}-$(path_hash "${model_path}")-${env_key}"

            env_prefix="${EXTRA_EVAL_ENVVARS:+${EXTRA_EVAL_ENVVARS} }EXTRA_OVERRIDES='${overrides}' N_ROLLOUTS=${n_rollouts} RAY_EVAL_TEMP_DIR=/tmp/ray_eb_${job_name} "
            [[ -n "${LIMIT}" ]] && env_prefix+="LIMIT=${LIMIT} "
            # %q-quote the paths: the command string is later run through eval,
            # so an unquoted path containing a space would word-split
            cmd="${env_prefix}bash $(printf '%q %q %q %q' "${EVAL_BIN}" "${model_path}" "${config_path}" "${PER_ENV_OUTPUT_DIR}")"

            info="  Job: model=${model_basename}  env=${env_key}
    Config:     ${config_path}
    N_ROLLOUTS: ${n_rollouts}
    Overrides:  ${overrides}
    Output:     ${PER_ENV_OUTPUT_DIR}/<config>/<difficulty>/${model_basename}"

            JOB_NAMES+=("${job_name}")
            JOB_CMDS+=("${cmd}")
            JOB_LABELS+=("${model_basename} x ${env_key}")
            JOB_PATHS+=("${model_path}")
            JOB_INFO+=("${info}")
        done
    done
}

##############################
# ckpt mode: build the job list
##############################
# Step discovery and output naming are per ckpt dir; only the job split differs:
#   per_dir     one job per dir      -> --loop-multi (all envs x all steps)
#   per_env_dir one job per (dir,env) -> --loop (one env, all steps)
# The output dir gains a {run_dir_name} level so the same step from different runs
# cannot collide.
build_jobs_ckpt() {
    local ckpt_entry ckpt_dir ckpt_steps_spec
    local env_key config_path overrides n_rollouts
    local ckpt_run_name run_dir_name job_name cmd info env_prefix
    local step_list steps_csv run_output_dir
    local env_keys_csv env_upper loop_injects

    CKPT_PLAN_LINES=""

    if [[ "${CKPT_JOB_GRANULARITY}" != "per_env_dir" && "${CKPT_JOB_GRANULARITY}" != "per_dir" ]]; then
        echo "Error: Unknown CKPT_JOB_GRANULARITY='${CKPT_JOB_GRANULARITY}'. Supported: per_env_dir, per_dir" >&2
        exit 1
    fi

    if [[ ${#CKPT_BASE_DIRS[@]} -eq 0 ]]; then
        echo "Error: CKPT_BASE_DIRS is empty. Add at least one checkpoint dir." >&2
        exit 1
    fi

    for ckpt_entry in "${CKPT_BASE_DIRS[@]}"; do
        if [[ "${ckpt_entry}" == *"::"* ]]; then
            ckpt_dir="${ckpt_entry%%::*}"
            ckpt_steps_spec="${ckpt_entry##*::}"
        else
            ckpt_dir="${ckpt_entry}"
            ckpt_steps_spec="${STEPS}"
        fi

        if [[ ! -d "${ckpt_dir}" ]]; then
            echo "Error: Checkpoint directory not found: ${ckpt_dir}" >&2
            exit 1
        fi

        step_list=($(discover_steps "${ckpt_dir}" "${ckpt_steps_spec}"))
        if [[ ${#step_list[@]} -eq 0 ]]; then
            echo "Error: No valid checkpoint steps found in ${ckpt_dir} (spec='${ckpt_steps_spec}')" >&2
            exit 1
        fi
        steps_csv="$(IFS=','; echo "${step_list[*]}")"

        ckpt_run_name="$(basename "${ckpt_dir}")"
        # hash the full path: two ckpt dirs can share a basename and must not
        # share an output dir or a log file
        run_dir_name="$(sanitize_name "${ckpt_run_name}")-$(path_hash "${ckpt_dir}")"
        run_output_dir="${CKPT_OUTPUT_DIR}/${run_dir_name}"

        for env_key in "${CKPT_ENV_KEYS[@]}"; do
            if [[ -z "${ENV_YAML[${env_key}]+_}" ]]; then
                echo "Error: Unknown env key '${env_key}' (not in ENV_YAML). Available: ${!ENV_YAML[*]}" >&2
                exit 1
            fi
        done

        CKPT_PLAN_LINES+="  - ${ckpt_run_name}: steps(${#step_list[@]})=${steps_csv}"$'\n'

        if [[ "${CKPT_JOB_GRANULARITY}" == "per_dir" ]]; then
            env_keys_csv="$(IFS=','; echo "${CKPT_ENV_KEYS[*]}")"
            job_name="${run_dir_name}-ckpt-multi"

            loop_injects=""
            for env_key in "${CKPT_ENV_KEYS[@]}"; do
                env_upper="$(echo "${env_key}" | tr '[:lower:]' '[:upper:]')"
                loop_injects+="LOOP_${env_upper}_YAML='${ENV_YAML[${env_key}]}' "
                loop_injects+="LOOP_${env_upper}_OVR='${ENV_OVERRIDES[${env_key}]}' "
                loop_injects+="LOOP_${env_upper}_NR=${ENV_NROLLOUTS[${env_key}]} "
            done

            env_prefix="${EXTRA_EVAL_ENVVARS:+${EXTRA_EVAL_ENVVARS} }${loop_injects}RAY_EVAL_TEMP_DIR=/tmp/ray_eb_${job_name} "
            [[ -n "${LIMIT}" ]] && env_prefix+="LIMIT=${LIMIT} "
            cmd="${env_prefix}bash $(printf '%q' "${SELF_PATH}") --loop-multi $(printf '%q %q' "${ckpt_dir}" "${run_output_dir}") ${env_keys_csv} ${steps_csv}"

            info="  Job: run=${ckpt_run_name}  (${#step_list[@]} steps x ${#CKPT_ENV_KEYS[@]} envs, serial)
    Steps:      ${steps_csv}
    Envs:       ${env_keys_csv}
    Output:     ${run_output_dir}/<config>/<difficulty>/global_step_<N>_hf_model"

            JOB_NAMES+=("${job_name}")
            JOB_CMDS+=("${cmd}")
            JOB_LABELS+=("${ckpt_run_name} (${#step_list[@]} steps x ${#CKPT_ENV_KEYS[@]} envs)")
            JOB_PATHS+=("${ckpt_dir}")
            JOB_INFO+=("${info}")
        else
            for env_key in "${CKPT_ENV_KEYS[@]}"; do
                config_path="${ENV_YAML[${env_key}]}"
                overrides="${ENV_OVERRIDES[${env_key}]}"
                n_rollouts="${ENV_NROLLOUTS[${env_key}]}"

                job_name="${run_dir_name}-${env_key}-ckpt"

                env_prefix="${EXTRA_EVAL_ENVVARS:+${EXTRA_EVAL_ENVVARS} }EXTRA_OVERRIDES='${overrides}' N_ROLLOUTS=${n_rollouts} RAY_EVAL_TEMP_DIR=/tmp/ray_eb_${job_name} "
                [[ -n "${LIMIT}" ]] && env_prefix+="LIMIT=${LIMIT} "
                cmd="${env_prefix}bash $(printf '%q' "${SELF_PATH}") --loop $(printf '%q %q %q' "${ckpt_dir}" "${config_path}" "${run_output_dir}") ${steps_csv}"

                info="  Job: run=${ckpt_run_name}  env=${env_key}  (${#step_list[@]} steps, serial)
    Steps:      ${steps_csv}
    Config:     ${config_path}
    N_ROLLOUTS: ${n_rollouts}
    Overrides:  ${overrides}
    Output:     ${run_output_dir}/<config>/<difficulty>/global_step_<N>_hf_model"

                JOB_NAMES+=("${job_name}")
                JOB_CMDS+=("${cmd}")
                JOB_LABELS+=("${ckpt_run_name} x ${env_key} (${#step_list[@]} steps)")
                JOB_PATHS+=("${ckpt_dir}")
                JOB_INFO+=("${info}")
            done
        fi
    done
}

##############################
# plan printing
##############################
print_plan() {
    local total="${#JOB_NAMES[@]}"
    local i

    echo "=========================================="
    echo "  Local Batch Evaluation (mode=${JOB_MODE})"
    echo "=========================================="
    echo "  Total jobs:  ${total}"
    echo "  Parallel:    ${PARALLEL}"
    echo "  Logs:        ${LOG_DIR}/<job>.log"
    [[ -n "${LIMIT}" ]] && echo "  LIMIT:       ${LIMIT}"
    echo "  Dry run:     ${DRY_RUN}"

    case "${JOB_MODE}" in
        per_env)
            echo "  Models:      ${#MODEL_PATHS[@]}"
            echo "  Envs:        ${EVAL_ENV_KEYS[*]}"
            echo "  Output root: ${PER_ENV_OUTPUT_DIR}"
            ;;
        ckpt)
            echo "  Ckpt dirs:   ${#CKPT_BASE_DIRS[@]}"
            echo "  Envs:        ${CKPT_ENV_KEYS[*]}"
            echo "  Granularity: ${CKPT_JOB_GRANULARITY}"
            echo "  Output root: ${CKPT_OUTPUT_DIR}"
            echo "  Checkpoint runs:"
            printf '%s' "${CKPT_PLAN_LINES}"
            ;;
    esac
    echo "=========================================="
    echo ""

    for ((i = 0; i < total; i++)); do
        printf '%s\n' "${JOB_INFO[$i]}"
        echo ""
    done
}

##############################
# main flow
##############################
cd "${RLLM_DIR}"
mkdir -p "${LOG_DIR}"

case "${JOB_MODE}" in
    per_env) build_jobs_per_env ;;
    ckpt)    build_jobs_ckpt ;;
    *)
        echo "Error: Unknown JOB_MODE='${JOB_MODE}'. Supported: per_env, ckpt" >&2
        exit 1
        ;;
esac

if [[ ${#JOB_NAMES[@]} -eq 0 ]]; then
    echo "Error: no jobs to run in mode '${JOB_MODE}'." >&2
    exit 1
fi

print_plan

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[batch] DRY_RUN=1, nothing executed. Commands that would run:"
    for ((i = 0; i < ${#JOB_NAMES[@]}; i++)); do
        echo "  [${JOB_NAMES[$i]}] ${JOB_CMDS[$i]}"
    done
    exit 0
fi

# check every model/ckpt path up front so a typo fails before hours of work
MISSING=0
for ((i = 0; i < ${#JOB_NAMES[@]}; i++)); do
    if [[ ! -d "${JOB_PATHS[$i]}" && ! -f "${JOB_PATHS[$i]}" ]]; then
        echo "Error: path does not exist for job '${JOB_NAMES[$i]}': ${JOB_PATHS[$i]}" >&2
        MISSING=1
    fi
done
[[ "${MISSING}" == "1" ]] && exit 1

# Feed NUL-delimited (name, command) pairs into xargs, which keeps at most
# PARALLEL workers alive and passes each pair as two argv entries (-n 2). A
# failing job does not stop the others.
for ((i = 0; i < ${#JOB_NAMES[@]}; i++)); do
    printf '%s\0%s\0' "${JOB_NAMES[$i]}" "${JOB_CMDS[$i]}"
done | LOG_DIR="${LOG_DIR}" xargs -0 -n 2 -P "${PARALLEL}" \
    bash "${SELF_PATH}" --run-one

echo ""
echo "=========================================="
echo "  Batch finished. Per-job results:"
echo "=========================================="
FAILED=0
for ((i = 0; i < ${#JOB_NAMES[@]}; i++)); do
    log_file="${LOG_DIR}/${JOB_NAMES[$i]}.log"
    if grep -qE "DONE \(rc=0\)" "${log_file}" 2>/dev/null; then
        echo "  OK      ${JOB_LABELS[$i]}"
    else
        echo "  FAILED  ${JOB_LABELS[$i]}  (see ${log_file})"
        FAILED=$((FAILED + 1))
    fi
done
echo "=========================================="
echo "  Total: ${#JOB_NAMES[@]}  Failed: ${FAILED}"
echo "  Logs:  ${LOG_DIR}/"
echo "=========================================="

[[ "${FAILED}" -gt 0 ]] && exit 1
exit 0
