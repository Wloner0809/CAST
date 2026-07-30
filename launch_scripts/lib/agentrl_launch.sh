#!/usr/bin/env bash
#
# Reusable launch helpers for AgentRL training scripts.
# - Designed for single-node and Ray-based multi-node launches.
# - Keeps environment setup and distributed logic separate from task overrides.
#
# Usage (from a launch script):
#   source "$(dirname "${BASH_SOURCE[0]}")/lib/agentrl_launch.sh"
#   agentrl::main "$@"
#
set -euo pipefail

agentrl::log() {
  # shellcheck disable=SC2059
  printf "[%s] %s\n" "$(date -Iseconds)" "$(printf "$@")" >&2
}

agentrl::die() {
  agentrl::log "ERROR: %s" "$*"
  exit 1
}

agentrl::maybe_enable_xtrace() {
  if [[ "${DEBUG:-0}" == "1" || "${DEBUG:-0}" == "true" ]]; then
    set -x
  fi
}

agentrl::require_cmd() {
  command -v "$1" >/dev/null 2>&1 || agentrl::die "Missing required command: $1"
}

agentrl::pick_free_port() {
  # Pick an unused local TCP port. Best-effort only; races are still possible.
  agentrl::require_cmd python3
  python3 - <<'PY'
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(("", 0))
port = sock.getsockname()[1]
sock.close()
print(port)
PY
}

agentrl::safe_ulimit() {
  # Best-effort only; don't fail if container forbids it.
  ulimit -n "${RLLM_ULIMIT_NOFILE:-65535}" >/dev/null 2>&1 || true
}

agentrl::sanitize_path() {
  # Some environments inject /root/bin which can cause permission issues.
  if [[ -n "${PATH:-}" ]]; then
    PATH="$(echo "$PATH" | tr ':' '\n' | grep -v '^/root' | tr '\n' ':' | sed 's/:$//')"
    export PATH
  fi
}

agentrl::detect_gpus() {
  local gpu_count gpu_list
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    gpu_count="$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l | tr -d ' ')"
    gpu_list="${CUDA_VISIBLE_DEVICES}"
  else
    if command -v nvidia-smi >/dev/null 2>&1; then
      gpu_count="$(nvidia-smi --list-gpus | wc -l | tr -d ' ')"
      if [[ "${gpu_count}" -gt 0 ]]; then
        gpu_list="$(seq -s, 0 $((gpu_count - 1)))"
      else
        gpu_list=""
      fi
    else
      agentrl::log "Warning: nvidia-smi not found and CUDA_VISIBLE_DEVICES not set. Defaulting to 8 GPUs."
      gpu_count="8"
      gpu_list="0,1,2,3,4,5,6,7"
    fi
  fi

  # Optional Hugging Face mirror, e.g. HF_ENDPOINT=https://hf-mirror.com for
  # networks that cannot reach huggingface.co directly. Only exported when the
  # caller set it: huggingface_hub reads os.getenv("HF_ENDPOINT", default), so an
  # empty value would override the default rather than fall back to it.
  if [[ -n "${HF_ENDPOINT:-}" ]]; then
    export HF_ENDPOINT
  fi
  export AUTO_DETECTED_GPU_COUNT="${gpu_count}"
  export AUTO_DETECTED_GPU_LIST="${gpu_list}"
  agentrl::log "Detected %s GPU(s): %s" "${AUTO_DETECTED_GPU_COUNT}" "${AUTO_DETECTED_GPU_LIST}"
}

agentrl::conda_activate() {
  # Controlled by env:
  # - RLLM_CONDA_ENV: conda env name to activate (default: base)
  # - RLLM_SKIP_CONDA_ACTIVATE=1 to skip activation
  if [[ "${RLLM_SKIP_CONDA_ACTIVATE:-0}" == "1" ]]; then
    return 0
  fi

  if ! command -v conda >/dev/null 2>&1; then
    agentrl::log "conda not found; skipping conda activation."
    return 0
  fi

  # Ensure "conda activate" works inside non-interactive shells.
  # shellcheck disable=SC1091
eval "$(conda shell.bash hook)"
  conda activate "${RLLM_CONDA_ENV:-base}" || agentrl::die "Failed to conda activate ${RLLM_CONDA_ENV:-base}"
}

agentrl::pip_install_editable_if_requested() {
  # Keep compatibility with existing scripts while allowing opt-out.
  # requirements.txt delegates to pyproject.toml so launch jobs, Docker, pip,
  # and uv all consume the same dependency declarations.
  # - RLLM_PIP_INSTALL_EDITABLE=1 enables installation (default: 1)
  # - RLLM_REQUIREMENTS_FILE selects another requirements entry point

  # Best-effort symlink; don't fail if it already exists.
  if [[ ! -e "data" ]]; then
    ln -s "${RLLM_SRC_ROOT:-.}/data" data || true
  fi
  if [[ "${RLLM_PIP_INSTALL_EDITABLE:-1}" == "1" ]]; then
    agentrl::require_cmd python3
    # Allow skipping the (slow) dependency (re)install when the env is already
    # provisioned. Symlinks above are still created; only dependency installation
    # is skipped. Set RLLM_SKIP_PIP_INSTALL=1 (e.g. by an orchestrator that pre-installs
    # once before fanning out many concurrent jobs) to avoid redundant reinstalls
    # and pip races. Default 0 = install as before (backward compatible).
    if [[ "${RLLM_SKIP_PIP_INSTALL:-0}" == "1" ]]; then
      agentrl::log "RLLM_SKIP_PIP_INSTALL=1; skipping dependency (re)install."
      # SPACY_MODEL_DIR is consumed at runtime by WebShop workers, so still export
      # it (cheap, no install) for parity with the install path below.
      export SPACY_MODEL_DIR="${SPACY_MODEL_DIR:-}"
      return 0
    fi
    python3 -m pip install -r "${RLLM_REQUIREMENTS_FILE:-requirements.txt}"
    # WebShop reads this at import time. Models are data assets rather than
    # Python dependencies and must be provisioned explicitly; see README.md.
    export SPACY_MODEL_DIR="${SPACY_MODEL_DIR:-}"
  fi
}

agentrl::resolve_repo_root() {
  # Resolve repo root by importing rllm package. Assumes editable install or installed wheel.
  agentrl::require_cmd python3
  python3 - <<'PY'
import os
import rllm
print(os.path.dirname(os.path.dirname(rllm.__file__)))
PY
}

agentrl::setup_log_dirs() {
  # Set up standardized logging directories for training runs.
  # Expects EXPERIMENT_NAME and MODEL_NAME to be set by the calling script.
  # If not set, uses sensible defaults.
  #
  # Controllable via env:
  # - LOG_DIR: base directory for all logs (default: rllm/outputs)
  # - EXPERIMENT_NAME: experiment identifier (default: "default")
  # - MODEL_NAME: model identifier (default: "model")
  #
  # Exports:
  # - CKPT_DIR: checkpoint directory
  # - TENSORBOARD_DIR: tensorboard logs
  # - ROLLOUT_DIR: rollout trajectory logs
  # - LOG_FILE: main training log file
  # - RAY_LOG_DIR: Ray runtime logs

  local repo_root
  repo_root="${RLLM_REPO_ROOT:-$(pwd)}"

  export LOG_DIR="${LOG_DIR:-${repo_root}/outputs}"
  export EXPERIMENT_NAME="${EXPERIMENT_NAME:-default}"
  export MODEL_NAME="${MODEL_NAME:-model}"
  export TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

  export CKPT_DIR="${LOG_DIR}/checkpoints/${EXPERIMENT_NAME}/${MODEL_NAME}/"
  export TENSORBOARD_DIR="${LOG_DIR}/tensorboard/${EXPERIMENT_NAME}/${MODEL_NAME}/"
  export ROLLOUT_DIR="${LOG_DIR}/rollout_trajectory/${EXPERIMENT_NAME}/${MODEL_NAME}/"
  export LOG_FILE="${LOG_DIR}/train_logs/${EXPERIMENT_NAME}/${MODEL_NAME}/output_${TIMESTAMP}.log"
  export RAY_LOG_DIR="${LOG_DIR}/ray/${EXPERIMENT_NAME}/${MODEL_NAME}/"
  # Ray's temp dir is recreated below; leave the rest of /tmp alone so we never
  # clobber files belonging to other processes or users on a shared machine.
  export RAY_TEMP_DIR="/dev/shm/ray/"
  rm -rf "${RAY_TEMP_DIR:?}" 2>/dev/null || true
  echo "RAY_TEMP_DIR: ${RAY_TEMP_DIR}"
  # Create directories if they don't exist
  mkdir -p "${CKPT_DIR}" "${TENSORBOARD_DIR}" "${ROLLOUT_DIR}" "${RAY_LOG_DIR}" "${RAY_TEMP_DIR}"
  mkdir -p "$(dirname "${LOG_FILE}")"

  agentrl::log "Log directories configured:"
  agentrl::log "  LOG_DIR=%s" "${LOG_DIR}"
  agentrl::log "  CKPT_DIR=%s" "${CKPT_DIR}"
  agentrl::log "  TENSORBOARD_DIR=%s" "${TENSORBOARD_DIR}"
  agentrl::log "  ROLLOUT_DIR=%s" "${ROLLOUT_DIR}"
  agentrl::log "  LOG_FILE=%s" "${LOG_FILE}"
  agentrl::log "  RAY_LOG_DIR=%s" "${RAY_LOG_DIR}"
  agentrl::log "  RAY_TEMP_DIR=%s" "${RAY_TEMP_DIR}"
}

agentrl::setup_python_and_system_env() {
  # --- Part 2: Python + system environment (reusable) ---
  agentrl::safe_ulimit
  agentrl::sanitize_path

  # Avoid tvm_ffi scanning disallowed dirs.
  export TVM_HOME="${TVM_HOME:-}"
  export TVM_LIBRARY_PATH="${TVM_LIBRARY_PATH:-}"

  # Common runtime env defaults (override via env as needed).
  export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
  export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
  export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

  # NCCL defaults (safe to override/disable at call site).
  export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
  export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
  export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-20}"
  export NCCL_IB_QPS_PER_CONNECTION="${NCCL_IB_QPS_PER_CONNECTION:-8}"
  export NCCL_IB_RETRY_CNT="${NCCL_IB_RETRY_CNT:-15}"
  export NCCL_CONNECT_TIMEOUT="${NCCL_CONNECT_TIMEOUT:-1200}"
  export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
  export TORCH_NCCL_AVOID_RECORD_STREAMS="${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}"

  # CUDA allocator defaults.
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"

  # Ray defaults.
  export RAY_worker_register_timeout_seconds="${RAY_worker_register_timeout_seconds:-600}"
  export RAY_DISABLE_DASHBOARD="${RAY_DISABLE_DASHBOARD:-1}"
  export RAY_NUM_CPUS="${RAY_NUM_CPUS:-32}"

  # Java is required by WebShop (pyserini/lucene). Override JAVA_HOME if needed.
  export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-11-openjdk}"
  export JVM_PATH="${JVM_PATH:-${JAVA_HOME}/lib/server/libjvm.so}"
  export PATH=$JAVA_HOME/bin:$PATH

  # Avoid hard-coding secrets in scripts:
  # - WANDB_API_KEY should be provided externally if using wandb
  # - http(s)_proxy should be provided externally if needed

  agentrl::log "https_proxy: %s" "${https_proxy:-}"
  agentrl::log "http_proxy: %s" "${http_proxy:-}"

  if [[ "${RLLM_WARN_MISSING_WANDB_KEY:-0}" == "1" && -z "${WANDB_API_KEY:-}" ]]; then
    agentrl::log "Note: WANDB_API_KEY is not set."
  fi

  # Optional arch list (commonly used on H100 etc.).
  if [[ -n "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    export TORCH_CUDA_ARCH_LIST
  fi
}

# -------------------------
# --- Part 1: Distributed (reusable)
# -------------------------

agentrl::detect_head_addr() {
  # Priority:
  # 1) HEAD_ADDR env
  # 2) CLUSTER_SPEC (json, as exported by many schedulers)
  # 3) CLUSTER_HOSTS ("ip1,ip2,...")
  # 4) hostname -i
  if [[ -n "${HEAD_ADDR:-}" ]]; then
    echo "${HEAD_ADDR}"
    return 0
  fi

  if [[ -n "${CLUSTER_SPEC:-}" ]]; then
    python3 - <<'PY'
import json, os, socket
spec = json.loads(os.environ["CLUSTER_SPEC"])
role = spec["role"]
master = spec[role][0]
master_addr, _ = master.split(":")
print(socket.gethostbyname(master_addr))
PY
    return 0
  fi

  if [[ -n "${CLUSTER_HOSTS:-}" ]]; then
    local first
    first="${CLUSTER_HOSTS%%,*}"
    echo "${first}"
    return 0
  fi

  hostname -i
}

agentrl::detect_main_and_rank() {
  # Prints: "<main_addr> <rank>"
  # Honours explicit env overrides first, then a scheduler-provided CLUSTER_SPEC,
  # then falls back to single-node defaults:
  # - MAIN_ADDR, NODE_RANK
  local main rank
  main="${MAIN_ADDR:-}"
  rank="${NODE_RANK:-}"

  if [[ -n "${main}" && -n "${rank}" ]]; then
    echo "${main} ${rank}"
    return 0
  fi

  if [[ -n "${CLUSTER_SPEC:-}" ]]; then
    rank="$(python3 - <<'PY'
import json, os
spec = json.loads(os.environ["CLUSTER_SPEC"])
print(spec.get("index", 0))
PY
)"
    main="$(agentrl::detect_head_addr)"
    echo "${main} ${rank}"
    return 0
  fi

  echo "localhost 0"
}

agentrl::ray_defaults() {
  export RAY_METRICS_EXPORT_PORT="${RAY_METRICS_EXPORT_PORT:-20541}"
  export RAY_DASHBOARD_AGENT_HTTP_PORT="${RAY_DASHBOARD_AGENT_HTTP_PORT:-52365}"
  export RAY_DASHBOARD_AGENT_GRPC_PORT="${RAY_DASHBOARD_AGENT_GRPC_PORT:-53589}"
  export RAY_RUNTIME_ENV_AGENT_PORT="${RAY_RUNTIME_ENV_AGENT_PORT:-48869}"
  export RAY_MIN_WORKER_PORT="${RAY_MIN_WORKER_PORT:-10002}"
  export RAY_MAX_WORKER_PORT="${RAY_MAX_WORKER_PORT:-12001}"
  export RAY_MASTER_PORT="${RAY_MASTER_PORT:-8278}"
  export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
  export RAY_REDIS_SHARD_PORTS="${RAY_REDIS_SHARD_PORTS:-20007,20008,20009,20010,20011,20012,20013,20014,20015,20016}"
  # Increase timeout for runtime env agent to avoid spurious failures
  export RAY_RUNTIME_ENV_TIMEOUT_SECONDS="${RAY_RUNTIME_ENV_TIMEOUT_SECONDS:-120}"
  # Increase raylet startup wait time to avoid GCS registration timeouts
  export RAY_raylet_start_wait_time_s="${RAY_raylet_start_wait_time_s:-180}"
}

agentrl::ray_cleanup() {
  # Clean up any stale Ray processes from previous runs.
  # This is important to avoid conflicts when starting a new Ray cluster.
  agentrl::log "Cleaning up any stale Ray processes..."
  ray stop --force 2>/dev/null || true
  # Give processes time to fully terminate
  sleep "${RLLM_RAY_CLEANUP_SLEEP_S:-5}"
}

agentrl::ray_start_head() {
  agentrl::require_cmd ray
  agentrl::ray_defaults

  ray start --head \
    --port="${RAY_MASTER_PORT}" \
    --dashboard-host='0.0.0.0' \
    --metrics-export-port="${RAY_METRICS_EXPORT_PORT}" \
    --dashboard-agent-grpc-port="${RAY_DASHBOARD_AGENT_GRPC_PORT}" \
    --dashboard-agent-listen-port="${RAY_DASHBOARD_AGENT_HTTP_PORT}" \
    --runtime-env-agent-port="${RAY_RUNTIME_ENV_AGENT_PORT}" \
    --dashboard-port="${RAY_DASHBOARD_PORT}" \
    --redis-shard-ports="${RAY_REDIS_SHARD_PORTS}" \
    --min-worker-port="${RAY_MIN_WORKER_PORT}" \
    --max-worker-port="${RAY_MAX_WORKER_PORT}" \
    --temp-dir="${RAY_TEMP_DIR}"
}

agentrl::ray_start_worker() {
  agentrl::require_cmd ray
  agentrl::ray_defaults
  local main_addr="$1"

  ray start --address="${main_addr}:${RAY_MASTER_PORT}" \
    --metrics-export-port="${RAY_METRICS_EXPORT_PORT}" \
    --dashboard-agent-grpc-port="${RAY_DASHBOARD_AGENT_GRPC_PORT}" \
    --runtime-env-agent-port="${RAY_RUNTIME_ENV_AGENT_PORT}" \
    --dashboard-agent-listen-port="${RAY_DASHBOARD_AGENT_HTTP_PORT}" \
    --dashboard-port="${RAY_DASHBOARD_PORT}" \
    --min-worker-port="${RAY_MIN_WORKER_PORT}" \
    --max-worker-port="${RAY_MAX_WORKER_PORT}" \
    --temp-dir="${RAY_TEMP_DIR}"
}

agentrl::ray_wait_for_nodes() {
  # Best-effort cluster readiness check using `ray status`.
  local expected_nodes="$1"
  local timeout_s="${2:-600}"
  local start_ts now elapsed active_nodes

  start_ts="$(date +%s)"
  while true; do
    now="$(date +%s)"
    elapsed=$((now - start_ts))

    # The output format varies; keep this intentionally forgiving.
    active_nodes="$(ray status 2>/dev/null | sed -n '/Active:/,/Pending:/p' | grep -E "node_|\\bnode\\b" | wc -l | tr -d ' ')"
    agentrl::log "Ray active nodes: %s / %s" "${active_nodes}" "${expected_nodes}"

    if [[ "${active_nodes}" -ge "${expected_nodes}" ]]; then
      return 0
    fi
    if [[ "${elapsed}" -ge "${timeout_s}" ]]; then
      agentrl::die "Timeout waiting for Ray nodes to join (${elapsed}s >= ${timeout_s}s)."
    fi
    sleep 10
  done
}

agentrl::ray_submit() {
  # Usage: agentrl::ray_submit <log_file> -- <cmd...>
  local log_file="$1"
  shift
  [[ "$1" == "--" ]] || agentrl::die "ray_submit expects: <log_file> -- <cmd...>"
  shift

  mkdir -p "$(dirname "${log_file}")"
  ray job submit --verbose -- "$@" | tee "${log_file}"
}

agentrl::run_single_or_ray_multinode() {
  # Usage: agentrl::run_single_or_ray_multinode <nnodes> <rank> <main> <done_file> <log_file> -- <cmd...>
  local nnodes="$1"
  local rank="$2"
  local main_addr="$3"
  local done_file="$4"
  local log_file="$5"
  shift 5
  [[ "$1" == "--" ]] || agentrl::die "run_single_or_ray_multinode expects: ... -- <cmd...>"
  shift

  if [[ "${nnodes}" -le 1 ]]; then
    "$@"
    return 0
  fi

  agentrl::require_cmd ray
  if [[ "${rank}" -eq 0 ]]; then
    agentrl::log "Launching Ray head (nnodes=%s) at %s" "${nnodes}" "${main_addr}"
    agentrl::ray_start_head
    sleep "${RLLM_RAY_HEAD_START_SLEEP_S:-60}"
    agentrl::log "Ray head started; waiting for nodes to join"
    agentrl::ray_wait_for_nodes "${nnodes}" "${RLLM_RAY_WAIT_TIMEOUT_S:-600}"
    agentrl::log "All nodes joined; submitting job"
    agentrl::ray_submit "${log_file}" -- "$@"
    mkdir -p "$(dirname "${done_file}")"
    touch "${done_file}"
    sleep 5
  else
    agentrl::log "Joining Ray cluster as worker (rank=%s) -> %s" "${rank}" "${main_addr}"
    sleep "${RLLM_RAY_WORKER_START_SLEEP_S:-30}"
    agentrl::ray_start_worker "${main_addr}"
    agentrl::log "Ray worker joined; waiting for done file: %s" "${done_file}"
    # If the done file is not on a shared filesystem, workers can hang forever.
    # As a fallback, also exit if the Ray head becomes unreachable for too long.
    local poll_s head_lost_grace_s head_lost_fail_count_max fail_count start_ts now elapsed
    poll_s="${RLLM_RAY_WORKER_POLL_S:-60}"
    head_lost_grace_s="${RLLM_RAY_WORKER_HEAD_LOST_GRACE_S:-900}"
    head_lost_fail_count_max="${RLLM_RAY_WORKER_HEAD_LOST_FAILS:-5}"
    fail_count=0
    start_ts="$(date +%s)"

    while [[ ! -f "${done_file}" ]]; do
      if ! ray status >/dev/null 2>&1; then
        fail_count=$((fail_count + 1))
      else
        fail_count=0
      fi

      now="$(date +%s)"
      elapsed=$((now - start_ts))
      if [[ "${fail_count}" -ge "${head_lost_fail_count_max}" && "${elapsed}" -ge "${head_lost_grace_s}" ]]; then
        agentrl::log "Head seems unreachable (fail_count=%s, elapsed=%ss). Exiting worker without done file: %s" "${fail_count}" "${elapsed}" "${done_file}"
        return 0
      fi

      sleep "${poll_s}"
    done
  fi
}
