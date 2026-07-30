#!/usr/bin/env bash
set -uxo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_ROOT}/launch_scripts/lib/agentrl_launch.sh"

# --- Model ---
MODEL_PATH="${MODEL_PATH:-${MODEL_DIR:-./models}/Qwen3-4B-Instruct-2507}"
MODEL_BASENAME="Qwen3-4B-Instruct-2507"

# --- Dataset selection (controls difficulty, max_steps, data files) ---
# Set DATASET=id / unseen to switch.
DATASET="${DATASET:-id}"
case "${DATASET}" in
  id)
    DATASET_DIFFICULTY="${DATASET_DIFFICULTY:-id-oracle_adv}"
    MAX_STEPS="${MAX_STEPS:-20}"
    TRAIN_FILE="${TRAIN_FILE:-data/datasets/rush_hour/id/train_verl.parquet}"
    VAL_FILE="${VAL_FILE:-data/datasets/rush_hour/id/test_verl.parquet}"
    ;;
  unseen)
    DATASET_DIFFICULTY="${DATASET_DIFFICULTY:-unseen-oracle_adv}"
    MAX_STEPS="${MAX_STEPS:-20}"
    TRAIN_FILE="${TRAIN_FILE:-data/datasets/rush_hour/unseen/train_verl.parquet}"
    VAL_FILE="${VAL_FILE:-data/datasets/rush_hour/unseen/test_verl.parquet}"
    ;;
  *)
    echo "ERROR: Unknown DATASET='${DATASET}'. Use 'id' or 'unseen'." >&2
    exit 1
    ;;
esac

# --- Solver Oracle Advantage (Credit Assignment) ---
# Rush Hour env computes oracle advantage ONLY when compute_oracle_advantage=True
# (unlike Sokoban whose container always emits it). The solver search depth is
# bounded by solver_max_depth (see RushHourEnvConfig).
COMPUTE_ORACLE_ADVANTAGE="${COMPUTE_ORACLE_ADVANTAGE:-True}"

# --- Oracle N(s) backend (the solver type; part of the run name) ---
#   table       — build exact N(s) table online per board, cache + O(1) lookup (default).
#   precomputed — load a sidecar built with
#                 examples/rush_hour/build_oracle_sidecar.py; on a miss or
#                 missing file it safely falls back to online table build.
#   ida         — per-step IDA* (legacy; can report false INF on hard boards).
ORACLE_BACKEND="${ORACLE_BACKEND:-table}"
# Potential function N(s) encoded by the oracle advantage: 'moves' counts solver
# slides; 'cells' counts slid cells. Part of the run name so the two never share
# a checkpoint/tensorboard/wandb dir.
ORACLE_POTENTIAL="${ORACLE_POTENTIAL:-moves}"
# Default sidecar path follows the dataset difficulty dir AND the potential
# (cells -> *_oracle_cells.pkl.gz). Only read when ORACLE_BACKEND=precomputed; a
# nonexistent file degrades to online build.
# NOTE: `|| true` is required because the sourced launch lib enables `set -e`;
# under ORACLE_POTENTIAL=moves the `[[ == cells ]]` test returns non-zero, which
# would otherwise abort the whole script at this command substitution.
ORACLE_SIDECAR_PATH="${ORACLE_SIDECAR_PATH:-data/datasets/rush_hour/${DATASET}/train_oracle$([[ "${ORACLE_POTENTIAL}" == cells ]] && echo _cells || true).pkl.gz}"

# --- RL-network teacher backend (ORACLE_BACKEND=rlnet) ---
# A *soft* process-supervision teacher: a trained traditional-RL value head V(s)
# replaces the solver's exact N(s) as the oracle potential (env uses -scale*V so
# the unchanged adv=prev-curr yields scale*(V'-V)). CPU-side, firewall-safe
# (never calls the solver), and MUTUALLY EXCLUSIVE with a solver sidecar — so a
# rlnet run MUST clear ORACLE_SIDECAR_PATH (export ORACLE_SIDECAR_PATH='') or the
# env raises ValueError at construction. These knobs are ignored unless
# ORACLE_BACKEND=rlnet.
RLNET_CKPT_PATH="${RLNET_CKPT_PATH:-}"            # trained SB3 ckpt; REQUIRED for rlnet
RLNET_POTENTIAL_SCALE="${RLNET_POTENTIAL_SCALE:-10.0}"  # scales V∈(0,1) diff to ~±1
# Fixed potential for a SOLVED board; must exceed the teacher's non-solved maxQ so
# the winning step's adv=scale*(solved_value-parent_maxQ) stays positive.
RLNET_SOLVED_VALUE="${RLNET_SOLVED_VALUE:-1.5}"
RLNET_DEVICE="${RLNET_DEVICE:-cpu}"              # CPU recommended (keeps the teacher off the training GPU)
RLNET_MAX_PIECES="${RLNET_MAX_PIECES:-8}"        # MUST match the teacher's training

# --- CAST: Additive Asinh Oracle Shaping with Batch RMS Normalization ---
# asinh + batch-RMS self-normalizes, so no per-game re-tuning of the oracle
# advantage range is needed. "off" falls back to plain GRPO.
ORACLE_ADV_MODE="${ORACLE_ADV_MODE:-cast}"  # off / cast
CAST_ALPHA="${CAST_ALPHA:-0.1}"
CAST_EPS="${CAST_EPS:-1e-8}"

# --- Batch / Sequence lengths ---
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-${MODEL_BASENAME}-rush_hour-${DATASET_DIFFICULTY}-pot_${ORACLE_POTENTIAL}}"
export MODEL_NAME="${MODEL_BASENAME}_BS${TRAIN_BATCH_SIZE}_MPL${MAX_PROMPT_LENGTH}_MRL${MAX_RESPONSE_LENGTH}_MS${MAX_STEPS}-${ORACLE_ADV_MODE}-solver_${ORACLE_BACKEND}-pot_${ORACLE_POTENTIAL}"

agentrl::setup_log_dirs
agentrl::maybe_enable_xtrace

export TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d%H%M%S)}"
export WANDB_PROJECT="${WANDB_PROJECT:-rLLM-RushHour-RL-final}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-${REPO_ROOT:-.}/outputs/rush_hour-final}"

##############################
# Part 1) Multi-GPU / Multi-node (reusable)
##############################
N_NODE="${N_NODE:-1}"
agentrl::detect_gpus
n_gpus_per_node="${n_gpus_per_node:-${AUTO_DETECTED_GPU_COUNT}}"
read -r main rank <<<"$(agentrl::detect_main_and_rank)"

##############################
# Part 2) Python + system environment (reusable)
##############################
agentrl::conda_activate
agentrl::pip_install_editable_if_requested
agentrl::setup_python_and_system_env

RLLM_DIR="$(agentrl::resolve_repo_root)"
agentrl::log "RLLM_DIR: %s" "${RLLM_DIR}"

##############################
# Part 3) AgentRL training logic (task-specific)
##############################


CKPT_DIR_BASENAME="${CKPT_DIR_BASENAME:-${MODEL_BASENAME}-rush_hour-${DATASET_DIFFICULTY}}"

DONE_FILE="${LOG_DIR}/connection/${EXPERIMENT_NAME}/${MODEL_BASENAME}/main_done_${main}.txt"
mkdir -p "$(dirname "${DONE_FILE}")"

# --- Data (TRAIN_FILE, VAL_FILE, MAX_STEPS set by DATASET case above) ---
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-100}"

# --- Algorithm ---
ADV_ESTIMATOR="${ADV_ESTIMATOR:-grpo}"
KL_COEF="${KL_COEF:-}"
ALGO_MODE="${ALGO_MODE:-default}"  # default / grpo / gspo

# --- Actor / Optimizer ---
ACTOR_LR="${ACTOR_LR:-1e-6}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-True}"
PPO_MAX_TOKEN_LEN=40000
USE_KL_LOSS="${USE_KL_LOSS:-}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-}"
CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-}"
CLIP_RATIO_C="${CLIP_RATIO_C:-}"
KL_LOSS_COEF="${KL_LOSS_COEF:-}"
KL_LOSS_TYPE="${KL_LOSS_TYPE:-low_var_kl}"
LOSS_MODE="${LOSS_MODE:-}"
LOSS_AGG_MODE="${LOSS_AGG_MODE:-}"
FILTER_GROUPS_ENABLE="${FILTER_GROUPS_ENABLE:-}"
FILTER_GROUPS_METRIC="${FILTER_GROUPS_METRIC:-}"
ULYSSES_SP_SIZE="${ULYSSES_SP_SIZE:-1}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"
ENTROPY_COEFF="${ENTROPY_COEFF:-0}"
HYBRID_ENGINE="${HYBRID_ENGINE:-True}"
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-True}"

# --- FSDP ---
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-False}"
ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-False}"
REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-False}"

# --- Rollout ---
TP_SIZE="${TP_SIZE:-1}"
ROLLOUT_NAME="${ROLLOUT_NAME:-sglang}"
ROLLOUT_MODE="${ROLLOUT_MODE:-async}"
TEMPERATURE="${TEMPERATURE:-1.0}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
ROLLOUT_N="${ROLLOUT_N:-8}"
ENFORCE_EAGER="${ENFORCE_EAGER:-False}"
FREE_CACHE_ENGINE="${FREE_CACHE_ENGINE:-True}"
REF_LOG_PROB_MBS="${REF_LOG_PROB_MBS:-1}"
ROLLOUT_LOG_PROB_MBS="${ROLLOUT_LOG_PROB_MBS:-1}"

# --- Validation Rollout ---
VAL_N="${VAL_N:-4}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.6}"
VAL_TOP_P="${VAL_TOP_P:-0.95}"
VAL_TOP_K="${VAL_TOP_K:--1}"

# --- Trainer ---
CRITIC_WARMUP="${CRITIC_WARMUP:-0}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
SAVE_FREQ="${SAVE_FREQ:-20}"
TEST_FREQ="${TEST_FREQ:-20}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
# Hard cap on the number of optimizer steps. verl stops at
# global_steps >= TOTAL_TRAINING_STEPS regardless of dataset size / epochs.
# The default is the step count reported in the paper for this game; set to an
# empty string to fall back to the epoch-derived value (len(dataloader)*epochs).
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS-300}"

# --- Checkpoint format (save HF weights every step, roll the full checkpoint) ---
# SAVE_CONTENTS controls what each save writes. Adding 'hf_model' makes verl dump a
# full HuggingFace model into global_step_N/actor/huggingface; the rllm trainer
# (AgentPPOTrainer._save_checkpoint) then relocates it to a flat
# global_step_N_hf_model/ directory that survives checkpoint rotation.
# MAX_ACTOR_CKPT_TO_KEEP=1 keeps only the latest full checkpoint
# (model+optimizer+extra) for crash-resume, so disk holds exactly one resumable
# global_step at a time plus the accumulated per-step HF models.
SAVE_CONTENTS="${SAVE_CONTENTS:-['model','optimizer','extra','hf_model']}"
MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-1}"

# --- RLLM / Env / Agent ---
ENV_NAME="${ENV_NAME:-rush_hour}"
AGENT_NAME="${AGENT_NAME:-rush_hour_agent}"
MASK_TRUNCATED="${MASK_TRUNCATED:-False}"
REJECTION_SAMPLE="${REJECTION_SAMPLE:-False}"
REJECTION_MULTIPLIER="${REJECTION_MULTIPLIER:-1}"
STEPWISE_ADV="${STEPWISE_ADV:-True}"
DISABLE_THINKING="${DISABLE_THINKING:-False}"
USE_ACCUMULATE_HISTORY="${USE_ACCUMULATE_HISTORY:-True}"
USE_MULTISTEP_PROMPT="${USE_MULTISTEP_PROMPT:-False}"
ASYNC_ENGINE="${ASYNC_ENGINE:-True}"

# --- Algorithm Mode Defaults ---
case "${ALGO_MODE}" in
  default)
    USE_KL_LOSS="${USE_KL_LOSS:-False}"
    KL_COEF="${KL_COEF:-0.001}"
    KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
    CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.28}"
    CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"
    CLIP_RATIO_C="${CLIP_RATIO_C:-3.0}"
    LOSS_MODE="${LOSS_MODE:-vanilla}"
    LOSS_AGG_MODE="${LOSS_AGG_MODE:-token-mean}"
    FILTER_GROUPS_ENABLE="${FILTER_GROUPS_ENABLE:-False}"
    FILTER_GROUPS_METRIC="${FILTER_GROUPS_METRIC:-seq_final_reward}"
    ;;
  grpo)
    USE_KL_LOSS="${USE_KL_LOSS:-True}"
    KL_COEF="${KL_COEF:-0.001}"
    KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
    CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.2}"
    CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"
    CLIP_RATIO_C="${CLIP_RATIO_C:-3.0}"
    LOSS_MODE="${LOSS_MODE:-vanilla}"
    LOSS_AGG_MODE="${LOSS_AGG_MODE:-token-mean}"
    FILTER_GROUPS_ENABLE="${FILTER_GROUPS_ENABLE:-False}"
    FILTER_GROUPS_METRIC="${FILTER_GROUPS_METRIC:-seq_final_reward}"
    ;;
  gspo)
    USE_KL_LOSS="${USE_KL_LOSS:-False}"
    KL_COEF="${KL_COEF:-0.0}"
    KL_LOSS_COEF="${KL_LOSS_COEF:-0.0}"
    CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.0004}"
    CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.0003}"
    CLIP_RATIO_C="${CLIP_RATIO_C:-3.0}"
    LOSS_MODE="${LOSS_MODE:-gspo}"
    LOSS_AGG_MODE="${LOSS_AGG_MODE:-seq-mean-token-mean}"
    FILTER_GROUPS_ENABLE="${FILTER_GROUPS_ENABLE:-False}"
    FILTER_GROUPS_METRIC="${FILTER_GROUPS_METRIC:-seq_final_reward}"
    ;;
  *)
    echo "Unknown ALGO_MODE: ${ALGO_MODE}. Supported: default, grpo, gspo" >&2
    exit 1
    ;;
esac

CHECKPOINT_DIR="${CHECKPOINT_DIR:-${CKPT_DIR:-./model-ckpts}/${WANDB_PROJECT}/${CKPT_DIR_BASENAME}_BS${TRAIN_BATCH_SIZE}_MPL${MAX_PROMPT_LENGTH}_MRL${MAX_RESPONSE_LENGTH}_MS${MAX_STEPS}-${ORACLE_ADV_MODE}-solver_${ORACLE_BACKEND}-pot_${ORACLE_POTENTIAL}}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${CKPT_DIR_BASENAME}_BS${TRAIN_BATCH_SIZE}_MPL${MAX_PROMPT_LENGTH}_MRL${MAX_RESPONSE_LENGTH}_MS${MAX_STEPS}-${ORACLE_ADV_MODE}-solver_${ORACLE_BACKEND}-pot_${ORACLE_POTENTIAL}}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${WANDB_RUN_NAME}}"
export WANDB_RESUME="allow"

TRAINING_CMD=(
python3 -m examples.rush_hour.train_rush_hour_agent
"algorithm.adv_estimator=${ADV_ESTIMATOR}"
"+algorithm.oracle_adv_mapping_mode=${ORACLE_ADV_MODE}"
"+algorithm.cast_alpha=${CAST_ALPHA}"
"+algorithm.cast_eps=${CAST_EPS}"
"data.train_batch_size=${TRAIN_BATCH_SIZE}"
"data.val_batch_size=${VAL_BATCH_SIZE}"
"data.train_file=${TRAIN_FILE}"
"data.val_file=${VAL_FILE}"
"data.max_prompt_length=${MAX_PROMPT_LENGTH}"
"data.max_response_length=${MAX_RESPONSE_LENGTH}"
"actor_rollout_ref.model.path=${MODEL_PATH}"
"actor_rollout_ref.hybrid_engine=${HYBRID_ENGINE}"
"actor_rollout_ref.model.use_remove_padding=${USE_REMOVE_PADDING}"
"actor_rollout_ref.actor.optim.lr=${ACTOR_LR}"
"actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
"actor_rollout_ref.actor.use_dynamic_bsz=${USE_DYNAMIC_BSZ}"
"actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN}"
"actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS}"
"actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}"
"actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW}"
"actor_rollout_ref.actor.clip_ratio_c=${CLIP_RATIO_C}"
"actor_rollout_ref.actor.policy_loss.loss_mode=${LOSS_MODE}"
"actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE}"
"actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}"
"actor_rollout_ref.actor.kl_loss_type=${KL_LOSS_TYPE}"
"actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ULYSSES_SP_SIZE}"
"actor_rollout_ref.model.enable_gradient_checkpointing=${GRADIENT_CHECKPOINTING}"
"actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD}"
"actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD}"
"actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE}"
"actor_rollout_ref.rollout.name=${ROLLOUT_NAME}"
"actor_rollout_ref.rollout.mode=${ROLLOUT_MODE}"
"actor_rollout_ref.rollout.temperature=${TEMPERATURE}"
"actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL}"
"actor_rollout_ref.rollout.n=${ROLLOUT_N}"
"actor_rollout_ref.rollout.val_kwargs.n=${VAL_N}"
"actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE}"
"actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P}"
"actor_rollout_ref.rollout.val_kwargs.top_k=${VAL_TOP_K}"
"actor_rollout_ref.rollout.enforce_eager=${ENFORCE_EAGER}"
"actor_rollout_ref.rollout.free_cache_engine=${FREE_CACHE_ENGINE}"
"actor_rollout_ref.ref.fsdp_config.param_offload=${REF_PARAM_OFFLOAD}"
"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${REF_LOG_PROB_MBS}"
"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ROLLOUT_LOG_PROB_MBS}"
"actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}"
"algorithm.kl_ctrl.kl_coef=${KL_COEF}"
"+algorithm.filter_groups.enable=${FILTER_GROUPS_ENABLE}"
"+algorithm.filter_groups.metric=${FILTER_GROUPS_METRIC}"
"rllm.mask_truncated_samples=${MASK_TRUNCATED}"
"trainer.critic_warmup=${CRITIC_WARMUP}"
"trainer.logger=['console','wandb','tensorboard']"
"trainer.project_name=${WANDB_PROJECT:-Game-RL}"
"trainer.experiment_name=${WANDB_RUN_NAME}"
"trainer.val_before_train=${VAL_BEFORE_TRAIN}"
"trainer.n_gpus_per_node=${n_gpus_per_node}"
"trainer.nnodes=${N_NODE}"
"trainer.save_freq=${SAVE_FREQ}"
"trainer.default_local_dir=${CHECKPOINT_DIR}"
"trainer.test_freq=${TEST_FREQ}"
"actor_rollout_ref.actor.checkpoint.save_contents=${SAVE_CONTENTS}"
"trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP}"
trainer.default_hdfs_dir=null
"rllm.env.name=${ENV_NAME}"
"rllm.agent.name=${AGENT_NAME}"
"rllm.rejection_sample.enable=${REJECTION_SAMPLE}"
"rllm.rejection_sample.multiplier=${REJECTION_MULTIPLIER}"
"rllm.agent.max_steps=${MAX_STEPS}"
"+rllm.env.env_args.max_steps=${MAX_STEPS}"
"rllm.stepwise_advantage.enable=${STEPWISE_ADV}"
"rllm.stepwise_advantage.mode=stitched"
"rllm.disable_thinking=${DISABLE_THINKING}"
"+rllm.agent.agent_args.max_steps=${MAX_STEPS}"
"+rllm.agent.agent_args.use_accumulate_history=${USE_ACCUMULATE_HISTORY}"
"+agent.async_engine=${ASYNC_ENGINE}"
"trainer.total_epochs=${TOTAL_EPOCHS}"
"+rllm.env.env_args.compute_oracle_advantage=${COMPUTE_ORACLE_ADVANTAGE}"
"+rllm.env.env_args.oracle_backend=${ORACLE_BACKEND}"
"+rllm.env.env_args.oracle_sidecar_path=${ORACLE_SIDECAR_PATH}"
"+rllm.env.env_args.oracle_potential=${ORACLE_POTENTIAL}"
"+rllm.env.env_args.rlnet_ckpt_path=${RLNET_CKPT_PATH}"
"+rllm.env.env_args.rlnet_potential_scale=${RLNET_POTENTIAL_SCALE}"
"+rllm.env.env_args.rlnet_solved_value=${RLNET_SOLVED_VALUE}"
"+rllm.env.env_args.rlnet_device=${RLNET_DEVICE}"
"+rllm.env.env_args.rlnet_max_pieces=${RLNET_MAX_PIECES}"
)

# Hard cap on optimizer steps (empty string disables the cap).
if [[ -n "${TOTAL_TRAINING_STEPS}" ]]; then
  TRAINING_CMD+=("trainer.total_training_steps=${TOTAL_TRAINING_STEPS}")
fi

# Forward additional hydra overrides
TRAINING_CMD+=("$@")

agentrl::run_single_or_ray_multinode "${N_NODE}" "${rank}" "${main}" "${DONE_FILE}" "${LOG_FILE}" -- "${TRAINING_CMD[@]}"
