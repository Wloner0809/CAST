#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${REPO_ROOT}/launch_scripts/lib/agentrl_launch.sh"

# --- Model ---
MODEL_PATH="${MODEL_PATH:-${MODEL_DIR:-./models}/Qwen3-4B-Instruct-2507}"
MODEL_BASENAME="Qwen3-4B-Instruct-2507"

# --- Dataset selection (controls difficulty, max_steps, data files) ---
# Set DATASET=id or DATASET=unseen to switch
DATASET="${DATASET:-id}"
case "${DATASET}" in
  id)
    DATASET_DIFFICULTY="${DATASET_DIFFICULTY:-id-oracle_adv}"
    MAX_STEPS="${MAX_STEPS:-30}"
    TRAIN_FILE="${TRAIN_FILE:-data/datasets/sokoban/id/train_verl.parquet}"
    VAL_FILE="${VAL_FILE:-data/datasets/sokoban/id/test_verl.parquet}"
    ;;
  unseen)
    DATASET_DIFFICULTY="${DATASET_DIFFICULTY:-unseen-oracle_adv}"
    MAX_STEPS="${MAX_STEPS:-40}"
    TRAIN_FILE="${TRAIN_FILE:-data/datasets/sokoban/unseen/train_verl.parquet}"
    VAL_FILE="${VAL_FILE:-data/datasets/sokoban/unseen/test_verl.parquet}"
    ;;
  *)
    echo "ERROR: Unknown DATASET='${DATASET}'. Use 'id' or 'unseen'." >&2
    exit 1
    ;;
esac

# --- CAST: Additive Asinh Oracle Shaping with Batch RMS Normalization ---
# "cast" is the production mapping; "cast_ablate" additionally honors the
# CAST_TRANSFORM / CAST_NORM knobs below (used for the asinh/RMS ablation).
ORACLE_ADV_MODE="${ORACLE_ADV_MODE:-cast}"  # off / cast / cast_ablate
CAST_ALPHA="${CAST_ALPHA:-0.3}"
CAST_EPS="${CAST_EPS:-1e-8}"
# Ablation knobs, only consulted when ORACLE_ADV_MODE=cast_ablate (production
# "cast" ignores them). Defaults reproduce "cast" exactly: asinh + RMS norm.
CAST_TRANSFORM="${CAST_TRANSFORM:-asinh}"  # asinh / identity / signed_log
CAST_NORM="${CAST_NORM:-rms}"              # rms / none

# --- Batch / Sequence lengths ---
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"

# --- Deadlock game-over behavior + name suffix ---
# GAME_OVER_ON_DEADLOCK: 0 = detection only (episode never ends on deadlock);
# >=1 = end the episode after that many consecutive deadlock detections. The
# run/checkpoint name suffix is derived from this so runs are never mislabeled
# (value 0 keeps the historical "no_game_over" naming for backward compat).
GAME_OVER_ON_DEADLOCK="${GAME_OVER_ON_DEADLOCK:-0}"
if [[ "${GAME_OVER_ON_DEADLOCK}" == "0" ]]; then
  GAME_OVER_SUFFIX="no_game_over"
else
  GAME_OVER_SUFFIX="game_over_${GAME_OVER_ON_DEADLOCK}"
fi

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-${MODEL_BASENAME}-sokoban-${DATASET_DIFFICULTY}}"
export MODEL_NAME="${MODEL_BASENAME}_BS${TRAIN_BATCH_SIZE}_MPL${MAX_PROMPT_LENGTH}_MRL${MAX_RESPONSE_LENGTH}_MS${MAX_STEPS}-${ORACLE_ADV_MODE}-${GAME_OVER_SUFFIX}"

agentrl::setup_log_dirs
agentrl::maybe_enable_xtrace

export TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d%H%M%S)}"
export WANDB_PROJECT="${WANDB_PROJECT:-rLLM-Sokoban-RL-ablation}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-${REPO_ROOT:-.}/outputs/sokoban-ablation}"

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


CKPT_DIR_BASENAME="${CKPT_DIR_BASENAME:-${MODEL_BASENAME}-sokoban-${DATASET_DIFFICULTY}}"

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
# Hard cap on the number of optimizer steps. When set (>0), verl's trainer stops
# at global_steps >= TOTAL_TRAINING_STEPS regardless of dataset size / epochs.
# Leave empty to fall back to the epoch-derived default (len(dataloader)*epochs).
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"

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
ENV_NAME="${ENV_NAME:-sokoban}"
AGENT_NAME="${AGENT_NAME:-sokoban_agent}"
MASK_TRUNCATED="${MASK_TRUNCATED:-False}"
REJECTION_SAMPLE="${REJECTION_SAMPLE:-False}"
REJECTION_MULTIPLIER="${REJECTION_MULTIPLIER:-1}"
STEPWISE_ADV="${STEPWISE_ADV:-True}"
DISABLE_THINKING="${DISABLE_THINKING:-False}"
USE_ACCUMULATE_HISTORY="${USE_ACCUMULATE_HISTORY:-True}"
USE_MULTISTEP_PROMPT="${USE_MULTISTEP_PROMPT:-False}"
ASYNC_ENGINE="${ASYNC_ENGINE:-True}"
REWARD_MODE="${REWARD_MODE:-binary}"

# --- Deadlock Detection ---
DEADLOCK_DETECTION="${DEADLOCK_DETECTION:-solver}"
DEADLOCK_SOLVER_MAX_NODES="${DEADLOCK_SOLVER_MAX_NODES:-200000}"
# GAME_OVER_ON_DEADLOCK + GAME_OVER_SUFFIX are set earlier (near MODEL_NAME).

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

CHECKPOINT_DIR="${CHECKPOINT_DIR:-${CKPT_DIR:-./model-ckpts}/${WANDB_PROJECT}/${CKPT_DIR_BASENAME}_BS${TRAIN_BATCH_SIZE}_MPL${MAX_PROMPT_LENGTH}_MRL${MAX_RESPONSE_LENGTH}_MS${MAX_STEPS}-${ORACLE_ADV_MODE}-${GAME_OVER_SUFFIX}}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${CKPT_DIR_BASENAME}_BS${TRAIN_BATCH_SIZE}_MPL${MAX_PROMPT_LENGTH}_MRL${MAX_RESPONSE_LENGTH}_MS${MAX_STEPS}-${ORACLE_ADV_MODE}-${GAME_OVER_SUFFIX}}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${WANDB_RUN_NAME}}"
export WANDB_RESUME="allow"

TRAINING_CMD=(
python3 -m examples.sokoban.train_sokoban_agent
"algorithm.adv_estimator=${ADV_ESTIMATOR}"
"+algorithm.oracle_adv_mapping_mode=${ORACLE_ADV_MODE}"
"+algorithm.cast_alpha=${CAST_ALPHA}"
"+algorithm.cast_eps=${CAST_EPS}"
"+algorithm.cast_transform=${CAST_TRANSFORM}"
"+algorithm.cast_norm=${CAST_NORM}"
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
"+rllm.env.env_args.reward_mode=${REWARD_MODE}"
"rllm.stepwise_advantage.enable=${STEPWISE_ADV}"
"rllm.stepwise_advantage.mode=stitched"
"rllm.disable_thinking=${DISABLE_THINKING}"
"+rllm.agent.agent_args.max_steps=${MAX_STEPS}"
"+rllm.agent.agent_args.use_accumulate_history=${USE_ACCUMULATE_HISTORY}"
"+rllm.agent.agent_args.use_multistep_prompt=${USE_MULTISTEP_PROMPT}"
"+agent.async_engine=${ASYNC_ENGINE}"
"trainer.total_epochs=${TOTAL_EPOCHS}"
"+rllm.env.env_args.deadlock_detection=${DEADLOCK_DETECTION}"
"+rllm.env.env_args.deadlock_solver_max_nodes=${DEADLOCK_SOLVER_MAX_NODES}"
"+rllm.env.env_args.game_over_on_deadlock=${GAME_OVER_ON_DEADLOCK}"
)

# Optional hard cap on optimizer steps (e.g. TOTAL_TRAINING_STEPS=200).
if [[ -n "${TOTAL_TRAINING_STEPS}" ]]; then
  TRAINING_CMD+=("trainer.total_training_steps=${TOTAL_TRAINING_STEPS}")
fi

# Forward additional hydra overrides
TRAINING_CMD+=("$@")

agentrl::run_single_or_ray_multinode "${N_NODE}" "${rank}" "${main}" "${DONE_FILE}" "${LOG_FILE}" -- "${TRAINING_CMD[@]}"
