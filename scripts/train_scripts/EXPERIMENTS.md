# Experiment Recipes

Exact environment-variable settings for every run reported in the paper. Each recipe is a plain shell command — run it from the repository root.

---

## 1. CAST main results

The paper setting is `CAST_ALPHA=0.1`.

### Sokoban

```bash
DATASET=id ORACLE_ADV_MODE=cast CAST_ALPHA=0.1 GAME_OVER_ON_DEADLOCK=1 \
  TOTAL_TRAINING_STEPS=200 \
  DATASET_DIFFICULTY='id-cast-default-alpha0.1-run1' \
  bash scripts/train_scripts/train_sokoban_agent_cast.sh
```

### Minesweeper

```bash
DATASET=id ORACLE_ADV_MODE=cast CAST_ALPHA=0.1 \
  TOTAL_TRAINING_STEPS=400 \
  DATASET_DIFFICULTY='id-cast-default-alpha0.1-run1' \
  bash scripts/train_scripts/train_minesweeper_agent_cast.sh
```

### Rush Hour — search/table oracle

```bash
DATASET=id ORACLE_ADV_MODE=cast CAST_ALPHA=0.1 \
  ORACLE_POTENTIAL=moves MAX_STEPS=20 TOTAL_TRAINING_STEPS=300 \
  DATASET_DIFFICULTY='id-cast-default-alpha0.1-ms20-run1' \
  WANDB_RUN_NAME='Qwen3-4B-Instruct-2507-rush_hour-id-cast-default-alpha0.1-ms20-run1-pot_moves' \
  bash scripts/train_scripts/train_rush_hour_agent_cast.sh
```

### Rush Hour — learned DQN value network

Requires an external Stable-Baselines3 DQN teacher checkpoint.

```bash
DATASET=id ORACLE_ADV_MODE=cast CAST_ALPHA=0.1 \
  ORACLE_BACKEND=rlnet ORACLE_SIDECAR_PATH='' \
  RLNET_CKPT_PATH="${RLNET_CKPT_ROOT:-./stable-baselines3/rush_hour_rl}/checkpoints/<YOUR_TEACHER_CHECKPOINT>.zip" \
  RLNET_POTENTIAL_SCALE=2 RLNET_SOLVED_VALUE=1.0 RLNET_DEVICE=cpu RLNET_MAX_PIECES=8 \
  MAX_STEPS=20 TOTAL_TRAINING_STEPS=300 \
  DATASET_DIFFICULTY='id-cast-default-alpha0.1-ms20-rlnet-w005-s2-sv1.0-run1' \
  WANDB_RUN_NAME='Qwen3-4B-Instruct-2507-rush_hour-id-rlnet-w005-s2-sv1.0-alpha0.1-ms20-run1' \
  bash scripts/train_scripts/train_rush_hour_agent_cast.sh
```

---

## 2. Alpha sweep

Sokoban, in-distribution, capped at 200 optimizer steps. Run each value of `CAST_ALPHA` in turn:

```bash
for ALPHA in 0.01 0.1 0.3 0.5; do
  DATASET=id ORACLE_ADV_MODE=cast CAST_ALPHA="${ALPHA}" \
    GAME_OVER_ON_DEADLOCK=1 TOTAL_TRAINING_STEPS=200 \
    DATASET_DIFFICULTY="id-cast-default-alpha${ALPHA}-abl-run1" \
    bash scripts/train_scripts/train_sokoban_agent_ablation.sh
done
```

---

## 3. asinh / RMS-norm ablation

These use `ORACLE_ADV_MODE=cast_ablate`, which exposes the compression transform and the normalization as knobs. With `CAST_TRANSFORM=asinh CAST_NORM=rms` it is numerically equivalent to production `cast`; the three runs below each drop one component.

```bash
# (a) drop asinh: identity transform, RMS norm kept
DATASET=id ORACLE_ADV_MODE=cast_ablate CAST_ALPHA=0.1 \
  CAST_TRANSFORM=identity \
  GAME_OVER_ON_DEADLOCK=1 TOTAL_TRAINING_STEPS=200 \
  DATASET_DIFFICULTY='id-identity-alpha0.1-abl-run1' \
  bash scripts/train_scripts/train_sokoban_agent_ablation.sh

# (b) drop RMS norm: asinh kept
DATASET=id ORACLE_ADV_MODE=cast_ablate CAST_ALPHA=0.1 \
  CAST_TRANSFORM=asinh CAST_NORM=none \
  GAME_OVER_ON_DEADLOCK=1 TOTAL_TRAINING_STEPS=200 \
  DATASET_DIFFICULTY='id-asinh-nonorm-alpha0.1-abl-run1' \
  bash scripts/train_scripts/train_sokoban_agent_ablation.sh

# (c) drop both
DATASET=id ORACLE_ADV_MODE=cast_ablate CAST_ALPHA=0.1 \
  CAST_TRANSFORM=identity CAST_NORM=none \
  GAME_OVER_ON_DEADLOCK=1 TOTAL_TRAINING_STEPS=200 \
  DATASET_DIFFICULTY='id-identity-nonorm-alpha0.1-abl-run1' \
  bash scripts/train_scripts/train_sokoban_agent_ablation.sh
```

---

## 4. Solver-time profiling

Measures the wall-clock overhead of the solver teacher: an outcome-only baseline against CAST with the oracle-advantage cache enabled, both capped at 200 steps.

```bash
# baseline (no solver signal)
DATASET=id GAME_OVER_ON_DEADLOCK=1 TOTAL_TRAINING_STEPS=200 \
  WANDB_PROJECT='rLLM-Sokoban-RL-ablation' \
  WANDB_DIR="${OUTPUT_DIR:-./outputs}/sokoban-ablation" \
  CHECKPOINT_DIR="${CKPT_DIR:-./model-ckpts}/rLLM-Sokoban-RL-ablation/Qwen3-4B-Instruct-2507-sokoban-id-baseline-MS30-game_over_1-solver_time_abl" \
  WANDB_RUN_NAME='Qwen3-4B-Instruct-2507-sokoban-id-baseline-MS30-game_over_1-solver_time_abl' \
  DATASET_DIFFICULTY='id-baseline-solvertime-run1' \
  bash scripts/train_scripts/train_sokoban_agent.sh

# CAST with oracle-advantage caching
DATASET=id ORACLE_ADV_MODE=cast CAST_ALPHA=0.1 \
  GAME_OVER_ON_DEADLOCK=1 TOTAL_TRAINING_STEPS=200 \
  WANDB_RUN_NAME='Qwen3-4B-Instruct-2507-sokoban-id-cast-alpha0.1-MS30-game_over_1-solver_time_abl' \
  DATASET_DIFFICULTY='id-cast-alpha0.1-solver_time_abl-cache-run1' \
  bash scripts/train_scripts/train_sokoban_agent_ablation.sh \
    +rllm.env.env_args.oracle_advantage_cache=True
```

---

## 5. Baselines

### GRPO / GSPO / DAPO

`ALGO_MODE=default` is the DAPO configuration.

```bash
# Sokoban
for MODE in grpo gspo default; do
  ALGO_MODE="${MODE}" DATASET=id GAME_OVER_ON_DEADLOCK=1 \
    TOTAL_TRAINING_STEPS=200 \
    DATASET_DIFFICULTY="id-${MODE}-tp1_bs16_n8-run1" \
    bash scripts/train_scripts/train_sokoban_agent.sh
done

# Minesweeper
for MODE in grpo gspo default; do
  ALGO_MODE="${MODE}" DATASET=id \
    TOTAL_TRAINING_STEPS=400 \
    DATASET_DIFFICULTY="id-${MODE}-tp1_bs16_n8-run1" \
    bash scripts/train_scripts/train_minesweeper_agent.sh
done

# Rush Hour
for MODE in default grpo gspo; do
  ALGO_MODE="${MODE}" DATASET=id MAX_STEPS=20 \
    TOTAL_TRAINING_STEPS=300 \
    DATASET_DIFFICULTY="id-${MODE}-ms20-tp1_bs16_n8-run1" \
    WANDB_RUN_NAME="Qwen3-4B-Instruct-2507-rush_hour-agent-id-${MODE}-ms20-run1-off-pot_moves" \
    bash scripts/train_scripts/train_rush_hour_agent.sh
done
```

### GiGPO

All three games share the same hyperparameters: `gamma=0.95`, `mean_std_norm`, no episode-cross-steps, `KL_LOSS_COEF=0.001`.

```bash
# Sokoban
DATASET=id MAX_STEPS=200 GAME_OVER_ON_DEADLOCK=1 \
  GIGPO_GAMMA=0.95 GIGPO_MODE=mean_std_norm GIGPO_EPISODE_CROSS_STEPS=False \
  KL_LOSS_COEF=0.001 TOTAL_TRAINING_STEPS=200 \
  DATASET_DIFFICULTY='id-gigpo-g0.95-mean_std_norm-crossF-kl0.001-tp1_bs16_n8-run1' \
  WANDB_RUN_NAME='Qwen3-4B-sokoban-id-gigpo-g0.95-mean_std_norm-crossF-kl0.001-run1' \
  bash scripts/train_scripts/train_sokoban_agent_gigpo.sh

# Minesweeper
DATASET=id \
  GIGPO_GAMMA=0.95 GIGPO_MODE=mean_std_norm GIGPO_EPISODE_CROSS_STEPS=False \
  KL_LOSS_COEF=0.001 TOTAL_TRAINING_STEPS=400 \
  DATASET_DIFFICULTY='id-gigpo-g0.95-mean_std_norm-crossF-kl0.001-tp1_bs16_n8-run1' \
  WANDB_RUN_NAME='Qwen3-4B-minesweeper-id-gigpo-g0.95-mean_std_norm-crossF-kl0.001-run1' \
  bash scripts/train_scripts/train_minesweeper_agent_gigpo.sh

# Rush Hour
DATASET=id MAX_STEPS=20 \
  GIGPO_GAMMA=0.95 GIGPO_MODE=mean_std_norm GIGPO_EPISODE_CROSS_STEPS=False \
  KL_LOSS_COEF=0.001 TOTAL_TRAINING_STEPS=300 \
  DATASET_DIFFICULTY='id-gigpo-g0.95-mean_std_norm-crossF-kl0.001-tp1_bs16_n8-run1' \
  WANDB_RUN_NAME='Qwen3-4B-rush_hour-id-gigpo-g0.95-mean_std_norm-crossF-kl0.001-run1' \
  bash scripts/train_scripts/train_rush_hour_agent_gigpo.sh
```
