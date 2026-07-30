import hydra

from rllm.agents.minesweeper_agent import MinesweeperAgent
from rllm.data import DatasetRegistry
from rllm.data.dataset import Dataset
from rllm.environments.minesweeper.minesweeper_env import MinesweeperEnv
from rllm.trainer.agent_trainer import AgentTrainer


def _load(config, split: str):
    """Load dataset, using data.train_file / data.val_file when set."""
    file_key = "train_file" if split == "train" else "val_file"
    data_file = getattr(config.data, file_key, None)
    if data_file:
        return Dataset.load_data(data_file)
    return DatasetRegistry.load_dataset("minesweeper", split)


def _print_run_info(config, train_dataset, val_dataset):
    sep = "=" * 60
    train_file = getattr(config.data, "train_file", None) or "DatasetRegistry(minesweeper/train)"
    val_file = getattr(config.data, "val_file", None) or "DatasetRegistry(minesweeper/test)"
    print(sep)
    print("  MINESWEEPER TRAINING RUN INFO")
    print(sep)
    print(f"  mode       : baseline")
    print(f"  env        : {MinesweeperEnv.__name__}")
    print(f"  agent      : {MinesweeperAgent.__name__}")
    print(f"  model      : {config.actor_rollout_ref.model.path}")
    print(f"  train_file : {train_file}  ({len(train_dataset)} examples)")
    print(f"  val_file   : {val_file}  ({len(val_dataset)} examples)")
    print(sep)
    print()


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    train_dataset = _load(config, "train")
    val_dataset = _load(config, "test")

    _print_run_info(config, train_dataset, val_dataset)

    trainer = AgentTrainer(
        agent_class=MinesweeperAgent,
        env_class=MinesweeperEnv,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
