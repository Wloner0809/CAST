import hydra

from rllm.agents.rush_hour_agent import RushHourAgent
from rllm.data import DatasetRegistry
from rllm.data.dataset import Dataset
from rllm.environments.rush_hour.rush_hour_env import RushHourEnv
from rllm.trainer.agent_trainer import AgentTrainer


def _load(config, split: str):
    """Load dataset, using data.train_file / data.val_file when set."""
    file_key = "train_file" if split == "train" else "val_file"
    data_file = getattr(config.data, file_key, None)
    if data_file:
        return Dataset.load_data(data_file)
    return DatasetRegistry.load_dataset("rush_hour", split)


def _print_run_info(config, train_dataset, val_dataset):
    sep = "=" * 60
    train_file = getattr(config.data, "train_file", None) or "DatasetRegistry(rush_hour/train)"
    val_file = getattr(config.data, "val_file", None) or "DatasetRegistry(rush_hour/test)"
    print(sep)
    print("  RUSH HOUR TRAINING RUN INFO")
    print(sep)
    print(f"  mode       : baseline")
    print(f"  env        : {RushHourEnv.__name__}")
    print(f"  agent      : {RushHourAgent.__name__}")
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
        agent_class=RushHourAgent,
        env_class=RushHourEnv,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
