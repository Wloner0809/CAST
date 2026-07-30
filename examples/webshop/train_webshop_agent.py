#!/usr/bin/env python3
"""Training script for WebShop agent using PPO."""

import hydra
from omegaconf import DictConfig

from rllm.agents.webshop_agent import WebShopAgent
from rllm.data import DatasetRegistry
from rllm.data.dataset import Dataset
from rllm.environments.webshop.webshop_env import WebShopEnv
from rllm.trainer.agent_trainer import AgentTrainer


def _load(config, split: str):
    """Load dataset, using data.train_file / data.val_file when set."""
    file_key = "train_file" if split == "train" else "val_file"
    data_file = getattr(config.data, file_key, None)
    if data_file:
        return Dataset.load_data(data_file)
    return DatasetRegistry.load_dataset("webshop", split)


def _print_run_info(config, train_dataset, val_dataset):
    sep = "=" * 60
    train_file = getattr(config.data, "train_file", None) or "DatasetRegistry(webshop/train)"
    val_file = getattr(config.data, "val_file", None) or "DatasetRegistry(webshop/test)"
    print(sep)
    print("  WEBSHOP TRAINING RUN INFO")
    print(sep)
    print(f"  mode       : baseline")
    print(f"  env        : {WebShopEnv.__name__}")
    print(f"  agent      : {WebShopAgent.__name__}")
    print(f"  model      : {config.actor_rollout_ref.model.path}")
    print(f"  train_file : {train_file}  ({len(train_dataset)} examples)")
    print(f"  val_file   : {val_file}  ({len(val_dataset)} examples)")
    print(sep)
    print()


@hydra.main(
    config_path="pkg://rllm.trainer.config",
    config_name="agent_ppo_trainer",
    version_base=None,
)
def main(config: DictConfig):
    train_dataset = _load(config, "train")
    val_dataset = _load(config, "test")

    if train_dataset is None:
        raise ValueError(
            "Dataset webshop/train not found. "
            "Run: python -m examples.webshop.prepare_webshop_data"
        )

    _print_run_info(config, train_dataset, val_dataset)

    trainer = AgentTrainer(
        agent_class=WebShopAgent,
        env_class=WebShopEnv,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    try:
        trainer.train()
    finally:
        WebShopEnv.shutdown_pool()


if __name__ == "__main__":
    main()
