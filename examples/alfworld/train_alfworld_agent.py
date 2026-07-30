#!/usr/bin/env python3
"""Training script for ALFWorld agent using PPO."""

import hydra
from omegaconf import DictConfig

from rllm.agents.alfworld_agent import ALFWorldAgent
from rllm.data import DatasetRegistry
from rllm.environments.alfworld.alfworld_env import ALFWorldEnv
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(
    config_path="pkg://rllm.trainer.config",
    config_name="agent_ppo_trainer",
    version_base=None,
)
def main(config: DictConfig):
    # Load datasets
    train_dataset = DatasetRegistry.load_dataset("alfworld", "train")
    val_dataset = DatasetRegistry.load_dataset("alfworld", "test")

    if train_dataset is None:
        raise ValueError(
            "Dataset alfworld_train not found. "
            "Run: python -m examples.alfworld.prepare_alfworld_data --config-path <path_to_config>"
        )

    # Get config path from environment args
    config_path = config.rllm.env.env_args.get(
        "config_path",
        "rllm/environments/alfworld/alfworld_pkg/configs/config_tw.yaml"
    )

    # Build agent args
    agent_args = {
        "max_steps": config.rllm.agent.get("max_steps", 50),
        "use_admissible_commands": config.rllm.agent.agent_args.get("use_admissible_commands", True),
        "use_accumulate_history": config.rllm.agent.agent_args.get("use_accumulate_history", True),
        "use_accumulate_thinking": config.rllm.agent.agent_args.get("use_accumulate_thinking", True),
    }

    # Build environment args
    env_args = {
        "config_path": config_path,
        "max_steps": config.rllm.agent.get("max_steps", 50),
        "train_eval": config.rllm.env.env_args.get("train_eval", "train"),
        "task_types": config.rllm.env.env_args.get("task_types", None),
    }

    trainer = AgentTrainer(
        agent_class=ALFWorldAgent,
        env_class=ALFWorldEnv,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        agent_args=agent_args,
        env_args=env_args,
    )
    try:
        trainer.train()
    finally:
        ALFWorldEnv.shutdown_pool()


if __name__ == "__main__":
    main()
