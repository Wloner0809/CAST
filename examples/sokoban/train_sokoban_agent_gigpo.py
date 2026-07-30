"""Sokoban training entrypoint using GiGPO advantage estimation.

This is the GiGPO variant of ``train_sokoban_agent.py``.  The only
difference is the trainer class: ``AgentGiGPOTrainer`` replaces the
standard ``AgentPPOTrainer``, and the ``AgentTrainer`` wrapper is
replaced by a thin helper that wires in the correct trainer class.
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.device import is_cuda_available

from rllm.agents.sokoban_agent import SokobanAgent
from rllm.data import DatasetRegistry
from rllm.data.dataset import Dataset
from rllm.environments.sokoban.sokoban import SokobanEnv
from rllm.gigpo.agent_gigpo_trainer import AgentGiGPOTrainer
from rllm.trainer.verl.ray_runtime_env import get_ppo_ray_runtime_env


def _load(config, split: str):
    """Load dataset, using data.train_file / data.val_file when set."""
    file_key = "train_file" if split == "train" else "val_file"
    data_file = getattr(config.data, file_key, None)
    if data_file:
        return Dataset.load_data(data_file)
    return DatasetRegistry.load_dataset("sokoban", split)


def _print_run_info(config, train_dataset, val_dataset):
    sep = "=" * 60
    train_file = getattr(config.data, "train_file", None) or "DatasetRegistry(sokoban/train)"
    val_file = getattr(config.data, "val_file", None) or "DatasetRegistry(sokoban/test)"
    print(sep)
    print("  SOKOBAN TRAINING RUN INFO (GiGPO)")
    print(sep)
    print(f"  mode       : gigpo")
    print(f"  env        : {SokobanEnv.__name__}")
    print(f"  agent      : {SokobanAgent.__name__}")
    print(f"  model      : {config.actor_rollout_ref.model.path}")
    print(f"  train_file : {train_file}  ({len(train_dataset)} examples)")
    print(f"  val_file   : {val_file}  ({len(val_dataset)} examples)")
    gigpo_cfg = config.rllm.get("gigpo", {})
    print(f"  gigpo.enable          : {gigpo_cfg.get('enable', False)}")
    print(f"  gigpo.step_advantage_w: {gigpo_cfg.get('step_advantage_w', 1.0)}")
    print(f"  gigpo.mode            : {gigpo_cfg.get('mode', 'mean_norm')}")
    print(f"  gigpo.gamma           : {gigpo_cfg.get('gamma', 1.0)}")
    print(sep)
    print()


@ray.remote(num_cpus=1)
class GiGPOTaskRunner:
    """Ray remote task runner that uses AgentGiGPOTrainer."""

    def run(self, config, agent_class, env_class, agent_args=None, env_args=None):
        from pprint import pprint

        from verl.single_controller.ray import RayWorkerGroup
        from verl.utils.fs import copy_to_local

        print(f"GiGPOTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        OmegaConf.register_new_resolver("mul", lambda x, y: int(x) * int(y), replace=True)
        OmegaConf.resolve(config)
        pprint(OmegaConf.to_container(config))

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        from verl.utils import hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        ray_worker_group_cls = RayWorkerGroup

        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            assert config.critic.strategy in {"fsdp", "fsdp2"}
            from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
            if use_legacy_worker_impl in ["auto", "enable"]:
                from verl.workers.fsdp_workers import CriticWorker
            elif use_legacy_worker_impl == "disable":
                from verl.workers.roles import CriticWorker
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

            actor_rollout_cls = AsyncActorRolloutRefWorker
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.workers.megatron_workers import AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker
        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            from verl.workers.fsdp_workers import ActorRolloutRefWorker

            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0,
            **config.reward_model.get("reward_kwargs", {}),
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1,
            **config.reward_model.get("reward_kwargs", {}),
        )
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping,
        )

        env_args = env_args or {}
        agent_args = agent_args or {}
        if config.rllm.env.get("env_args") is not None:
            env_args.update(config.rllm.env.get("env_args"))
        if config.rllm.agent.get("agent_args") is not None:
            agent_args.update(config.rllm.agent.get("agent_args"))

        # Use AgentGiGPOTrainer instead of AgentPPOTrainer
        trainer = AgentGiGPOTrainer(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            env_class=env_class,
            agent_class=agent_class,
            env_args=env_args,
            agent_args=agent_args,
        )

        trainer.init_workers()
        try:
            trainer.fit_agent()
        finally:
            trainer.shutdown()


@hydra.main(
    config_path="pkg://rllm.trainer.config",
    config_name="agent_ppo_trainer",
    version_base=None,
)
def main(config):
    train_dataset = _load(config, "train")
    val_dataset = _load(config, "test")

    _print_run_info(config, train_dataset, val_dataset)

    # Set dataset paths for verl dataloader
    config.data.train_files = train_dataset.get_verl_data_path()
    config.data.val_files = val_dataset.get_verl_data_path()

    if not ray.is_initialized():
        if config is not None and hasattr(config, "ray_init"):
            ray_init_settings = {k: v for k, v in config.ray_init.items() if v is not None}
        else:
            ray_init_settings = {}
        ray.init(runtime_env=get_ppo_ray_runtime_env(), **ray_init_settings)

    runner = GiGPOTaskRunner.remote()
    ray.get(
        runner.run.remote(
            config=config,
            agent_class=SokobanAgent,
            env_class=SokobanEnv,
        )
    )


if __name__ == "__main__":
    main()
