import asyncio
from types import SimpleNamespace

from rllm.eval.portal import EvalPortal, _sanitize_path_for_workers


class _DummyAgent:
    pass


class _DummyEnv:
    @staticmethod
    def is_multithread_safe():
        return True


def test_eval_portal_normalizes_to_single_engine_key():
    portal = EvalPortal(
        {
            "env": "sokoban",
            "engine_name": "verl",
            "model": {"name": "dummy-model"},
        }
    )

    assert portal.config["engine"] == "verl"
    assert "engine_name" not in portal.config


def test_sanitize_path_for_workers_filters_root_and_invalid_entries():
    raw = "/usr/bin:/root/bin:/does/not/exist:/bin:/usr/bin"
    sanitized = _sanitize_path_for_workers(raw)
    parts = sanitized.split(":")
    assert "/usr/bin" in parts
    assert "/bin" in parts
    assert "/root/bin" not in parts
    assert "/does/not/exist" not in parts


def test_eval_portal_verl_uses_rollout_engine_param(monkeypatch):
    import rllm.eval.portal as portal_module
    import rllm.engine.agent_execution_engine as exec_engine_module
    from rllm.eval.registry import EnvConfigRegistry

    captured = {}

    class DummyEngine:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    def fake_get(_env_name):
        return SimpleNamespace(
            env_args={},
            agent_args={},
            agent_class=_DummyAgent,
            env_class=_DummyEnv,
        )

    monkeypatch.setattr(EnvConfigRegistry, "get", staticmethod(fake_get))
    captured_verl_cfg = {}

    def fake_setup_verl_rollout_manager(cfg):
        captured_verl_cfg.update(cfg)
        return "mock_rollout_manager", "mock_verl_config"

    monkeypatch.setattr(
        portal_module,
        "_setup_verl_rollout_manager",
        fake_setup_verl_rollout_manager,
    )
    monkeypatch.setattr(EvalPortal, "_get_tokenizer", lambda self: None)
    monkeypatch.setattr(exec_engine_module, "AgentExecutionEngine", DummyEngine)

    portal = EvalPortal(
        {
            "env": "sokoban",
            "engine": "verl",
            "model": {"name": "dummy-model"},
            "execution": {
                "n_parallel_agents": 2,
                "max_steps": 4,
                "max_response_length": 128,
                "max_prompt_length": 64,
                "max_workers": 2,
                "n_rollouts_per_prompt": 1,
            },
        }
    )

    portal.setup_engine()

    assert captured["engine_name"] == "verl"
    assert captured["rollout_engine"] == "mock_rollout_manager"
    assert captured["config"] == "mock_verl_config"
    assert "rollout_engine_args" not in captured
    assert "do_sample" not in captured["sampling_params"]
    assert "model" not in captured["sampling_params"]
    assert "return_logprobs" not in captured["sampling_params"]
    assert "entropy_mode" not in captured["sampling_params"]
    assert captured_verl_cfg["trainer"]["nnodes"] == 1
    assert captured_verl_cfg["trainer"]["n_gpus_per_node"] == 1
    assert captured_verl_cfg["actor_rollout_ref"]["rollout"]["_target_"] == "verl.workers.config.RolloutConfig"
    assert captured_verl_cfg["actor_rollout_ref"]["rollout"]["prompt_length"] == 64
    assert captured_verl_cfg["actor_rollout_ref"]["rollout"]["response_length"] == 128
    assert "do_sample" in captured_verl_cfg["actor_rollout_ref"]["rollout"]
    assert captured_verl_cfg["actor_rollout_ref"]["rollout"]["val_kwargs"]["_target_"] == "verl.workers.config.SamplingConfig"
    assert captured_verl_cfg["actor_rollout_ref"]["rollout"]["agent"]["_target_"] == "verl.workers.config.AgentLoopConfig"
    assert captured_verl_cfg["actor_rollout_ref"]["rollout"]["prometheus"]["_target_"] == "verl.workers.config.PrometheusConfig"
    assert captured_verl_cfg["actor_rollout_ref"]["rollout"]["agent"]["num_workers"] == 0
    assert captured_verl_cfg["actor_rollout_ref"]["rollout"]["prometheus"]["enable"] is False


def test_eval_portal_sglang_dp_falls_back_to_one_by_default(monkeypatch):
    import rllm.eval.portal as portal_module
    import rllm.engine.agent_execution_engine as exec_engine_module
    from rllm.eval.registry import EnvConfigRegistry

    captured = {}
    captured_verl_cfg = {}

    class DummyEngine:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    def fake_get(_env_name):
        return SimpleNamespace(
            env_args={},
            agent_args={},
            agent_class=_DummyAgent,
            env_class=_DummyEnv,
        )

    def fake_setup_verl_rollout_manager(cfg):
        captured_verl_cfg.update(cfg)
        return "mock_rollout_manager", "mock_verl_config"

    monkeypatch.setattr(EnvConfigRegistry, "get", staticmethod(fake_get))
    monkeypatch.setattr(portal_module, "_setup_verl_rollout_manager", fake_setup_verl_rollout_manager)
    monkeypatch.setattr(EvalPortal, "_get_tokenizer", lambda self: None)
    monkeypatch.setattr(exec_engine_module, "AgentExecutionEngine", DummyEngine)

    portal = EvalPortal(
        {
            "env": "sokoban",
            "engine": "verl",
            "model": {"name": "dummy-model"},
            "verl": {
                "rollout_name": "sglang",
                "tp_size": 4,
                "dp_size": 2,
            },
            "execution": {
                "n_parallel_agents": 2,
                "max_steps": 4,
                "max_response_length": 128,
                "max_prompt_length": 64,
                "max_workers": 2,
                "n_rollouts_per_prompt": 1,
            },
        }
    )

    portal.setup_engine()

    rollout_cfg = captured_verl_cfg["actor_rollout_ref"]["rollout"]
    assert rollout_cfg["name"] == "sglang"
    assert rollout_cfg["data_parallel_size"] == 1


def test_eval_portal_patches_verl_application_id_to_string(monkeypatch):
    import rllm.eval.portal as portal_module
    import rllm.engine.agent_execution_engine as exec_engine_module
    from rllm.eval.registry import EnvConfigRegistry

    class DummyEngine:
        def __init__(self, **_kwargs):
            pass

        async def get_model_response(self, _prompt, application_id, **_kwargs):
            return application_id

    def fake_get(_env_name):
        return SimpleNamespace(
            env_args={},
            agent_args={},
            agent_class=_DummyAgent,
            env_class=_DummyEnv,
        )

    monkeypatch.setattr(EnvConfigRegistry, "get", staticmethod(fake_get))
    monkeypatch.setattr(
        portal_module,
        "_setup_verl_rollout_manager",
        lambda _cfg: ("mock_rollout_manager", "mock_verl_config"),
    )
    monkeypatch.setattr(EvalPortal, "_get_tokenizer", lambda self: None)
    monkeypatch.setattr(exec_engine_module, "AgentExecutionEngine", DummyEngine)

    portal = EvalPortal(
        {
            "env": "sokoban",
            "engine": "verl",
            "model": {"name": "dummy-model"},
            "execution": {
                "n_parallel_agents": 2,
                "max_steps": 4,
                "max_response_length": 128,
                "max_prompt_length": 64,
                "max_workers": 2,
                "n_rollouts_per_prompt": 1,
            },
        }
    )

    portal.setup_engine()
    result = asyncio.run(portal.engine.get_model_response([], application_id=123))
    assert result == "123"
