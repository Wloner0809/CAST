import asyncio

from transformers import AutoTokenizer

from rllm.agents.sokoban_agent import SokobanAgent
from rllm.data.dataset import DatasetRegistry
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.environments.sokoban.sokoban import SokobanEnv
from rllm.utils import compute_pass_at_k


def load_sokoban_data():
    if DatasetRegistry.dataset_exists("sokoban", "test"):
        test_dataset = DatasetRegistry.load_dataset("sokoban", "test")
        return test_dataset.get_data()

    print("Sokoban datasets not found. Preparing datasets...")
    from prepare_sokoban_data import prepare_sokoban_data

    _, test_dataset = prepare_sokoban_data()

    return test_dataset.get_data()


if __name__ == "__main__":
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    n_parallel_agents = 128
    model_name = "Qwen/Qwen3-4B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    sampling_params = {"temperature": 0.6, "top_p": 0.95, "model": model_name}

    agent_args = {
        "max_steps": 30,
        "use_accumulate_history": True,
    }

    env_args = {}

    engine = AgentExecutionEngine(
        agent_class=SokobanAgent,
        env_class=SokobanEnv,
        agent_args=agent_args,
        env_args=env_args,
        engine_name="openai",
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        rollout_engine_args={
            "base_url": "http://localhost:30000/v1",
            "api_key": "None",
        },
        max_response_length=16384,
        max_prompt_length=4096,
        n_parallel_agents=n_parallel_agents,
    )

    tasks = load_sokoban_data()

    results = asyncio.run(engine.execute_tasks(tasks))
    compute_pass_at_k(results, success_threshold=5.0, env_name="sokoban")
