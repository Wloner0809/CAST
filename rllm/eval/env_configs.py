"""Pre-registered environment configurations.

This module registers all supported environments with their default
configurations. It is automatically imported when using the EvalPortal.
"""

import logging

from rllm.eval.registry import EnvConfigRegistry, EnvConfig
from rllm.eval.data_loaders import (
    load_webshop_data,
    load_alfworld_data,
    load_sokoban_data,
    load_minesweeper_data,
    load_rush_hour_data,
)

logger = logging.getLogger(__name__)


def _safe_import(module_path: str, class_name: str):
    """Safely import a class from a module.

    Args:
        module_path: Full module path.
        class_name: Name of the class to import.

    Returns:
        The imported class, or None if import fails.
    """
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        # Keep debug for common "optional dependency missing" cases.
        logger.debug(f"Could not import {class_name} from {module_path}: {e}")
        return None
    except Exception as e:
        # If module import raises a non-ImportError (e.g., missing shared libs, runtime init),
        # don't fail the whole registry import — but DO surface the root cause.
        logger.warning(
            "Failed to import %s from %s; environment may not be registered. Error: %s",
            class_name,
            module_path,
            e,
            exc_info=True,
        )
        return None


def register_all_envs() -> None:
    """Register all supported environments."""

    # WebShop
    WebShopEnv = _safe_import("rllm.environments.webshop.webshop_env", "WebShopEnv")
    WebShopAgent = _safe_import("rllm.agents.webshop_agent", "WebShopAgent")
    if WebShopEnv and WebShopAgent:
        EnvConfigRegistry.register(
            "webshop",
            EnvConfig(
                env_class=WebShopEnv,
                agent_class=WebShopAgent,
                env_args={"observation_mode": "text", "max_steps": 50},
                agent_args={},
                data_loader=load_webshop_data,
                metrics=["pass_at_k", "success_rate", "avg_reward", "avg_steps"],
                requires_multithread_safe=True,
                default_max_steps=50,
                description="E-commerce shopping simulation environment",
            ),
        )
        logger.debug("Registered webshop environment")

    # ALFWorld
    ALFWorldEnv = _safe_import("rllm.environments.alfworld.alfworld_env", "ALFWorldEnv")
    ALFWorldAgent = _safe_import("rllm.agents.alfworld_agent", "ALFWorldAgent")
    if ALFWorldEnv and ALFWorldAgent:
        EnvConfigRegistry.register(
            "alfworld",
            EnvConfig(
                env_class=ALFWorldEnv,
                agent_class=ALFWorldAgent,
                env_args={"max_steps": 50},
                agent_args={"use_admissible_commands": True, "use_accumulate_history": True},
                data_loader=load_alfworld_data,
                metrics=["pass_at_k", "success_rate", "avg_steps"],
                requires_multithread_safe=True,
                default_max_steps=50,
                description="Text-based household task environment",
            ),
        )
        logger.debug("Registered alfworld environment")

    # Sokoban
    SokobanEnv = _safe_import("rllm.environments.sokoban.sokoban", "SokobanEnv")
    SokobanAgent = _safe_import("rllm.agents.sokoban_agent", "SokobanAgent")
    if SokobanEnv and SokobanAgent:
        EnvConfigRegistry.register(
            "sokoban",
            EnvConfig(
                env_class=SokobanEnv,
                agent_class=SokobanAgent,
                env_args={},
                agent_args={"max_steps": 30, "use_accumulate_history": True},
                data_loader=load_sokoban_data,
                metrics=["pass_at_k", "success_rate", "avg_steps"],
                requires_multithread_safe=True,
                default_max_steps=30,
                description="Sokoban puzzle game environment",
                success_threshold=5.0,
            ),
        )
        logger.debug("Registered sokoban environment")

    # Minesweeper
    MinesweeperEnv = _safe_import(
        "rllm.environments.minesweeper.minesweeper_env", "MinesweeperEnv"
    )
    MinesweeperAgent = _safe_import(
        "rllm.agents.minesweeper_agent", "MinesweeperAgent"
    )
    if MinesweeperEnv and MinesweeperAgent:
        EnvConfigRegistry.register(
            "minesweeper",
            EnvConfig(
                env_class=MinesweeperEnv,
                agent_class=MinesweeperAgent,
                env_args={},
                agent_args={"max_steps": 25, "use_accumulate_history": True},
                data_loader=load_minesweeper_data,
                metrics=["pass_at_k", "success_rate", "avg_steps", "avg_reward"],
                requires_multithread_safe=True,
                default_max_steps=25,
                description="Minesweeper grid puzzle environment",
                success_threshold=0.0,
            ),
        )
        logger.debug("Registered minesweeper environment")

    # Rush Hour
    RushHourEnv = _safe_import("rllm.environments.rush_hour.rush_hour_env", "RushHourEnv")
    RushHourAgent = _safe_import("rllm.agents.rush_hour_agent", "RushHourAgent")
    if RushHourEnv and RushHourAgent:
        EnvConfigRegistry.register(
            "rush_hour",
            EnvConfig(
                env_class=RushHourEnv,
                agent_class=RushHourAgent,
                env_args={},
                agent_args={"max_steps": 30, "use_accumulate_history": True},
                data_loader=load_rush_hour_data,
                metrics=["pass_at_k", "success_rate", "avg_steps", "avg_reward"],
                requires_multithread_safe=True,
                default_max_steps=30,
                description="Rush Hour sliding-block puzzle environment",
                success_threshold=0.0,
            ),
        )
        logger.debug("Registered rush_hour environment")


# Auto-register on import
register_all_envs()
