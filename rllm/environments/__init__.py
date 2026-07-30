from rllm.environments.base.base_env import BaseEnv
from rllm.environments.base.single_turn_env import SingleTurnEnvironment
from rllm.environments.tools.tool_env import ToolEnvironment

__all__ = ["BaseEnv", "SingleTurnEnvironment", "ToolEnvironment"]

_ENVIRONMENT_IMPORTS = [
    ("rllm.environments.sokoban.sokoban", "SokobanEnv"),
    ("rllm.environments.alfworld.alfworld_env", "ALFWorldEnv"),
    ("rllm.environments.webshop.webshop_env", "WebShopEnv"),
    ("rllm.environments.minesweeper.minesweeper_env", "MinesweeperEnv"),
    ("rllm.environments.rush_hour.rush_hour_env", "RushHourEnv"),
]

_LAZY_CLASS_TO_MODULE = {class_name: module_path for module_path, class_name in _ENVIRONMENT_IMPORTS}
__all__.extend([class_name for _, class_name in _ENVIRONMENT_IMPORTS])


def __getattr__(name):
    """Lazy-load optional environments to avoid import-time side effects."""
    module_path = _LAZY_CLASS_TO_MODULE.get(name)
    if module_path is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    module = __import__(module_path, fromlist=[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
