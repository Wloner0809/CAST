from rllm.agents.action_utils import parse_action_block
from rllm.agents.agent import Action, BaseAgent, Episode, Step, Trajectory
from rllm.agents.context_manager import ContextManager
from rllm.agents.game_agent import GameAgent
from rllm.agents.math_agent import MathAgent
from rllm.agents.tool_agent import ToolAgent

__all__ = [
    "BaseAgent", "Action", "Step", "Trajectory", "Episode",
    "ContextManager", "GameAgent", "MathAgent", "ToolAgent", "parse_action_block",
]


def safe_import(module_path, class_name):
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except ImportError:
        return None


# Define all agent imports
AGENT_IMPORTS = [
    ("rllm.agents.sokoban_agent", "SokobanAgent"),
    ("rllm.agents.alfworld_agent", "ALFWorldAgent"),
    ("rllm.agents.webshop_agent", "WebShopAgent"),
    ("rllm.agents.minesweeper_agent", "MinesweeperAgent"),
    ("rllm.agents.rush_hour_agent", "RushHourAgent"),
]

for module_path, class_name in AGENT_IMPORTS:
    imported_class = safe_import(module_path, class_name)
    if imported_class is not None:
        globals()[class_name] = imported_class
        __all__.append(class_name)
