def safe_import(module_path, class_name):
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except (ImportError, AttributeError, ModuleNotFoundError):
        return None


# Import environment classes
ENV_CLASSES = {
    "sokoban": safe_import("rllm.environments.sokoban.sokoban", "SokobanEnv"),
    "minesweeper": safe_import("rllm.environments.minesweeper.minesweeper_env", "MinesweeperEnv"),
    "rush_hour": safe_import("rllm.environments.rush_hour.rush_hour_env", "RushHourEnv"),
    "alfworld": safe_import("rllm.environments.alfworld.alfworld_env", "ALFWorldEnv"),
    "webshop": safe_import("rllm.environments.webshop.webshop_env", "WebShopEnv"),
    "tool": safe_import("rllm.environments.tools.tool_env", "ToolEnvironment"),
    "math": safe_import("rllm.environments.base.single_turn_env", "SingleTurnEnvironment"),
    "code": safe_import("rllm.environments.base.single_turn_env", "SingleTurnEnvironment"),
    "single_turn_env": safe_import("rllm.environments.base.single_turn_env", "SingleTurnEnvironment"),
}

# Import agent classes
AGENT_CLASSES = {
    "sokobanagent": safe_import("rllm.agents.sokoban_agent", "SokobanAgent"),
    "minesweeper_agent": safe_import("rllm.agents.minesweeper_agent", "MinesweeperAgent"),
    "rush_hour_agent": safe_import("rllm.agents.rush_hour_agent", "RushHourAgent"),
    "alfworld_agent": safe_import("rllm.agents.alfworld_agent", "ALFWorldAgent"),
    "webshop_agent": safe_import("rllm.agents.webshop_agent", "WebShopAgent"),
    "tool_agent": safe_import("rllm.agents.tool_agent", "ToolAgent"),
    "math_agent": safe_import("rllm.agents.math_agent", "MathAgent"),
}

WORKFLOW_CLASSES = {
    "single_turn_workflow": safe_import("rllm.workflows.single_turn_workflow", "SingleTurnWorkflow"),
    "multi_turn_workflow": safe_import("rllm.workflows.multi_turn_workflow", "MultiTurnWorkflow"),
    "simple_workflow": safe_import("rllm.workflows.simple_workflow", "SimpleWorkflow"),
    "cumulative_workflow": safe_import("rllm.workflows.cumulative_workflow", "CumulativeWorkflow"),
}

# Filter out None values for unavailable imports
ENV_CLASS_MAPPING = {k: v for k, v in ENV_CLASSES.items() if v is not None}
AGENT_CLASS_MAPPING = {k: v for k, v in AGENT_CLASSES.items() if v is not None}
WORKFLOW_CLASS_MAPPING = {k: v for k, v in WORKFLOW_CLASSES.items() if v is not None}
