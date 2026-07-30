import pytest

from rllm.agents.action_utils import parse_action_block
from rllm.agents.minesweeper_agent import MinesweeperAgent
from rllm.environments.minesweeper.minesweeper_env import MinesweeperEnv


# ---------------------------------------------------------------------------
# Basic creation and reset
# ---------------------------------------------------------------------------

def test_init_default():
    """Test agent creation with default parameters."""
    agent = MinesweeperAgent(max_steps=25)
    assert agent is not None
    assert agent.max_steps == 25
    assert agent.step == 0
    assert len(agent.messages) == 1  # System prompt only
    assert agent.messages[0]["role"] == "system"
    assert len(agent.trajectory.steps) == 0


def test_init_accumulate_options():
    """Test agent creation with accumulate options."""
    agent = MinesweeperAgent(
        max_steps=10,
        use_accumulate_thinking=False,
        use_accumulate_history=False,
    )
    assert agent.accumulate_thinking is False
    assert agent.accumulate_history is False


def test_reset():
    """Test agent reset clears state properly."""
    agent = MinesweeperAgent(max_steps=25)

    # Simulate some state
    agent.step = 5
    agent.messages.append({"role": "user", "content": "test"})

    agent.reset()

    assert agent.step == 0
    assert len(agent.messages) == 1  # Only system prompt
    assert len(agent.trajectory.steps) == 0


# ---------------------------------------------------------------------------
# update_from_env
# ---------------------------------------------------------------------------

def test_update_from_env_basic():
    """Test that update_from_env adds a user message with board state."""
    agent = MinesweeperAgent(max_steps=25)
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3, seed=42)
    obs, info = env.reset()

    agent.update_from_env(obs, 0.0, False, info)

    assert len(agent.messages) == 2  # System + user
    assert agent.messages[-1]["role"] == "user"
    # Should contain board info from suffix
    assert "board layout" in agent.messages[-1]["content"].lower() or "." in agent.messages[-1]["content"]


def test_update_from_env_includes_step_number():
    """Test that user message includes step number."""
    agent = MinesweeperAgent(max_steps=25)
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3, seed=42)
    obs, info = env.reset()

    agent.update_from_env(obs, 0.0, False, info)

    # The new GameAgent uses "Current Observation (step N):" format
    assert "step 0" in agent.messages[-1]["content"].lower()


def test_update_from_env_includes_remaining_steps():
    """Test that user message includes remaining steps."""
    agent = MinesweeperAgent(max_steps=25)
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3, seed=42)
    obs, info = env.reset()

    agent.update_from_env(obs, 0.0, False, info)

    # The new GameAgent uses "steps remaining" instead of "turns remaining"
    assert "25 steps remaining" in agent.messages[-1]["content"]


def test_update_from_env_format_error_message():
    """Test that format error feedback is included in next user message."""
    agent = MinesweeperAgent(max_steps=25)
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3, seed=42)
    obs, info = env.reset()

    # First update
    agent.update_from_env(obs, 0.0, False, info)

    # Simulate a model response that will give a valid action
    agent.update_from_model("```action\nreveal 2 2\n```")

    # Now simulate the env giving a format error on next step
    # The new API uses action_is_valid=False (not action_invalid=True)
    info_with_error = {"suffix": "board here", "format_error": True, "action_is_valid": False}
    agent.update_from_env("error obs", 0.0, False, info_with_error)

    last_msg = agent.messages[-1]["content"]
    assert "could not be parsed" in last_msg.lower()


def test_update_from_env_already_revealed_message():
    """Test that already-revealed feedback is included in next user message."""
    agent = MinesweeperAgent(max_steps=25)
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3, seed=42)
    obs, info = env.reset()

    # First update
    agent.update_from_env(obs, 0.0, False, info)

    # Simulate a model response
    agent.update_from_model("```action\nreveal 2 2\n```")

    # Now simulate the env giving already_revealed
    # The new API uses action_is_valid=False (not action_invalid=True)
    info_with_already = {"suffix": "board here", "already_revealed": True, "action_is_valid": False}
    agent.update_from_env("already obs", 0.0, False, info_with_already)

    last_msg = agent.messages[-1]["content"]
    assert "already revealed" in last_msg.lower()


def test_update_from_env_out_of_bounds_message():
    """Test that out-of-bounds feedback is included in next user message."""
    agent = MinesweeperAgent(max_steps=25)
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3, seed=42)
    obs, info = env.reset()

    # First update
    agent.update_from_env(obs, 0.0, False, info)

    # Simulate a model response
    agent.update_from_model("```action\nreveal 2 2\n```")

    # Now simulate the env giving out_of_bounds
    # The new API uses action_is_valid=False (not action_invalid=True)
    info_with_oob = {"suffix": "board here", "out_of_bounds": True, "action_is_valid": False}
    agent.update_from_env("oob obs", 0.0, False, info_with_oob)

    last_msg = agent.messages[-1]["content"]
    assert "outside the board" in last_msg.lower() or "outside" in last_msg.lower()


# ---------------------------------------------------------------------------
# update_from_model
# ---------------------------------------------------------------------------

def test_update_from_model_reveal():
    """Test parsing reveal action from model response."""
    agent = MinesweeperAgent(max_steps=25)
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3, seed=42)
    obs, info = env.reset()
    agent.update_from_env(obs, 0.0, False, info)

    action = agent.update_from_model(
        "I'll reveal cell (2, 3). ```action\nreveal 2 3\n```"
    )

    assert action.action == "reveal 2 3"
    assert agent.step == 1
    assert len(agent.trajectory.steps) == 1


def test_update_from_model_flag():
    """Test parsing flag action from model response."""
    agent = MinesweeperAgent(max_steps=25)
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3, seed=42)
    obs, info = env.reset()
    agent.update_from_env(obs, 0.0, False, info)

    action = agent.update_from_model(
        "I suspect a mine here. ```action\nflag 1 4\n```"
    )

    assert action.action == "flag 1 4"


def test_update_from_model_no_action():
    """Test handling when no valid action is found in response."""
    agent = MinesweeperAgent(max_steps=25)
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3, seed=42)
    obs, info = env.reset()
    agent.update_from_env(obs, 0.0, False, info)

    action = agent.update_from_model(
        "I'm not sure what to do here."
    )

    assert action.action == ""


def test_update_from_model_last_match_wins():
    """Test that when multiple action blocks exist, the last one wins."""
    agent = MinesweeperAgent(max_steps=25)
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3, seed=42)
    obs, info = env.reset()
    agent.update_from_env(obs, 0.0, False, info)

    action = agent.update_from_model(
        "First I thought ```action\nreveal 0 0\n``` but actually ```action\nreveal 3 4\n```"
    )

    assert action.action == "reveal 3 4"


def test_update_from_model_case_insensitive():
    """Test that action parsing is case-insensitive."""
    agent = MinesweeperAgent(max_steps=25)
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3, seed=42)
    obs, info = env.reset()
    agent.update_from_env(obs, 0.0, False, info)

    action = agent.update_from_model("```action\nREVEAL 2 3\n```")
    assert action.action == "reveal 2 3"

    agent2 = MinesweeperAgent(max_steps=25)
    agent2.update_from_env(obs, 0.0, False, info)
    action2 = agent2.update_from_model("```action\nFlag 1 1\n```")
    assert action2.action == "flag 1 1"


# ---------------------------------------------------------------------------
# parse_action_block + _parse_action (replaces _parse_model_response)
# ---------------------------------------------------------------------------

def test_parse_action_block_reveal():
    """Test parsing a reveal action via parse_action_block + _parse_action."""
    agent = MinesweeperAgent(max_steps=25)
    thought, raw_action = parse_action_block("Thinking... ```action\nreveal 3 2\n```")
    assert raw_action is not None
    action = agent._parse_action(raw_action)
    assert action == "reveal 3 2"
    assert "Thinking" in thought


def test_parse_action_block_flag():
    """Test parsing a flag action via parse_action_block + _parse_action."""
    agent = MinesweeperAgent(max_steps=25)
    thought, raw_action = parse_action_block("This cell is dangerous ```action\nflag 0 4\n```")
    assert raw_action is not None
    action = agent._parse_action(raw_action)
    assert action == "flag 0 4"


def test_parse_action_block_no_action():
    """Test parsing when no valid action is present."""
    agent = MinesweeperAgent(max_steps=25)
    thought, raw_action = parse_action_block("I don't know what to do")
    # No action block found -> raw_action is None
    assert raw_action is None
    action = agent._parse_action("")
    assert action == ""


def test_parse_action_block_invalid_in_backticks():
    """Test parsing when backticks contain invalid content."""
    agent = MinesweeperAgent(max_steps=25)
    thought, raw_action = parse_action_block("```action\nsome random text\n```")
    assert raw_action is not None
    action = agent._parse_action(raw_action)
    assert action == ""


def test_parse_action_block_with_think_tags():
    """Test that think tags don't interfere with action parsing."""
    agent = MinesweeperAgent(max_steps=25)
    response = "<think>Let me think about this carefully</think>Based on analysis ```action\nreveal 1 1\n```"
    thought, raw_action = parse_action_block(response)
    assert raw_action is not None
    action = agent._parse_action(raw_action)
    assert action == "reveal 1 1"


# ---------------------------------------------------------------------------
# chat_completions property
# ---------------------------------------------------------------------------

def test_chat_completions_with_history():
    """Test chat_completions returns full history when accumulate_history=True."""
    agent = MinesweeperAgent(max_steps=25, use_accumulate_history=True)
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3, seed=42)
    obs, info = env.reset()

    agent.update_from_env(obs, 0.0, False, info)
    agent.update_from_model("```action\nreveal 2 2\n```")

    completions = agent.chat_completions
    assert len(completions) == 3  # system + user + assistant


def test_chat_completions_without_history():
    """Test chat_completions returns system + last user-assistant pair when accumulate_history=False."""
    agent = MinesweeperAgent(max_steps=25, use_accumulate_history=False)
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3, seed=42)
    obs, info = env.reset()

    # After update_from_env: [system, user]
    agent.update_from_env(obs, 0.0, False, info)
    completions = agent.chat_completions
    assert len(completions) == 2  # system + user
    assert completions[0]["role"] == "system"
    assert completions[1]["role"] == "user"

    # After update_from_model: [system, user, assistant]
    agent.update_from_model("```action\nreveal 2 2\n```")
    completions = agent.chat_completions
    assert len(completions) == 3  # system + user + assistant
    assert completions[0]["role"] == "system"
    assert completions[1]["role"] == "user"
    assert completions[2]["role"] == "assistant"

    # After another update_from_env: [system, user_new] (previous pair dropped)
    obs2, _, _, info2 = env.step("reveal 2 2")
    agent.update_from_env(obs2, 0.5, False, info2)
    completions = agent.chat_completions
    assert len(completions) == 2  # system + new user message
    assert completions[0]["role"] == "system"
    assert completions[1]["role"] == "user"


# ---------------------------------------------------------------------------
# trajectory property
# ---------------------------------------------------------------------------

def test_trajectory_tracks_steps():
    """Test that trajectory properly tracks steps through agent-env interaction."""
    agent = MinesweeperAgent(max_steps=25)
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3, seed=42)
    obs, info = env.reset()

    # Step 1
    agent.update_from_env(obs, 0.0, False, info)
    agent.update_from_model("```action\nreveal 2 2\n```")

    # Step 2
    obs2, reward2, done2, info2 = env.step("reveal 2 2")
    agent.update_from_env(obs2, reward2, done2, info2)
    agent.update_from_model("```action\nreveal 0 0\n```")

    traj = agent.trajectory
    assert len(traj.steps) == 2
    assert traj.steps[0].action == "reveal 2 2"
    assert traj.steps[1].action == "reveal 0 0"


# ---------------------------------------------------------------------------
# Full agent-environment interaction
# ---------------------------------------------------------------------------

def test_full_agent_env_interaction():
    """Test a complete agent-environment interaction loop."""
    agent = MinesweeperAgent(max_steps=10)
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3, seed=42)
    obs, info = env.reset()

    done = False
    reward = 0.0
    total_reward = 0.0
    steps = 0

    while not done and steps < 5:
        agent.update_from_env(obs, reward, done, info)
        action = agent.update_from_model(f"```action\nreveal {steps} {steps}\n```")

        obs, reward, done, info = env.step(action.action)
        total_reward += reward
        steps += 1

    # Verify the agent tracked all steps
    assert len(agent.trajectory.steps) == steps
    assert agent.step == steps


def test_system_prompt_content():
    """Test that system prompt contains expected content."""
    agent = MinesweeperAgent(max_steps=25)
    prompt = agent.SYSTEM_PROMPT

    assert "Minesweeper" in prompt
    assert "reveal" in prompt
    assert "flag" in prompt
    assert "mine" in prompt.lower()
    assert "```" in prompt  # Action format uses ``` blocks


# ---------------------------------------------------------------------------
# Backward compatibility: generic ``` ``` blocks also work (fallback)
# ---------------------------------------------------------------------------

def test_update_from_model_generic_backtick_fallback():
    """Test that generic ``` ``` blocks are accepted via fallback parsing."""
    agent = MinesweeperAgent(max_steps=25)
    env = MinesweeperEnv(rows=5, cols=5, num_mines=3, seed=42)
    obs, info = env.reset()
    agent.update_from_env(obs, 0.0, False, info)

    # Use generic backticks (no "action" label) -- parse_action_block has a
    # fallback for these so they should still work.
    action = agent.update_from_model("I'll try this ```reveal 2 3```")
    assert action.action == "reveal 2 3"
