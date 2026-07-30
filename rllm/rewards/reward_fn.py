import re
import string
from collections import Counter
from typing import Protocol, runtime_checkable

from rllm.agents.agent import Action
from rllm.rewards.code_reward import RewardCodeFn
from rllm.rewards.math_reward import RewardMathFn
from rllm.rewards.reward_types import RewardConfig, RewardInput, RewardOutput
from rllm.rewards.search_reward import RewardSearchFn


@runtime_checkable
class RewardFunction(Protocol):
    """Protocol for reward functions"""

    def __call__(self, task_info: dict, action: str) -> RewardOutput:
        """
        Calculate the reward for an agent's action.

        Args:
            task_info: The task dictionary containing question, answer, and other metadata
            action: The agent's response/solution

        Returns:
            RewardOutput: The calculated reward value, either as a float or RewardOutput object
        """
        ...


# Simple example implementation
def zero_reward(task_info: dict, action: str) -> RewardOutput:
    """
    A simple reward function that always returns zero.
    Useful as a placeholder when no specific reward logic is needed.

    Args:
        task: The task dictionary
        action: The agent's response

    Returns:
        float: Always returns 0.0
    """
    return RewardOutput(reward=0.0, metadata={})


def math_verify_reward_fn(task_info: dict, action: str) -> RewardOutput:
    """
    A reward function for math tasks using the math_verify package.

    Uses math_verify.parse() + math_verify.verify() for answer grading,
    which is the standard evaluator for AIME/MATH benchmarks
    (reward_model.style = "rule-lighteval/MATH_v2").

    Args:
        task_info: The task dictionary containing ground_truth and other metadata
        action: The agent's response/solution

    Returns:
        RewardOutput: reward=1.0 if correct, 0.0 otherwise
    """
    import threading

    from math_verify import parse, verify

    from rllm.globals import THOUGHT_DELIMITER_END
    from rllm.rewards.math_utils.utils import extract_answer

    if isinstance(action, Action):
        action = action.action

    if not action:
        return RewardOutput(reward=0.0, is_correct=False)

    # Strip <think>...</think> prefix
    model_solution = action
    if THOUGHT_DELIMITER_END in model_solution:
        model_solution = model_solution.split(THOUGHT_DELIMITER_END, 1)[1]

    # Extract answer from \boxed{}
    model_answer = extract_answer(model_solution)
    if model_answer is None:
        return RewardOutput(reward=0.0, is_correct=False)

    # Get ground truth(s)
    ground_truths = task_info.get("ground_truth", None)
    if ground_truths is None:
        return RewardOutput(reward=0.0, is_correct=False)

    if isinstance(ground_truths, (str, float, int)):
        ground_truths = [ground_truths]

    # math_verify uses signal.SIGALRM for timeouts, which only works in the
    # main thread.  The eval engine (and verl trainer) call env.step() inside
    # ThreadPoolExecutor workers, so we must disable signal-based timeouts
    # there to avoid ValueError.  In the main thread we keep the defaults so
    # that pathological expressions cannot hang the process.
    in_main_thread = threading.current_thread() is threading.main_thread()
    parse_kwargs = {} if in_main_thread else {"parsing_timeout": None}
    verify_kwargs = {} if in_main_thread else {"timeout_seconds": None}

    # Check against all possible correct answers using math_verify
    for gt in ground_truths:
        gt = str(gt)
        # Extract from \boxed{} if ground truth is in that format
        if "\\boxed" in gt:
            extracted_gt = extract_answer(gt)
            if extracted_gt is not None:
                gt = extracted_gt

        try:
            # Wrap in $ for math_verify parser (required for LaTeX parsing)
            pred_str = f"${model_answer}$" if not model_answer.startswith("$") else model_answer
            gt_str = f"${gt}$" if not gt.startswith("$") else gt

            pred_parsed = parse(pred_str, **parse_kwargs)
            gt_parsed = parse(gt_str, **parse_kwargs)

            if verify(pred_parsed, gt_parsed, **verify_kwargs):
                return RewardOutput(reward=1.0, is_correct=True)
        except Exception:
            # math_verify can fail on unusual expressions; continue to next ground truth
            continue

    return RewardOutput(reward=0.0, is_correct=False)


def math_reward_fn(task_info: dict, action: str) -> RewardOutput:
    """
    A reward function for math tasks that implements the RewardFunction protocol.

    Args:
        task: The task dictionary containing data_source, ground_truth and other metadata
        action: The agent's response/solution

    Returns:
        float: The calculated reward value based on math evaluation
    """
    reward_config = RewardConfig()
    reward_fn = RewardMathFn(reward_config)
    if isinstance(action, Action):
        action = action.action
    return reward_fn(task_info, action)


def search_reward_fn(task_info: dict, action: str) -> RewardOutput:
    """
    A reward function for search tasks that implements the RewardFunction protocol.

    Args:
        task_info: The task dictionary containing data_source, ground_truth and other metadata
        action: The agent's response/solution

    Returns:
        RewardOutput: The calculated reward value based on search evaluation
    """
    reward_config = RewardConfig()
    reward_fn = RewardSearchFn(reward_config)
    if isinstance(action, Action):
        action = action.action

    # Create RewardInput from task_info and action
    reward_input = RewardInput(task_info=task_info, action=action)

    return reward_fn(reward_input)


def code_reward_fn(task_info: dict, action: str) -> RewardOutput:
    """
    A reward function for code tasks that implements the RewardFunction protocol.

    Args:
        task: The task dictionary containing data_source, ground_truth and other metadata
        action: The agent's response/solution

    Returns:
        float: The calculated reward value based on code execution results
    """
    reward_config = RewardConfig()
    reward_fn = RewardCodeFn(reward_config)
    if isinstance(action, Action):
        action = action.action
    return reward_fn(task_info, action)


def f1_reward_fn(task_info: dict, action: str) -> RewardOutput:
    """
    A reward function that computes F1 score between predicted text and gold text.

    This function normalizes both texts (lowercase, remove punctuation, remove articles,
    fix whitespace) before tokenizing and computing F1 score based on token overlap.

    Args:
        task_info: The task dictionary containing ground_truth (gold text)
        action: The agent's predicted response/solution

    Returns:
        RewardOutput: The calculated reward value (F1 score)

    Example:
        >>> task_info = {"ground_truth": "Hello, world!"}
        >>> action = "hello there world"
        >>> output = f1_reward_fn(task_info, action)
        >>> print(output.reward)  # F1 score between the texts
    """
    if isinstance(action, Action):
        action = action.action

    # Extract gold text from task_info
    gold_text = task_info.get("ground_truth", "")
    if gold_text is None:
        gold_text = ""

    def normalize_text(s: str) -> str:
        """Normalize text for evaluation (following HotpotQA/SQuAD standards)"""

        def remove_articles(text: str) -> str:
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text: str) -> str:
            return " ".join(text.split())

        def remove_punc(text: str) -> str:
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text: str) -> str:
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    # Normalize and tokenize both texts
    predicted_normalized = normalize_text(str(action))
    gold_normalized = normalize_text(str(gold_text))
    predicted_tokens = predicted_normalized.split()
    gold_tokens = gold_normalized.split()

    # Handle empty cases - if neither predicted nor gold are passed, return 0
    if not predicted_tokens and not gold_tokens:
        f1_score = 0.0
    elif not predicted_tokens or not gold_tokens:
        f1_score = 0.0
    else:
        # Compute token overlap using Counter intersection
        predicted_counter = Counter(predicted_tokens)
        gold_counter = Counter(gold_tokens)
        common = predicted_counter & gold_counter
        num_same = sum(common.values())

        if num_same == 0:
            f1_score = 0.0
        else:
            precision = num_same / len(predicted_tokens)
            recall = num_same / len(gold_tokens)
            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return RewardOutput(reward=f1_score, metadata={})
