# %%
import pandas as pd
import random
from pathlib import Path
from pprint import pprint

from rllm.environments.webshop.webshop_env import (
    RENDER_INSTRUCTIONS,
    WebShopEnv,
)

# %%
# Load data from parquet files
data_dir = Path(__file__).parent.parent.parent / "data" / "datasets" / "webshop"
test_path = data_dir / "test.parquet"
train_path = data_dir / "train.parquet"

print(f"Test data path: {test_path}")
df_test = pd.read_parquet(test_path)
df_train = pd.read_parquet(train_path)

print(f"\nTest set: {len(df_test)} examples")
print(f"Train set: {len(df_train)} examples")
print(f"Columns: {df_test.columns.tolist()}")

# %%
# Show first 10 examples
print("\nFirst 10 test examples:")
pprint(df_test.head(10).to_dict())

# %%
# Show column statistics
print("\nDataset Statistics (test):")
for col in df_test.columns:
    print(f"\n  {col}:")
    if df_test[col].dtype == "object":
        unique_count = df_test[col].nunique()
        print(f"    Unique values: {unique_count}")
        if unique_count < 20:
            print(f"    Values: {df_test[col].unique().tolist()}")
    else:
        print(f"    Min: {df_test[col].min()}, Max: {df_test[col].max()}, Mean: {df_test[col].mean():.2f}")

# %%
# Also inspect verl format
verl_test_path = data_dir / "test_verl.parquet"
if verl_test_path.exists():
    df_verl = pd.read_parquet(verl_test_path)
    print(f"\nVeRL test format: {len(df_verl)} rows, columns: {df_verl.columns.tolist()}")
    print("\nFirst example extra_info:")
    pprint(df_verl.iloc[0]["extra_info"])

# %%
# Show RENDER_INSTRUCTIONS constant
print("\n" + "=" * 80)
print("RENDER_INSTRUCTIONS:")
print("=" * 80)
print(RENDER_INSTRUCTIONS)

# %%
# Randomly sample some examples and create WebShopEnv instances
num_examples = 3
random_indices = random.sample(range(len(df_test)), min(num_examples, len(df_test)))

print(f"\n{'='*80}")
print(f"Randomly sampled {len(random_indices)} WebShop examples:")
print(f"{'='*80}\n")

for i, idx in enumerate(random_indices, 1):
    example = df_test.iloc[idx]
    print(f"Example {i} (index {idx}):")
    print(f"{'-'*80}")

    session_id = int(example["session_id"])
    print(f"  Session ID: {session_id}")
    print(f"  UID: {example['uid']}")
    print(f"  Split: {example['split']}")

    # Create the environment and interact with it
    try:
        env = WebShopEnv(
            observation_mode="text",
            max_steps=15,
            session_id=session_id,
            seed=42,
        )
        obs, info = env.reset()

        # Show instruction
        print(f"\n  Instruction: {info.get('instruction', 'N/A')}")

        # Show initial observation
        print(f"\n  Initial Observation:")
        print(f"  {'-'*60}")
        obs_text = obs if isinstance(obs, str) else str(obs)
        obs_lines = obs_text.strip().split("\n")
        for line in obs_lines[:10]:
            print(f"    {line}")
        if len(obs_lines) > 10:
            print(f"    ... ({len(obs_lines) - 10} more lines)")

        # Show available actions (raw dict)
        available = info.get("available_actions", {})
        print(f"\n  Search bar: {available.get('has_search_bar', False)}")
        clickables = available.get("clickables", [])
        print(f"  Clickables ({len(clickables)}): {clickables[:10]}")
        if len(clickables) > 10:
            print(f"    ... and {len(clickables) - 10} more")

        # Show formatted available actions (new method)
        formatted = env.get_available_actions_formatted()
        print(f"\n  Formatted actions ({len(formatted)}):")
        for action in formatted[:10]:
            print(f"    - {action}")
        if len(formatted) > 10:
            print(f"    ... and {len(formatted) - 10} more")

        # Show render() output (first 600 chars)
        rendered = env.render()
        print(f"\n  Rendered output (first 600 chars):")
        print(f"  {'-'*60}")
        rendered_display = rendered[:600] + "..." if len(rendered) > 600 else rendered
        for line in rendered_display.strip().split("\n"):
            print(f"    {line}")

        # Show success property
        print(f"\n  Success: {env.success}")

        # Take a search action
        if available.get("has_search_bar", False):
            instruction = info.get("instruction", "")
            search_words = instruction.split()[:3]
            search_action = f"search[{' '.join(search_words)}]"
            print(f"\n  Taking action: '{search_action}'")
            obs, reward, done, step_info = env.step(search_action)

            print(f"  Result:")
            obs_short = obs[:200] + "..." if len(obs) > 200 else obs
            print(f"    Observation: {obs_short}")
            print(f"    Reward: {reward}")
            print(f"    Done: {done}")
            print(f"    action_is_valid: {step_info.get('action_is_valid')}")
            print(f"    action_is_effective: {step_info.get('action_is_effective')}")
            print(f"    end_of_page: {step_info.get('end_of_page')}")
            print(f"    success_purchase: {step_info.get('success_purchase')}")
            print(f"    raw_reward: {step_info.get('raw_reward')}")
            print(f"    task_score: {step_info.get('task_score')}")
            print(f"    won: {step_info.get('won')}")

            # Click on first product if available
            new_clickables = step_info.get("available_actions", {}).get("clickables", [])
            if new_clickables:
                click_target = new_clickables[0]
                click_action = f"click[{click_target}]"
                print(f"\n  Taking action: '{click_action}'")
                obs, reward, done, step_info = env.step(click_action)
                obs_short = obs[:200] + "..." if len(obs) > 200 else obs
                print(f"  Result:")
                print(f"    Observation: {obs_short}")
                print(f"    Reward: {reward}")
                print(f"    Done: {done}")
                print(f"    action_is_valid: {step_info.get('action_is_valid')}")
                print(f"    action_is_effective: {step_info.get('action_is_effective')}")

        env.close()

    except Exception as e:
        print(f"  Failed to create/run environment: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*80}\n")

# %%
# Cleanup: Shutdown the worker pool
print("Shutting down WebShop worker pool...")
WebShopEnv.shutdown_pool()
print("Done!")

# %%
