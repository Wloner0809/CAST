# %%
import pandas as pd
import random
from pathlib import Path
from pprint import pprint

from rllm.environments.minesweeper.minesweeper_env import MinesweeperEnv, DIFFICULTY_CONFIGS

# %%
# Load data from parquet file
data_path = Path(__file__).parent.parent.parent / "data" / "datasets" / "minesweeper" / "train.parquet"
print(f"Data path: {data_path}")
df = pd.read_parquet(data_path)

print(f"\nTotal number of examples: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# %%
# Show first 10 examples
print("\nFirst 10 examples:")
pprint(df.head(10).to_dict())

# %%
# Show difficulty distribution
if "difficulty" in df.columns:
    print("\nDifficulty Distribution:")
    print(df["difficulty"].value_counts())

# %%
# Show difficulty configs
print("\nDifficulty Configurations:")
for name, config in DIFFICULTY_CONFIGS.items():
    print(f"  {name}: {config['rows']}x{config['cols']}, {config['num_mines']} mines, {config['max_steps']} max_steps")

# %%
# Randomly sample some examples and create MinesweeperEnv instances
num_examples = 5
random_indices = random.sample(range(len(df)), min(num_examples, len(df)))

print(f"\n{'='*80}")
print(f"Randomly sampled {len(random_indices)} Minesweeper examples:")
print(f"{'='*80}\n")

for i, idx in enumerate(random_indices, 1):
    example = df.iloc[idx]
    print(f"Example {i} (index {idx}):")
    print(f"{'-'*80}")

    # Extract env_info from the example
    env_info = example.to_dict()
    print(f"Env Info: seed={env_info.get('seed', 'N/A')}, "
          f"rows={env_info.get('rows', 'N/A')}, "
          f"cols={env_info.get('cols', 'N/A')}, "
          f"num_mines={env_info.get('num_mines', 'N/A')}, "
          f"difficulty={env_info.get('difficulty', 'N/A')}")

    # Create the environment and render the board
    try:
        env = MinesweeperEnv.from_dict(env_info)
        obs, info = env.reset()
        print(f"\nInitial board (all unrevealed):")
        print(env.render())

        # Take first reveal action at center
        rows = env_info.get("rows", 8)
        cols = env_info.get("cols", 8)
        center_r, center_c = rows // 2, cols // 2

        print(f"\nFirst reveal at ({center_r}, {center_c}):")
        obs, reward, done, step_info = env.step(f"reveal {center_r} {center_c}")
        print(env.render())
        print(f"  Reward: {reward:.4f}")
        print(f"  Done: {done}")
        print(f"  Success: {step_info.get('success', False)}")

        # Take a second action (flag a random unrevealed cell)
        if not done:
            for r in range(rows):
                for c in range(cols):
                    if not env.revealed[r][c] and not env.flags[r][c]:
                        print(f"\nFlag at ({r}, {c}):")
                        obs, reward, done, step_info = env.step(f"flag {r} {c}")
                        print(env.render())
                        print(f"  Reward: {reward:.4f}")
                        break
                else:
                    continue
                break

        env.close()

    except Exception as e:
        print(f"  Failed to create environment: {e}")

    print(f"\n{'='*80}\n")

# %%
print("Inspection complete!")

# %%
