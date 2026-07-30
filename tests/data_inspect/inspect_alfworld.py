# %%
import pandas as pd
import random
from pathlib import Path
from pprint import pprint
import glob
import os
# Point at a local Java 11 only when the caller has not already chosen one.
if not os.path.isdir(os.environ.get('JAVA_HOME', '')):
    _java11 = next(
        (p for p in sorted(glob.glob('/usr/lib/jvm/java-11-openjdk*'))
         + ['/usr/lib/jvm/java-11-openjdk-amd64', '/usr/lib/jvm/java-11']
         if os.path.isdir(p)),
        '',
    )
    if _java11:
        os.environ['JAVA_HOME'] = _java11
        os.environ['JVM_PATH'] = os.path.join(_java11, 'lib', 'server', 'libjvm.so')
        os.environ['PATH'] = os.path.join(_java11, 'bin') + ':' + os.environ.get('PATH', '')


from rllm.environments.alfworld.alfworld_env import ALFWorldEnv, TASK_TYPES

# %%
# Load data from parquet file
data_path = Path(__file__).parent.parent.parent / "data" / "datasets" / "alfworld" / "train.parquet"
print(f"Data path: {data_path}")
df = pd.read_parquet(data_path)

print(f"\nTotal number of examples: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# %%
# Show first 10 examples
print("\nFirst 10 examples:")
pprint(df.head(10).to_dict())

# %%
# Show task type distribution
print("\nTask Type Distribution:")
print(df['task_type'].value_counts())

# %%
# Randomly sample some examples and create ALFWorldEnv instances
num_examples = 3
random_indices = random.sample(range(len(df)), min(num_examples, len(df)))

print(f"\n{'='*80}")
print(f"Randomly sampled {len(random_indices)} ALFWorld examples:")
print(f"{'='*80}\n")

for i, idx in enumerate(random_indices, 1):
    example = df.iloc[idx]
    print(f"Example {i} (index {idx}):")
    print(f"{'-'*80}")

    # Print example info
    print(f"  Task ID: {example['task_id']}")
    print(f"  Task Type: {example['task_type']}")
    print(f"  Split: {example['split']}")
    print(f"  Game File: {example['game_file']}")

    # Create the environment and get initial observation
    try:
        env = ALFWorldEnv(
            game_file=example['game_file'],
            max_steps=50,
        )
        obs, info = env.reset()

        print(f"\n  Initial Observation:")
        print(f"  {'-'*60}")
        # Format the observation for better readability
        for line in obs.strip().split('\n'):
            print(f"    {line}")

        print(f"\n  Admissible Commands ({len(info['admissible_commands'])}):")
        for cmd in info['admissible_commands'][:10]:  # Show first 10 commands
            print(f"    - {cmd}")
        if len(info['admissible_commands']) > 10:
            print(f"    ... and {len(info['admissible_commands']) - 10} more")

        # Take a sample action if available
        if info['admissible_commands']:
            sample_action = info['admissible_commands'][0]
            print(f"\n  Taking sample action: '{sample_action}'")
            next_obs, reward, done, step_info = env.step(sample_action)
            print(f"  Result:")
            print(f"    Observation: {next_obs[:200]}..." if len(next_obs) > 200 else f"    Observation: {next_obs}")
            print(f"    Reward: {reward}")
            print(f"    Done: {done}")

        env.close()

    except Exception as e:
        print(f"  Failed to create environment: {e}")

    print(f"\n{'='*80}\n")

# %%
# Cleanup: Shutdown the worker pool
print("Shutting down ALFWorld worker pool...")
ALFWorldEnv.shutdown_pool()
print("Done!")

# %%
