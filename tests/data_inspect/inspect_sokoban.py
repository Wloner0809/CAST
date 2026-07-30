# %%
import pandas as pd
import random
import json
from pathlib import Path
from rllm.environments.sokoban.sokoban import SokobanEnv

# %%
# Load data from parquet file
data_path = Path(__file__).parent.parent.parent / "data" / "datasets" / "sokoban" / "train.parquet"
print(data_path)
df = pd.read_parquet(data_path)

print(f"Total number of examples: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# %%
from pprint import pprint
pprint(df.head(10).to_dict())
# %%
# Randomly sample some examples and show the sokoban map by SokobanEnv
num_examples = 5
random_indices = random.sample(range(len(df)), min(num_examples, len(df)))

print(f"\n{'='*80}")
print(f"Randomly sampled {len(random_indices)} Sokoban maps:")
print(f"{'='*80}\n")

for i, idx in enumerate(random_indices, 1):
    example = df.iloc[idx]
    print(f"Example {i} (index {idx}):")
    print(f"{'-'*80}")
    
    # Extract env_info from the example
    # The env config could be in 'extra_info' field or at the top level
    if 'extra_info' in example and isinstance(example['extra_info'], dict):
        env_info = example['extra_info']
    else:
        # Try to extract env-related fields from the example directly
        env_info = example.to_dict()
    print(env_info)
    # Print env configuration
    print(f"Env Info: seed={env_info.get('seed', 'N/A')}, "
          f"dim_room={env_info.get('dim_room', env_info.get('dim_x', 'N/A'))}, "
          f"num_boxes={env_info.get('num_boxes', 'N/A')}")
    
    # Create the environment and render the map
    try:
        env = SokobanEnv.from_dict(env_info)
        obs, _ = env.reset()
        print(f"\nSokoban Map (grid view):")
        print(obs)
        
        # Also show coordinate view
        print(f"\nCoordinate view:")
        print(env.render("coord"))
    except Exception as e:
        print(f"Failed to create environment: {e}")
    
    print(f"\n{'='*80}\n")

# %%
