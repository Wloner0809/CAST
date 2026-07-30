import json

import numpy as np

from rllm.trainer.verl.agent_ppo_trainer import _to_json_compatible


def test_to_json_compatible_converts_nested_numpy_values():
    value = {
        "chat_completions": [
            {
                "content": np.array(["reasoning", "Right"], dtype=object),
                "token_ids": np.array([123, 456]),
            }
        ],
        "turn_advantages": np.array(
            [np.array([1.0, -2.0]), np.array([0.0])],
            dtype=object,
        ),
        "count": np.int64(2),
        "nested": (np.float32(0.5),),
    }

    converted = _to_json_compatible(value)

    assert converted == {
        "chat_completions": [
            {
                "content": ["reasoning", "Right"],
                "token_ids": [123, 456],
            }
        ],
        "turn_advantages": [[1.0, -2.0], [0.0]],
        "count": 2,
        "nested": [0.5],
    }
    assert json.loads(json.dumps(converted)) == converted
