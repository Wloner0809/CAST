import random
import marshal
from contextlib import contextmanager
from typing import Dict, List, Tuple

import numpy as np

CoordDict = Dict[str, List[Tuple[int, int]]]


@contextmanager
def seed_context(seed: int | None):
    """Create thread-safe, isolated RNG instances for deterministic room generation.

    Yields a ``(py_rng, np_rng)`` tuple of *independent* random number generators
    seeded with *seed*.  These instances are **not** tied to any global state, so
    concurrent calls from different threads will never interfere with each other.

    Callers that previously used global ``random`` / ``np.random`` inside the
    context should switch to the yielded RNG objects instead.

    For backward compatibility the context manager still works when the caller
    ignores the yielded value (``with seed_context(seed):``), but in that case
    the global-state race condition is still present.  All internal callers in
    this module have been updated to use the yielded RNGs.
    """
    if seed is None:
        yield None, None
        return

    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    yield py_rng, np_rng


def derive_retry_seed(seed: int | None) -> int | None:
    """Derive a deterministic next seed for retry attempts.

    Uses a 32-bit LCG step so retry chains are stable across processes/runs.
    """
    if seed is None:
        return None
    return (1664525 * (int(seed) & 0xFFFFFFFF) + 1013904223) & 0xFFFFFFFF


def _to_tuple_list(array: np.ndarray, index_origin: int = 0) -> List[Tuple[int, int]]:
    if array.size == 0:
        return []
    return [(int(r) + index_origin, int(c) + index_origin) for r, c in array]


def collect_entity_coordinates(room_state: np.ndarray, room_fixed: np.ndarray, index_origin: int = 0) -> CoordDict:
    coords: CoordDict = {
        "targets": _to_tuple_list(np.argwhere(room_fixed == 2), index_origin),
        "boxes_on_target": _to_tuple_list(np.argwhere(room_state == 3), index_origin),
        "boxes": _to_tuple_list(np.argwhere(room_state == 4), index_origin),
    }

    player_positions = np.argwhere(room_state == 5)
    player_on_target = []
    player_regular = []
    for pos in player_positions:
        r, c = int(pos[0]), int(pos[1])
        if room_fixed[r, c] == 2:
            player_on_target.append((r + index_origin, c + index_origin))
        else:
            player_regular.append((r + index_origin, c + index_origin))

    coords["player"] = player_regular
    coords["player_on_target"] = player_on_target
    return coords


def format_coordinate_render(entity_coords: CoordDict, board_shape: Tuple[int, int], index_origin: int = 0) -> str:
    rows, cols = board_shape
    origin_str = "zero-indexed" if index_origin == 0 else f"origin at {index_origin}"
    lines = [f"Board size: {rows} rows x {cols} cols ({origin_str})."]
    ordered_labels = [
        ("targets", "Targets"),
        ("boxes", "Boxes"),
        ("boxes_on_target", "Boxes on target"),
        ("player", "Player"),
        ("player_on_target", "Player on target"),
    ]

    for key, label in ordered_labels:
        coords = entity_coords.get(key, [])
        if not coords:
            continue
        coord_str = ", ".join(f"({r}, {c})" for r, c in coords)
        lines.append(f"{label}: {coord_str}")

    return "\n".join(lines)


def add_random_player_movement(room_state, room_structure, move_probability=0.5, continue_probability=0.5, max_steps=3, py_rng=None):
    _rng = py_rng or random
    if _rng.random() > move_probability:
        return room_state

    player_pos = np.where(room_state == 5)
    player_pos = np.array([player_pos[0][0], player_pos[1][0]])

    previous_positions = [tuple(player_pos)]

    steps_taken = 0
    while steps_taken < max_steps:
        valid_moves = []
        for action in range(4):
            change = CHANGE_COORDINATES[action]
            next_pos = player_pos + change

            if room_state[next_pos[0], next_pos[1]] in [1, 2] and tuple(next_pos) not in previous_positions:
                valid_moves.append((action, next_pos))

        if not valid_moves:
            break

        _, next_pos = _rng.choice(valid_moves)

        room_state[player_pos[0], player_pos[1]] = room_structure[player_pos[0], player_pos[1]]
        room_state[next_pos[0], next_pos[1]] = 5

        player_pos = next_pos
        previous_positions.append(tuple(player_pos))

        steps_taken += 1

        if steps_taken >= max_steps or _rng.random() > continue_probability:
            break

    return room_state


# Adapted from gym-sokoban generation routines (via RAGEN).

def generate_room(dim=(13, 13), p_change_directions=0.35, num_steps=25, num_boxes=3, tries=4, second_player=False, search_depth=100, py_rng=None, np_rng=None):
    room_state = np.zeros(shape=dim)
    room_structure = np.zeros(shape=dim)

    for _ in range(tries):
        room = room_topology_generation(dim, p_change_directions, num_steps, py_rng=py_rng)
        room = place_boxes_and_player(room, num_boxes=num_boxes, second_player=second_player, np_rng=np_rng)

        room_structure = np.copy(room)
        room_structure[room_structure == 5] = 1

        room_state = room.copy()
        room_state[room_state == 2] = 4

        room_state, box_mapping, action_sequence = reverse_playing(room_state, room_structure, search_depth)
        room_state[room_state == 3] = 4

        if box_displacement_score(box_mapping) > 0:
            break

    if box_displacement_score(box_mapping) == 0:
        raise RuntimeWarning("Generated model with score == 0")

    if box_displacement_score(box_mapping) == 1:
        move_probability = 0.8
    else:
        move_probability = 0.5

    room_state = add_random_player_movement(
        room_state,
        room_structure,
        move_probability=move_probability,
        continue_probability=0.5,
        max_steps=3,
        py_rng=py_rng,
    )

    return room_structure, room_state, box_mapping, action_sequence


def room_topology_generation(dim=(10, 10), p_change_directions=0.35, num_steps=15, py_rng=None):
    _rng = py_rng or random
    dim_x, dim_y = dim

    masks = [
        [
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 0],
        ],
        [
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
        ],
        [
            [0, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
        ],
        [
            [0, 0, 0],
            [1, 1, 0],
            [1, 1, 0],
        ],
        [
            [0, 0, 0],
            [0, 1, 1],
            [0, 1, 0],
        ],
    ]

    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    direction = _rng.sample(directions, 1)[0]

    position = np.array([
        _rng.randint(1, dim_x - 1),
        _rng.randint(1, dim_y - 1),
    ])

    level = np.zeros(dim, dtype=int)

    for _ in range(num_steps):
        if _rng.random() < p_change_directions:
            direction = _rng.sample(directions, 1)[0]

        position = position + direction
        position[0] = max(min(position[0], dim_x - 2), 1)
        position[1] = max(min(position[1], dim_y - 2), 1)

        mask = _rng.sample(masks, 1)[0]
        mask_start = position - 1
        level[mask_start[0] : mask_start[0] + 3, mask_start[1] : mask_start[1] + 3] += mask

    level[level > 0] = 1
    level[:, [0, dim_y - 1]] = 0
    level[[0, dim_x - 1], :] = 0

    return level


def place_boxes_and_player(room, num_boxes, second_player, np_rng=None):
    _rng = np_rng or np.random
    possible_positions = np.where(room == 1)
    num_possible_positions = possible_positions[0].shape[0]
    num_players = 2 if second_player else 1

    if num_possible_positions <= num_boxes + num_players:
        raise RuntimeError(
            "Not enough free spots (#{} ) to place {} player and {} boxes.".format(
                num_possible_positions,
                num_players,
                num_boxes,
            )
        )

    ind = _rng.integers(num_possible_positions) if np_rng is not None else _rng.randint(num_possible_positions)
    player_position = possible_positions[0][ind], possible_positions[1][ind]
    room[player_position] = 5

    if second_player:
        ind = _rng.integers(num_possible_positions) if np_rng is not None else _rng.randint(num_possible_positions)
        player_position = possible_positions[0][ind], possible_positions[1][ind]
        room[player_position] = 5

    for _ in range(num_boxes):
        possible_positions = np.where(room == 1)
        num_possible_positions = possible_positions[0].shape[0]

        ind = _rng.integers(num_possible_positions) if np_rng is not None else _rng.randint(num_possible_positions)
        box_position = possible_positions[0][ind], possible_positions[1][ind]
        room[box_position] = 2

    return room


def reverse_playing(room_state, room_structure, search_depth=100):
    box_mapping = {}
    box_locations = np.where(room_structure == 2)
    num_boxes = len(box_locations[0])
    for idx in range(num_boxes):
        box = (box_locations[0][idx], box_locations[1][idx])
        box_mapping[box] = box

    explored_states = set()
    best = {
        "score": -1,
        "room": None,
        "box_mapping": box_mapping.copy(),
        "action_sequence": [],
    }

    def depth_first_search(room_state_inner, box_mapping_inner, box_swaps=0, last_pull=(-1, -1), ttl=300, action_sequence=None):
        if action_sequence is None:
            action_sequence = []

        ttl -= 1
        if ttl <= 0 or len(explored_states) >= 300000:
            return

        state_tohash = marshal.dumps(room_state_inner)
        if state_tohash in explored_states:
            return

        room_score = box_swaps * box_displacement_score(box_mapping_inner)
        if np.where(room_state_inner == 2)[0].shape[0] != num_boxes:
            room_score = 0

        if room_score > best["score"]:
            best["room"] = room_state_inner.copy()
            best["score"] = room_score
            best["box_mapping"] = box_mapping_inner.copy()
            best["action_sequence"] = action_sequence.copy()

        explored_states.add(state_tohash)

        for action in ACTION_LOOKUP.keys():
            if action >= 4:
                continue

            room_state_next = room_state_inner.copy()
            box_mapping_next = box_mapping_inner.copy()

            room_state_next, box_mapping_next, last_pull_next = reverse_move(
                room_state_next, room_structure, box_mapping_next, last_pull, action
            )

            box_swaps_next = box_swaps + (1 if last_pull_next != last_pull else 0)
            depth_first_search(
                room_state_next,
                box_mapping_next,
                box_swaps_next,
                last_pull_next,
                ttl,
                action_sequence + [action],
            )

    depth_first_search(room_state, box_mapping, ttl=search_depth, action_sequence=[])

    return best["room"], best["box_mapping"], best["action_sequence"]


def reverse_move(room_state, room_structure, box_mapping, last_pull, action):
    player_position = np.where(room_state == 5)
    player_position = np.array([player_position[0][0], player_position[1][0]])

    change = np.array(CHANGE_COORDINATES[action % 4])
    next_position = player_position + change

    if room_state[next_position[0], next_position[1]] in [1, 2]:
        room_state[player_position[0], player_position[1]] = room_structure[player_position[0], player_position[1]]
        room_state[next_position[0], next_position[1]] = 5

        if action < 4:
            possible_box_location = player_position - change

            if room_state[possible_box_location[0], possible_box_location[1]] in [3, 4]:
                room_state[player_position[0], player_position[1]] = 3
                room_state[possible_box_location[0], possible_box_location[1]] = room_structure[
                    possible_box_location[0], possible_box_location[1]
                ]

                for k in list(box_mapping.keys()):
                    if box_mapping[k] == (possible_box_location[0], possible_box_location[1]):
                        box_mapping[k] = (player_position[0], player_position[1])
                        last_pull = k

    return room_state, box_mapping, last_pull


def box_displacement_score(box_mapping):
    score = 0
    for box_target, box_location in box_mapping.items():
        box_location = np.array(box_location)
        box_target = np.array(box_target)
        dist = np.sum(np.abs(box_location - box_target))
        score += dist
    return score


ACTION_LOOKUP = {
    0: "push up",
    1: "push down",
    2: "push left",
    3: "push right",
    4: "move up",
    5: "move down",
    6: "move left",
    7: "move right",
}

CHANGE_COORDINATES = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
}
