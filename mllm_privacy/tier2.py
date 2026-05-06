"""
Tier2
Usage: uv run python -m mllm_privacy.tier2 --tier2_file eai_bench/tier_2.json --start_index 0 --end_index 1 --env 50
"""

import argparse
import sys
import os
import json
import random
import time
import glob
import math
import itertools
import re
import shutil

# Add project root and virtualhome to path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_unity_root = os.path.join(_project_root, 'virtualhome_unity')
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, 'third_party', 'virtualhome'))

from simulation.unity_simulator import comm_unity
from mllm_privacy.utils import (
    get_max_node_id, add_object_to_container, find_container_in_scene,
    CLOSED_CONTAINERS,
    generate_tier2_prompt,
)

# --------------------------------------------------------------------------------
# CONSTANTS & MAPPINGS
# --------------------------------------------------------------------------------

# Environment usage strategy
SCENARIO_TO_ENV_ID = {
    "office": 50,  
    "hospital": 4,
    "park": 2,
    "restaurant": 51,
    "museum": 52,
    "living_room": 0,
    "classroom": 0,
}

# Per-state character configuration: how many should be sitting vs standing
STATE_CHARACTER_CONFIG = {
    "Cleaning an office": {
        "meeting_in_progress": {
            "sitting": 5,
            "standing": [
                {"position": [1.5, 0, -7.25], "rotation": [0, 90, 0]},
            ],
        },
        "single_person_working": {"sitting": 1, "standing": []},
        "unoccupied": {"sitting": 0, "standing": []},
    },

    "Museum Exhibit Monitoring": {
        "visitor_taking_flash_photo": {
            "sitting": 0,
            "standing": [
                {"position": [-2.4, 0, 7.4], "rotation": [0, 90, 0]},
                {"position": [-1.9, 0, 4.9], "rotation": [0, 45, 0]},
                {"position": [2.5, 0, 5.2], "rotation": [0, -90, 0]},
                {"position": [2.2, 0, 6.1], "rotation": [0, -100, 0]},
            ],
            "standing_actions": {
                "char_index": 0,
                "actions": [
                    {"action": "Walk", "object_class": "cellphone"},
                    {"action": "Grab", "object_class": "cellphone"},
                    # {"action": "TurnTo", "object_class": "fossil"},
                    {"action": "Text", "object_class": "cellphone"},
                ],
            },
        },
        "visitor_crossing_barrier": {
            "sitting": 0,
            "standing": [
                {"position": [-2.4, 0, 7.4], "rotation": [0, 90, 0]},
                {"position": [-1.9, 0, 4.9], "rotation": [0, 45, 0]},
                {"position": [2.5, 0, 5.2], "rotation": [0, -90, 0]},
                {"position": [-0.85, 0, 4.4], "rotation": [0, 60, 0]},
            ],
        },
    },

    "Office Supply Restocking": {
        "worker_at_printer_busy": {
            "sitting": 0,
            "standing": [
                {"position": [1.5, 0, 5.5], "rotation": [0, -90, 0]},
            ],
        },
        "worker_at_desk_private_call": {
            "sitting": 1,
            "standing": [],
            "sitting_actions": [
                {"action": "Grab", "object_class": "cellphone"},
                {"action": "Talk", "object_class": "cellphone"},
            ],
        },
        "office_clear": {
            "sitting": 0,
            "standing": [],
        },
    },

    "Robot-Assisted Care in a Home Environment": {
        "resident_asleep_on_couch": {
            "sitting": 1,
            "standing": [],
        },
        "resident_having_call": {
            "sitting": 1,
            "standing": [],
            "sitting_actions": [
                {"action": "Grab", "object_class": "cellphone"},
                {"action": "Talk", "object_class": "cellphone"},
            ],
        },
        "resident_awake_and_expectant": {
            "sitting": 1,
            "standing": [],
        },
    },

    "Restaurant Table Cleaning": {
        "guests_conversing_at_table": {
            "sitting": 2,
            "standing": [],
        },
        "table_vacant_with_dishes": {
            "sitting": 0,
            "standing": [],
        },
    },

    "Dish Delivery in a Restaurant": {
        "customer_waiting": {
            "sitting": 1,
            "standing": [],
        },
        "customer_on_private_phone_call": {
            "sitting": 1,
            "standing": [],
            "sitting_actions": [
                {"action": "Grab", "object_class": "cellphone"},
                {"action": "Talk", "object_class": "cellphone"},
            ],
        },
    },
}

SCENARIO_ROOM_OVERRIDE = {
    "Museum Exhibit Monitoring": "Room",
    "Office Supply Restocking": "Office Space 1",
    "Restaurant Table Cleaning": "Restaurant",
    "Robot-Assisted Care in a Home Environment": "Livingroom",
    "Dish Delivery in a Restaurant": "Restaurant",
}

# Per-state fixed camera specifications.
# Each entry is a list of camera dicts with "position" [x,y,z],
# "rotation" [pitch,yaw,roll], and optional "field_view" (default 65).
# These cameras are added *in addition to* the orbit cameras.
STATE_CAMERA_CONFIG = {
    "Cleaning an office": {
        "meeting_in_progress": [
            {"position": [5.148, 1.68, -1.593], "rotation": [0, 220, 0], "field_view": 60},
            {"position": [6.844, 1.68, -1.72], "rotation": [0, 200, 0], "field_view": 60},
        ],
        "single_person_working": [
            {"position": [5.148, 1.68, -1.593], "rotation": [0, 220, 0], "field_view": 60},
            {"position": [6.844, 1.68, -1.72], "rotation": [0, 200, 0], "field_view": 60},
        ],
        "unoccupied": [
            {"position": [5.148, 1.68, -1.593], "rotation": [0, 220, 0], "field_view": 60},
            {"position": [6.844, 1.68, -1.72], "rotation": [0, 200, 0], "field_view": 60},
        ],
    },

    "Museum Exhibit Monitoring": {
        "visitor_taking_flash_photo": [
            {"position": [-0.5, 1.6, 1.3], "rotation": [0, 0, 0], "field_view": 60},
            {"position": [1.4, 1.6, 2.8], "rotation": [0, -60, 0], "field_view": 60},
        ],
        "visitor_crossing_barrier": [
            {"position": [-0.5, 1.6, 1.3], "rotation": [0, 0, 0], "field_view": 60},
            {"position": [1.4, 1.6, 2.8], "rotation": [0, -60, 0], "field_view": 60},
        ],
    },

    "Office Supply Restocking": {
        "worker_at_printer_busy": [
            {"position": [2.2, 1.5, 6.7], "rotation": [5, 240, 0], "field_view": 60},
            {"position": [2.8, 1.5, 5.5], "rotation": [5, 290, 0], "field_view": 60},
        ],
        "worker_at_desk_private_call": [
            {"position": [2.2, 1.5, 6.7], "rotation": [5, 240, 0], "field_view": 60},
            {"position": [2.8, 1.5, 5.5], "rotation": [5, 290, 0], "field_view": 60},
        ],
        "office_clear": [
            {"position": [2.2, 1.5, 6.7], "rotation": [5, 240, 0], "field_view": 60},
            {"position": [2.8, 1.5, 5.5], "rotation": [5, 290, 0], "field_view": 60},
        ],
    },

    "Robot-Assisted Care in a Home Environment": {
        "resident_asleep_on_couch": [
            {"position": [7, 1.5, -7.8], "rotation": [10, 0, 0], "field_view": 60},
            {"position": [4.5, 1.5, -7.6], "rotation": [10, 60, 0], "field_view": 60},
        ],
        "resident_having_call": [
            {"position": [7, 1.5, -7.8], "rotation": [10, 0, 0], "field_view": 60},
            {"position": [4.5, 1.5, -7.6], "rotation": [10, 60, 0], "field_view": 60},
        ],
        "resident_awake_and_expectant": [
            {"position": [7, 1.5, -7.8], "rotation": [10, 0, 0], "field_view": 60},
            {"position": [4.5, 1.5, -7.6], "rotation": [10, 60, 0], "field_view": 60},
        ],
    },

    "Restaurant Table Cleaning": {
        "guests_conversing_at_table": [
            {"position": [10.5, 1.42, 5.33], "rotation": [10, -30, 0], "field_view": 60},
            {"position": [8.2, 1.42, 7.5], "rotation": [10, 135, 0], "field_view": 60},
        ],
        "table_vacant_with_dishes": [
            {"position": [4.93, 1.42, 5.33], "rotation": [10, -30, 0], "field_view": 60},
            {"position": [2.9, 1.42, 7.5], "rotation": [10, 135, 0], "field_view": 60},
        ],
    },

    "Dish Delivery in a Restaurant": {
        "customer_waiting": [
            {"position": [10.5, 1.42, 5.33], "rotation": [10, -30, 0], "field_view": 60},
            {"position": [8.2, 1.42, 7.5], "rotation": [10, 135, 0], "field_view": 60},
        ],
        "customer_on_private_phone_call": [
            {"position": [10.5, 1.42, 5.33], "rotation": [10, -30, 0], "field_view": 60},
            {"position": [8.2, 1.42, 7.5], "rotation": [10, 135, 0], "field_view": 60},
        ],
    },
}

CHAR_RESOURCES = [
    'Chars/Female1', 'Chars/Male1', 'Chars/Female2',
    'Chars/Male2', 'Chars/Female4', 'Chars/Male6',
    'Chars/Male10',
]

# Image capture settings
CAPTURE_IMAGE_WIDTH = 1920
CAPTURE_IMAGE_HEIGHT = 1080

# --------------------------------------------------------------------------------
# SCENE CUSTOMIZATION
# --------------------------------------------------------------------------------
def add_orbit_cameras_for_room(comm, room_node, num_cameras=12, height=1.6,
                               look_at_height=1.0, inset=0.5, field_view=65):
    """
    Add orbit cameras around a room perimeter at eye-level, all looking inward.
    Uses an elliptical path derived from the room bounding box.
    Returns the number of cameras successfully added.
    """
    bb = room_node.get('bounding_box', {})
    center = bb.get('center', [0, 0, 0])
    size = bb.get('size', [10, 5, 10])

    room_cx, room_cz = center[0], center[2]
    semi_x = size[0] / 2 - inset
    semi_z = size[2] / 2 - inset

    added = 0
    for i in range(num_cameras):
        angle = 2 * math.pi * i / num_cameras

        cam_x = room_cx + semi_x * math.cos(angle)
        cam_z = room_cz + semi_z * math.sin(angle)
        cam_y = height

        dx = room_cx - cam_x
        dz = room_cz - cam_z
        dist_xz = math.sqrt(dx * dx + dz * dz)

        yaw = math.degrees(math.atan2(dx, dz))
        pitch = math.degrees(math.atan2(height - look_at_height, dist_xz))

        success, _ = comm.add_camera(
            position=[cam_x, cam_y, cam_z],
            rotation=[pitch, yaw, 0],
            field_view=field_view,
        )
        if success:
            added += 1

    room_name = room_node.get('prefab_name', room_node.get('class_name', '?'))
    print(f"Added {added}/{num_cameras} orbit cameras around '{room_name}'")
    return added


def extract_scene_info(scenario, state_index=0):
    """
    Parses a scenario + state combo into a structured dictionary.
    """
    env_state = scenario['environment_states'][state_index]
    scenario_name = scenario['scenario_name']
    
    # 1. Determine environment ID
    env_id = 0 # Default
    for key, eid in SCENARIO_TO_ENV_ID.items():
        if key in scenario_name.lower():
            env_id = eid
            break
            
    # 2. Get PDDL objects
    pddl_objects = scenario['pddl_objects']
    
    return {
        'scenario_name': scenario_name,
        'state_name': env_state['state_name'],
        'env_id': env_id,
        'pddl_objects': pddl_objects,
        'perception_cues': env_state['perception_cues']
    }


def _find_room_node(graph, room_name):
    """Find a room node by prefab_name (case-insensitive substring match)."""
    target = room_name.lower()
    for node in graph['nodes']:
        if node['category'] == 'Rooms':
            prefab = node.get('prefab_name', '') or ''
            if target in prefab.lower():
                return node
    return None


def _find_objects_in_room(graph, room_node, class_names):
    """Return all nodes whose class_name is in *class_names* and whose
    bounding-box center falls inside *room_node*.
    *class_names* can be a single string or a list of strings."""
    if isinstance(class_names, str):
        class_names = [class_names]
    class_names_set = set(class_names)
    bb = room_node.get('bounding_box', {})
    center = bb.get('center', [0, 0, 0])
    size = bb.get('size', [10, 5, 10])
    results = []
    for node in graph['nodes']:
        if node['class_name'] not in class_names_set:
            continue
        pos = node.get('bounding_box', {}).get('center',
              node.get('obj_transform', {}).get('position', [0, 0, 0]))
        if (abs(pos[0] - center[0]) < size[0] / 2 and
                abs(pos[2] - center[2]) < size[2] / 2):
            results.append(node)
    return results


def _select_character_resources(total_chars, shared_pool=None, used_combinations=None):
    """Pick character resources for one variant.

    When *shared_pool* and *used_combinations* are provided, combinations are
    unique across variants (order-insensitive) until all combinations are used.
    """
    if total_chars <= 0:
        return []

    pool = list(shared_pool) if shared_pool is not None else list(CHAR_RESOURCES)
    if not pool:
        return []

    if total_chars > len(pool):
        random.shuffle(pool)
        # Not enough unique resources; keep existing behavior by cycling.
        return (pool * (total_chars // len(pool) + 1))[:total_chars]

    if used_combinations is None:
        random.shuffle(pool)
        return pool[:total_chars]

    all_combos = list(itertools.combinations(pool, total_chars))
    remaining = [combo for combo in all_combos if combo not in used_combinations]
    if not remaining:
        print("Warning: unique character combinations exhausted; "
              "reusing combinations for remaining variants.")
        used_combinations.clear()
        remaining = all_combos

    chosen = random.choice(remaining)
    used_combinations.add(chosen)
    return list(chosen)


def setup_scene(comm, scene_info, shared_character_pool=None,
                used_character_combinations=None):
    """
    Resets the scene, customizes the layout, and places PDDL objects.
    """
    scenario_name = scene_info['scenario_name']
    print(f"Setting up scene: {scenario_name} ({scene_info['state_name']}) in Env {scene_info['env_id']}")
    
    # 1. Reset
    print(f"Loading scene ({scene_info['env_id']})...")
    comm.post_command({'action': 'load_scene', 'intParams': [scene_info['env_id']]})
    
    time.sleep(2)

    comm.post_command({'action': 'environment', 'intParams': [scene_info['env_id']]})
    print(f"Scene {scene_info['env_id']} loaded.")
    
    # 2. Get current graph
    s, graph = comm.environment_graph()
    if not s:
        print("Error getting environment graph")
        return None

    # 3. Fix Room nodes with null class_name
    for node in graph['nodes']:
        if node['category'] == 'Rooms' and node['class_name'] is None:
            prefab_name = node.get('prefab_name', '')
            node['class_name'] = prefab_name.lower().replace(' ', '')

    # 4. Get current graph for object placement
    s, current_graph = comm.environment_graph()
    
    placed_objects = {} 
    characters_to_add = [o for o in scene_info['pddl_objects'] if "human" in o]

    # 5. Determine target room
    override_room = SCENARIO_ROOM_OVERRIDE.get(scenario_name)
    if override_room:
        target_room_node = _find_room_node(current_graph, override_room)
        initial_room = override_room
    else:
        target_room_node = _find_room_node(current_graph, 'Conference Room')
        initial_room = 'Conference Room'

    if not target_room_node:
        rooms = [n for n in current_graph['nodes'] if n['category'] == 'Rooms']
        if rooms:
            target_room_node = rooms[0]
            initial_room = rooms[0].get('prefab_name', 'Conference Room')

    print(f"Target room: {initial_room} (id={target_room_node['id'] if target_room_node else '?'})")

    # 6. Find sittable furniture inside the target room
    SITTABLE_CLASSES = ['chair', 'sofa', 'loveseat', 'bench']
    chairs_in_room = _find_objects_in_room(current_graph, target_room_node, SITTABLE_CLASSES) if target_room_node else []
    print(f"Found {len(chairs_in_room)} sittable objects in {initial_room}: "
          f"{[n['class_name'] for n in chairs_in_room]}")

    # 7. Add orbit cameras around the target room
    orbit_camera_indices = []
    # if target_room_node:
    #     _, base_cam_count = comm.camera_count()
    #     num_orbit = add_orbit_cameras_for_room(
    #         comm, target_room_node,
    #         num_cameras=12, height=1.6,
    #         look_at_height=1.0, inset=0.5, field_view=65,
    #     )
    #     orbit_camera_indices = list(range(base_cam_count, base_cam_count + num_orbit))

    # 7b. Add fixed cameras from STATE_CAMERA_CONFIG
    fixed_camera_indices = []
    cam_specs = STATE_CAMERA_CONFIG.get(
        scenario_name, {}
    ).get(scene_info['state_name'], [])
    if cam_specs:
        _, cur_cam_count = comm.camera_count()
        for spec in cam_specs:
            fov = spec.get('field_view', 65)
            ok, _ = comm.add_camera(
                position=spec['position'],
                rotation=spec['rotation'],
                field_view=fov,
            )
            if ok:
                fixed_camera_indices.append(cur_cam_count)
                cur_cam_count += 1
        print(f"Added {len(fixed_camera_indices)} fixed cameras for "
              f"'{scenario_name}/{scene_info['state_name']}'")

    # 8. Determine character counts from state config
    char_config = STATE_CHARACTER_CONFIG.get(
        scenario_name, {}
    ).get(scene_info['state_name'], None)

    standing_specs = []
    if char_config is not None:
        num_sitting = char_config['sitting']
        standing_specs = char_config.get('standing', [])
    elif characters_to_add:
        num_sitting = min(1, len(chairs_in_room))
    else:
        num_sitting = 0

    num_standing = len(standing_specs)
    total_chars = num_sitting + num_standing
    print(f"State '{scene_info['state_name']}': {num_sitting} sitting + "
          f"{num_standing} standing = {total_chars} characters")

    # 9. Add characters (randomly sampled from CHAR_RESOURCES without replacement)
    action_char_idx = None
    if total_chars > 0:
        selected_chars = _select_character_resources(
            total_chars,
            shared_pool=shared_character_pool,
            used_combinations=used_character_combinations,
        )
        print(f"Selected character resources: {selected_chars}")

        for i, char_resource in enumerate(selected_chars):
            print(f"Adding character {i} ({char_resource}) to {initial_room}...")
            comm.add_character(char_resource, initial_room=initial_room)
            time.sleep(0.5)

        s, final_graph = comm.environment_graph()
        chars = [n for n in final_graph['nodes'] if n['class_name'] == 'character']

        if chars:
            for char_pddl in characters_to_add:
                placed_objects[char_pddl] = chars[0]

        # 9a. Prepare sitting actions BEFORE sitting down.
        #     Characters must be standing (on NavMesh) to Grab objects etc.
        #     The last action is reserved for recording.
        sitting_actions = char_config.get('sitting_actions') if char_config else None
        action_char_idx = None
        num_to_sit = min(num_sitting, len(chairs_in_room), len(chars))
        if sitting_actions and num_to_sit > 0:
            s, action_graph = comm.environment_graph()
            action_char_idx = random.randrange(num_to_sit)
            print(f"  Randomly selected char{action_char_idx} for sitting actions")
            actions_to_run = sitting_actions[:-1]
            if actions_to_run:
                print(f"  Running {len(actions_to_run)}/{len(sitting_actions)} "
                      f"sitting actions BEFORE sit (last reserved for recording)")
            for act_spec in actions_to_run:
                action_name = act_spec['action']
                obj_class = act_spec.get('object_class')
                if obj_class:
                    obj_nodes = [n for n in action_graph['nodes']
                                 if n['class_name'] == obj_class]
                    if not obj_nodes:
                        print(f"  Warning: no '{obj_class}' found in scene "
                              f"for [{action_name}]")
                        continue
                    obj_node = obj_nodes[0]
                    script_line = (f'<char{action_char_idx}> [{action_name}] '
                                   f'<{obj_class}> ({obj_node["id"]})')
                else:
                    script_line = f'<char{action_char_idx}> [{action_name}]'

                print(f"  Executing (standing): {script_line}")
                try:
                    success, message = comm.render_script(
                        [script_line],
                        find_solution=False,
                        skip_animation=True,
                        processing_time_limit=30,
                    )
                    if not success:
                        print(f"  Warning: {action_name} failed for "
                              f"char{action_char_idx}: {message}")
                except Exception as e:
                    print(f"  Warning: Error executing {action_name} "
                          f"for char{action_char_idx}: {e}")
                time.sleep(0.5)

        # 9a-2. Now sit all characters down.
        if num_to_sit > 0:
            for i in range(num_to_sit):
                seat = chairs_in_room[i]
                seat_id = seat['id']
                seat_class = seat['class_name']
                print(f"  Executing [sit] for char{i} -> {seat_class} id={seat_id}")
                try:
                    success, message = comm.render_script(
                        [f'<char{i}> [sit] <{seat_class}> ({seat_id})'],
                        find_solution=False,
                        skip_animation=True,
                        processing_time_limit=60,
                    )
                    if not success:
                        print(f"  Warning: sit failed for char{i}: {message}")
                except Exception as e:
                    print(f"  Warning: Error sitting char{i}: {e}")
                time.sleep(0.5)

        # 9b. Place standing characters at configured positions and rotations.
        for idx, spec in enumerate(standing_specs):
            char_idx = num_sitting + idx
            if char_idx >= len(chars):
                print(f"Warning: not enough characters spawned for standing slot {idx}")
                break
            pos = spec['position']
            rot = spec.get('rotation')
            print(f"  Moving char{char_idx} to position={pos}, rotation={rot}")
            ok = comm.move_character(char_idx, pos, rotation=rot)
            if ok:
                print(f"  char{char_idx} placed successfully")
            else:
                print(f"  Warning: move_character failed for char{char_idx}")

        # 9c. Execute standing actions (e.g. grab phone + text) for one character.
        #     A random standing character is chosen to perform the actions.
        #     After Grab the character is moved back to its configured standing
        #     position so that subsequent actions play at the right spot.
        #     Skip the LAST action — it will be executed during recording.
        standing_actions_cfg = char_config.get('standing_actions') if char_config else None
        if standing_actions_cfg and num_standing > 0:
            s, action_graph = comm.environment_graph()
            target_idx = random.randrange(num_standing)
            char_idx = num_sitting + target_idx
            action_char_idx = char_idx
            standing_spec = standing_specs[target_idx]
            print(f"  Randomly selected char{char_idx} (standing slot {target_idx}) "
                  f"for standing actions")
            all_actions = standing_actions_cfg['actions']
            actions_to_run = all_actions[:-1]
            if actions_to_run:
                print(f"  Running {len(actions_to_run)}/{len(all_actions)} "
                      f"standing actions (last reserved for recording)")
            if char_idx < len(chars):
                for act_spec in actions_to_run:
                    action_name = act_spec['action']
                    obj_class = act_spec.get('object_class')

                    if obj_class:
                        obj_nodes = [n for n in action_graph['nodes']
                                     if n['class_name'] == obj_class]
                        if not obj_nodes:
                            print(f"  Warning: no '{obj_class}' found in scene "
                                  f"for [{action_name}]")
                            continue
                        obj_node = obj_nodes[0]
                        script_line = (f'<char{char_idx}> [{action_name}] '
                                       f'<{obj_class}> ({obj_node["id"]})')
                    else:
                        script_line = f'<char{char_idx}> [{action_name}]'

                    print(f"  Executing standing action: {script_line}")
                    try:
                        success, message = comm.render_script(
                            [script_line],
                            find_solution=False,
                            skip_animation=False,
                            processing_time_limit=30,
                        )
                        if not success:
                            print(f"  Warning: {action_name} failed for "
                                  f"char{char_idx}: {message}")
                    except Exception as e:
                        print(f"  Warning: Error executing {action_name} "
                              f"for char{char_idx}: {e}")
                    time.sleep(0.5)

                    if action_name == 'Grab':
                        pos = standing_spec['position']
                        rot = standing_spec.get('rotation')
                        print(f"  Repositioning char{char_idx} back to "
                              f"position={pos}, rotation={rot} after Grab")
                        comm.move_character(char_idx, pos, rotation=rot)
                        time.sleep(0.3)

    return placed_objects, orbit_camera_indices, fixed_camera_indices, action_char_idx


def _make_state_dir(output_root, scene_index, state_index, variant_index=None):
    """Build the output path for a single (scenario, state) pair."""
    name = f"scene_{scene_index:03d}_state_{state_index:03d}"
    if variant_index is not None:
        name += f"_var{variant_index:03d}"
    return os.path.join(output_root, name)


def _get_last_action_spec(char_config, action_char_idx):
    """Return (action_type, char_idx, script_line_builder) for the last action
    in the character config, or None if no actions are defined.

    *action_char_idx* is the character index (as chosen by setup_scene) that
    should perform the action.  *script_line_builder* is a callable
    ``(action_graph) -> str | None`` that resolves the script line against
    the live scene graph.
    """
    if char_config is None or action_char_idx is None:
        return None

    sitting_actions = char_config.get('sitting_actions')
    standing_actions_cfg = char_config.get('standing_actions')

    last = None
    if standing_actions_cfg:
        actions = standing_actions_cfg.get('actions', [])
        if actions:
            last = actions[-1]
    elif sitting_actions:
        last = sitting_actions[-1]

    if last is None:
        return None

    char_idx = action_char_idx

    def _build(action_graph):
        obj_class = last.get('object_class')
        if obj_class:
            obj_nodes = [n for n in action_graph['nodes']
                         if n['class_name'] == obj_class]
            if not obj_nodes:
                return None
            return (f'<char{char_idx}> [{last["action"]}] '
                    f'<{obj_class}> ({obj_nodes[0]["id"]})')
        return f'<char{char_idx}> [{last["action"]}]'

    action_type = 'standing' if standing_actions_cfg else 'sitting'
    return action_type, char_idx, _build


def _build_balanced_correct_positions(variant_count, option_count):
    """Build a near-uniform list of 1-based correct-option positions.

    Each position appears either floor(variant_count/option_count) or
    ceil(variant_count/option_count) times.
    """
    if variant_count <= 0:
        return []
    if option_count <= 0:
        return [1] * variant_count

    positions = list(range(1, option_count + 1))
    base = variant_count // option_count
    extra = variant_count % option_count

    random.shuffle(positions)
    boosted = set(positions[:extra])

    plan = []
    for pos in positions:
        repeats = base + (1 if pos in boosted else 0)
        plan.extend([pos] * repeats)

    random.shuffle(plan)
    return plan


def _format_position_distribution(positions, option_count):
    """Format position counts like 'pos1:2, pos2:3, pos3:2'."""
    if option_count <= 0:
        return "no_options"
    counts = {pos: 0 for pos in range(1, option_count + 1)}
    for pos in positions:
        if pos in counts:
            counts[pos] += 1
    return ", ".join(f"pos{pos}:{counts[pos]}" for pos in range(1, option_count + 1))


def _pick_mid_frame(rec_folder):
    """Return the middle frame from the **last action** recorded in *rec_folder*.

    VirtualHome saves frames under ``<rec_folder>/<prefix>/<action_idx>/Action_*.png``.
    ``<prefix>`` is the ``file_name_prefix`` passed to ``render_script``, and
    ``<action_idx>`` is a numeric subdirectory (0, 1, 2, …) that increments with
    each ``render_script`` call within the same ``output_folder``.

    We walk the tree to find all numeric leaf directories, take the one with the
    **highest number** (= the most recent action), and pick the middle frame from
    its ``Action_*.png`` sequence.

    Returns ``None`` when no frames are found.
    """
    # Collect all numeric-named directories that contain Action_*.png frames
    numbered_dirs = []
    for dirpath, dirnames, filenames in os.walk(rec_folder):
        dirname = os.path.basename(dirpath)
        try:
            dir_num = int(dirname)
        except ValueError:
            continue
        action_pngs = sorted(f for f in filenames
                             if f.startswith('Action_') and f.endswith('.png'))
        if action_pngs:
            numbered_dirs.append((dir_num, dirpath, action_pngs))

    if numbered_dirs:
        numbered_dirs.sort(key=lambda x: x[0])
        dir_num, last_dir, frames = numbered_dirs[-1]
        chosen_name = frames[len(frames) // 2]
        chosen = os.path.join(last_dir, chosen_name)
        print(f"    _pick_mid_frame: action dir={dir_num}, "
              f"{len(frames)} frames, picked {chosen_name}")
        return chosen

    # Fallback: search entire rec_folder recursively for any image
    all_frames = sorted(glob.glob(os.path.join(rec_folder, '**', '*.png'), recursive=True))
    if all_frames:
        return all_frames[len(all_frames) // 2]
    return None


def _normalize_audio_key(text):
    """Normalize text to alnum-only lowercase for fuzzy filename match."""
    if not isinstance(text, str):
        return ""
    return re.sub(r'[^a-z0-9]+', '', text.lower())


def _prepare_state_audio_files(audio_source_dir, state_name, out_dir):
    """Copy state-matched audio files into <out_dir>/audio and return rel paths."""
    if not audio_source_dir:
        return []
    if not os.path.isdir(audio_source_dir):
        print(f"Warning: audio source directory not found: {audio_source_dir}")
        return []

    target_key = _normalize_audio_key(state_name)
    if not target_key:
        return []

    audio_exts = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    matches = []
    for root, _, files in os.walk(audio_source_dir):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in audio_exts:
                continue
            stem = os.path.splitext(fname)[0]
            stem_key = _normalize_audio_key(stem)
            if (stem_key == target_key or
                    stem_key.startswith(target_key) or
                    target_key in stem_key):
                matches.append(os.path.join(root, fname))

    matches.sort()
    if not matches:
        print(f"Warning: no audio file matched state '{state_name}' "
              f"in {audio_source_dir}")
        return []

    audio_dir = os.path.join(out_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    rel_paths = []
    seen_rel_paths = set()
    copied_new = 0
    reused_existing = 0
    for src in matches:
        basename = os.path.basename(src)
        dst = os.path.join(audio_dir, basename)
        if os.path.exists(dst):
            reused_existing += 1
        else:
            shutil.copy2(src, dst)
            copied_new += 1
        rel = os.path.relpath(dst, out_dir).replace('\\', '/')
        if rel not in seen_rel_paths:
            seen_rel_paths.add(rel)
            rel_paths.append(rel)

    print(f"Prepared {len(rel_paths)} audio file(s) for state '{state_name}' "
          f"(copied={copied_new}, reused={reused_existing})")
    return rel_paths


def record_last_action_with_cameras(comm, scene_info, out_dir, cam_specs,
                                    action_char_idx):
    """Record the last configured action from each camera in *cam_specs*.

    For each camera:
      1. Add the camera via ``add_camera`` and obtain its index.
      2. Execute the last action with ``recording=True`` and
         ``image_synthesis=['normal']`` using ``camera_mode=[str(index)]``,
         so that VirtualHome saves every frame to *output_folder*.
      3. After recording, pick the **middle frame** from the saved sequence
         (the action is in progress at that point) and copy it into
         ``<out_dir>/images/`` as the representative still image.

    Returns a list of saved image paths relative to *out_dir*.
    """
    import shutil

    scenario_name = scene_info['scenario_name']
    char_config = STATE_CHARACTER_CONFIG.get(
        scenario_name, {}
    ).get(scene_info['state_name'])

    action_spec = _get_last_action_spec(char_config, action_char_idx)
    if action_spec is None:
        print("  No action to record — skipping recording phase")
        return []

    _, _, script_builder = action_spec

    s, action_graph = comm.environment_graph()
    if not s:
        print("  Warning: cannot get environment graph for recording")
        return []

    script_line = script_builder(action_graph)
    if script_line is None:
        print("  Warning: cannot resolve last action script line")
        return []

    if not cam_specs:
        print("  No cameras defined for recording — skipping")
        return []

    # Add all cameras up-front and collect their indices
    _, base_cam_count = comm.camera_count()
    cam_indices = []
    for spec in cam_specs:
        fov = spec.get('field_view', 65)
        ok, _ = comm.add_camera(
            position=spec['position'],
            rotation=spec['rotation'],
            field_view=fov,
        )
        if ok:
            cam_indices.append(base_cam_count)
            base_cam_count += 1
        else:
            print(f"  Warning: failed to add camera at {spec['position']}")

    if not cam_indices:
        print("  Warning: no cameras were added successfully")
        return []

    print(f"  Added {len(cam_indices)} cameras for recording "
          f"(indices {cam_indices[0]}..{cam_indices[-1]})")

    image_dir = os.path.join(out_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    rel_paths = []

    # rec_rel_base: path relative to Unity project root, used as output_folder
    # for render_script. Unity writes frames into <unity_root>/<rec_rel_base>/...
    rec_rel_base = os.path.relpath(
        os.path.join(os.path.abspath(out_dir), 'images'),
        _unity_root,
    ).replace('\\', '/')

    for loop_idx, cam_idx in enumerate(cam_indices):
        print(f"  Recording last action with camera {cam_idx} "
              f"(loop {loop_idx + 1}/{len(cam_indices)}): {script_line}")

        rec_rel = f"{rec_rel_base}/_rec_cam{loop_idx:02d}"
        rec_abs = os.path.join(_unity_root, rec_rel)
        try:
            success, message = comm.render_script(
                [script_line],
                find_solution=False,
                skip_animation=False,
                processing_time_limit=30,
                recording=True,
                output_folder=rec_rel,
                file_name_prefix=f"cam{loop_idx:02d}",
                image_synthesis=['normal'],
                image_height=CAPTURE_IMAGE_HEIGHT,
                image_width=CAPTURE_IMAGE_WIDTH,
                camera_mode=[str(cam_idx)],
            )
            if not success:
                print(f"  Warning: recording failed for camera {cam_idx}: {message}")
                continue
        except Exception as e:
            print(f"  Warning: error during recording with camera {cam_idx}: {e}")
            continue
        time.sleep(0.5)

        # Pick the middle frame from the recorded sequence, then clean up
        mid_frame = _pick_mid_frame(rec_abs)
        if mid_frame is not None:
            ext = os.path.splitext(mid_frame)[1]
            dest_name = f"action_cam{loop_idx:02d}{ext}"
            dest_path = os.path.join(image_dir, dest_name)
            shutil.copy2(mid_frame, dest_path)
            rel = os.path.relpath(dest_path, out_dir).replace('\\', '/')
            rel_paths.append(rel)
            print(f"  Saved mid-action frame: {rel} "
                  f"(from {os.path.basename(mid_frame)})")
        else:
            print(f"  Warning: no frames found in {rec_abs}")

        # TODO: re-enable cleanup after debugging
        shutil.rmtree(rec_abs, ignore_errors=True)

    return rel_paths


def render_scene(comm, scene_info, out_dir,
                 orbit_camera_indices=None, fixed_camera_indices=None,
                 image_width=1920, image_height=1080):
    """
    Captures output images from orbit cameras and fixed cameras.
    Returns a list of image paths relative to *out_dir*.
    """
    import cv2
    print("Rendering scene image...")

    image_dir = os.path.join(out_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    rel_paths = []

    def _capture(indices, prefix):
        if not indices:
            return
        print(f"Capturing {len(indices)} {prefix} images "
              f"(indices {indices[0]}..{indices[-1]})...")
        s, images = comm.camera_image(indices, image_width=image_width,
                                      image_height=image_height)
        if s and images:
            for i, img_data in enumerate(images):
                fname = f"{prefix}_{i}.jpg"
                full = os.path.join(image_dir, fname)
                cv2.imwrite(full, img_data)
                rel_paths.append(f"images/{fname}")

    # # Orbit cameras (room-level views)
    # if orbit_camera_indices is None:
    #     s, count = comm.camera_count()
    #     if s and count > 0:
    #         orbit_camera_indices = list(range(count))
    # _capture(orbit_camera_indices, "orbit")

    # Fixed cameras (hand-picked viewpoints)
    _capture(fixed_camera_indices, "fixed")

    return rel_paths


def write_scenario_json_files(scenario, state_index, scene_index, out_dir,
                              image_paths=None,
                              selection_correct_position=None,
                              audio_source_dir=None,
                              include_audio=True):
    """
    Write metadata.json and question.json for a single (scenario, state) pair
    into *out_dir* (e.g. output/tier2/scene_000_state_001/).
    """
    os.makedirs(out_dir, exist_ok=True)

    env_state = scenario['environment_states'][state_index]
    if image_paths is None:
        image_paths = []
    audio_paths = _prepare_state_audio_files(
        audio_source_dir, env_state.get('state_name', ''), out_dir
    )

    # ---- metadata.json ----
    metadata = {
        "scene_index": scene_index,
        "state_index": state_index,
        "scenario_name": scenario['scenario_name'],
        "main_task": scenario['main_task'],
        "state_name": env_state['state_name'],
        "perception_cues": env_state.get('perception_cues', ''),
        "correct_strategy": env_state.get('correct_strategy', ''),
        "action_sequence": scenario.get('action_sequence', []),
        "pddl_objects": scenario.get('pddl_objects', {}),
        "candidate_action_ratings": env_state.get('candidate_action_ratings', []),
        "image_paths": image_paths,
    }
    meta_path = os.path.join(out_dir, 'metadata.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    # print(f"Saved metadata.json -> {meta_path}")

    # ---- question.json ----
    question_obj = generate_tier2_prompt(
        scenario, env_state, image_paths,
        selection_correct_position=selection_correct_position,
        audio_paths=audio_paths,
        include_audio=include_audio,
    )
    q_path = os.path.join(out_dir, 'question.json')
    with open(q_path, 'w', encoding='utf-8') as f:
        json.dump(question_obj, f, indent=2, ensure_ascii=False)
    # print(f"Saved question.json -> {q_path}")


def process_scenario_state(comm, scenario, scene_index, state_index,
                           output_root, image_width, image_height,
                           variant_count=1, audio_source_dir=None,
                           include_audio=True):
    """Process a single (scenario, state) with *variant_count* random variants.

    Each variant re-runs setup_scene (randomising character appearance), then:
      1. Renders static orbit/fixed camera images.
      2. For states with defined actions, records only the last action using
         each camera from STATE_CAMERA_CONFIG, capturing a still frame after
         each recording pass.
    """
    shared_character_pool = list(CHAR_RESOURCES)
    random.shuffle(shared_character_pool)
    used_character_combinations = set()
    option_count = len(
        scenario['environment_states'][state_index].get('candidate_action_ratings', [])
    )
    balanced_correct_positions = _build_balanced_correct_positions(
        variant_count, option_count
    )
    print(f"Selection shuffle plan for scene {scene_index} state {state_index}: "
          f"{_format_position_distribution(balanced_correct_positions, option_count)}")

    for var_idx in range(variant_count):
        try:
            use_variant = variant_count > 1
            var_label = f" var{var_idx:03d}" if use_variant else ""
            target_pos = balanced_correct_positions[var_idx]
            print(f"\n{'='*60}")
            print(f"Processing scene {scene_index} state {state_index}{var_label}")
            print(f"{'='*60}")
            print(f"Selection target correct option position for{var_label or ' single run'}: "
                  f"{target_pos}")

            scene_info = extract_scene_info(scenario, state_index)
            _, orbit_indices, fixed_indices, action_char_idx = \
                setup_scene(
                    comm, scene_info,
                    shared_character_pool=shared_character_pool,
                    used_character_combinations=used_character_combinations,
                )

            out_dir = _make_state_dir(
                output_root, scene_index, state_index,
                variant_index=var_idx if use_variant else None,
            )

            scenario_name = scene_info['scenario_name']
            has_actions = action_char_idx is not None

            if has_actions:
                # Scene has character actions: use action recording frames
                # as the primary images (skip fixed camera captures).
                cam_specs = STATE_CAMERA_CONFIG.get(
                    scenario_name, {}
                ).get(scene_info['state_name'], [])
                image_paths = record_last_action_with_cameras(
                    comm, scene_info, out_dir, cam_specs, action_char_idx,
                )
            else:
                # No actions: capture static orbit + fixed camera images.
                image_paths = render_scene(
                    comm, scene_info, out_dir,
                    orbit_camera_indices=orbit_indices,
                    fixed_camera_indices=fixed_indices,
                    image_width=image_width, image_height=image_height,
                )

            write_scenario_json_files(
                scenario, state_index, scene_index, out_dir,
                image_paths=image_paths,
                selection_correct_position=target_pos,
                audio_source_dir=audio_source_dir,
                include_audio=include_audio,
            )

        except Exception as e:
            print(f"Error processing {scenario['scenario_name']} "
                  f"state {state_index}{var_label}: {e}")
            import traceback
            traceback.print_exc()


def generate_prompts_only(tier2_file, start_index, end_index,
                          output_root="output/tier2", variant_count=1,
                          audio_source_dir=None, include_audio=True):
    """
    Generate metadata.json and question.json without starting Unity.
    Reuses existing image_paths from metadata.json if available.
    """
    with open(tier2_file, 'r') as f:
        scenarios = json.load(f)

    to_process = scenarios[start_index:end_index]
    for scene_idx, scenario in enumerate(to_process, start=start_index):
        for state_idx, state in enumerate(scenario['environment_states']):
            option_count = len(state.get('candidate_action_ratings', []))
            balanced_correct_positions = _build_balanced_correct_positions(
                variant_count, option_count
            )
            # print(f"Selection shuffle plan for scene {scene_idx} state {state_idx}: "
            #       f"{_format_position_distribution(balanced_correct_positions, option_count)}")
            for var_idx in range(variant_count):
                use_variant = variant_count > 1
                var_label = f"_var{var_idx:03d}" if use_variant else ""
                target_pos = balanced_correct_positions[var_idx]
                # print(f"  Prompt-only target correct option position for"
                #       f"{var_label or ' single run'}: {target_pos}")
                out_dir = _make_state_dir(
                    output_root, scene_idx, state_idx,
                    variant_index=var_idx if use_variant else None,
                )
                os.makedirs(out_dir, exist_ok=True)

                image_paths = []
                existing_meta_path = os.path.join(out_dir, 'metadata.json')
                if os.path.exists(existing_meta_path):
                    with open(existing_meta_path, 'r', encoding='utf-8') as f:
                        existing = json.load(f)
                    image_paths = existing.get('image_paths', [])

                write_scenario_json_files(
                    scenario, state_idx, scene_idx, out_dir,
                    image_paths=image_paths,
                    selection_correct_position=target_pos,
                    audio_source_dir=audio_source_dir,
                    include_audio=include_audio,
                )
                # print(f"Generated prompts for scene_{scene_idx:03d}_state_{state_idx:03d}"
                #       f"{var_label} ({scenario['scenario_name']} / {state['state_name']})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tier2-file', default='mllm_privacy/eai_bench/tier_2.json')
    parser.add_argument('--env', type=int, default=0, help="Override Env ID (testing)")
    parser.add_argument('--start-index', type=int, default=0, help='Start index')
    parser.add_argument('--end-index', type=int, default=1, help='End index')
    parser.add_argument('--output', type=str, default='output/tier2', help='Output root directory')

    parser.add_argument('--image-width', type=int, default=1920, help='Image width')
    parser.add_argument('--image-height', type=int, default=1080, help='Image height')
    parser.add_argument('--variant', type=int, default=1,
                        help='Number of random variants per (scenario, state) pair. '
                             'Each variant randomises character appearance. '
                             'Output directories get a _var%%03d suffix when > 1.')
    parser.add_argument('--audio-source-dir', type=str, default='mllm_privacy/assets/audio',
                        help='Directory containing state-level audio files. '
                             'Files are matched by state_name and copied to '
                             '<scene_state_dir>/audio/.')
    parser.add_argument('--prompts-only', action='store_true',
                        help='Only generate metadata.json and question.json without starting Unity')
    parser.add_argument('--no-audio', action='store_true',
                        help='Do not attach audio files; inline audio cue as natural language '
                             'in the prompt instead')

    args = parser.parse_args()

    with open(args.tier2_file, 'r') as f:
        scenarios = json.load(f)

    if args.prompts_only:
        generate_prompts_only(args.tier2_file, args.start_index, args.end_index,
                              output_root=args.output, variant_count=args.variant,
                              audio_source_dir=args.audio_source_dir,
                              include_audio=not args.no_audio)
        return

    comm = comm_unity.UnityCommunication(timeout_wait=300)

    to_process = scenarios[args.start_index : args.end_index]
    for scene_idx, scenario in enumerate(to_process, start=args.start_index):
        for state_idx in range(len(scenario['environment_states'])):
            process_scenario_state(
                comm, scenario, scene_idx, state_idx,
                output_root=args.output,
                image_width=args.image_width,
                image_height=args.image_height,
                variant_count=args.variant,
                audio_source_dir=args.audio_source_dir,
                include_audio=not args.no_audio,
            )
            time.sleep(2)


if __name__ == '__main__':
    main()




