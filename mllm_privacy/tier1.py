"""
Tier1

Usage:
    python -m mllm_privacy.tier1 --prompts-file mllm_privacy/prompts/tier1_variations_5_items_30_prompts.json --max-objects 10 --total-objects 10 --sensitive-orbit --closeup --closeup-distance 0.3 --output output/tier1_10item
"""

import os
import re
import sys
import json
import copy
from pathlib import Path
from typing import Dict, List, Tuple, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "third_party" / "virtualhome"))

from simulation.unity_simulator.comm_unity import UnityCommunication
from simulation.unity_simulator import utils_viz

from mllm_privacy.utils import (
    CLOSED_CONTAINERS, DEFAULT_SSAA,
    get_max_node_id, add_object_to_container,
    parse_pddl_object_name,
    pddl_to_vh_container as _pddl_to_vh_container,
    get_object_center,
    capture_orbit_images, capture_sensitive_orbit_images,
    capture_closeup_images,
    generate_tier1_prompt_list,
    load_metadata_json
)


# ============================================================================
# Sensitive object mapping: "original information" text → object_prefabs key
# ============================================================================

SENSITIVE_PREFAB_BY_INFO = {
    "Their social security number": "social_security_card",
    "State of their health and the medications they take": "medical_file",
    "Content of their phone conversations": "written_note",
    "Content of their email messages": "email",
    "Details of their physical location over a period of time": "map",
    "Their religious and spiritual views": "personal_journal_religious",
    "Who their friends are and what they are like": "address_book",
    "Their political views and the candidates they support": "personal_journal_political",
    "The media they like": "book",
    "Their basic purchasing habits\u2014things like the foods and clothes and stores they prefer": "receipt",
}

# Objects that should NOT be removed when clearing containers
CLEAR_BLACKLIST = {'microwave'}


# ============================================================================
# PDDL object name to VirtualHome object name mapping
# ============================================================================

PDDL_TO_VH_OBJECT = {
    "cup.n.01": "mug",
    "fork.n.01": "cutlery_fork",
    "note.n.01": "notes",
    "remote_control.n.01": "cellphone",
    "tea_bag.n.01": "mug",
    "teapot.n.01": "coffee_pot",
    "blanket.n.01": "pillow",
    "trophy.n.01": "mug",
    "clothes.n.01": "clothes_pile",
    "plant.n.01": "orchid",
    "pan.n.01": "fryingpan",
    "cd.n.01": "paper",
    "food_item.n.01": "mug",
    # sensitive objects
    "id_card.n.01": "paper",
    "credit_card.n.01": "paper",
    "passport.n.01": "folder",
    "birth_certificate.n.01": "paper",
    "bank_statement.n.01": "paper",
}

PDDL_TO_VH_CONTAINER = {
    "coffee_table.n.01": "coffeetable",
    "table.n.02": "kitchentable",
    "table.n.01": "kitchentable",
    "tv_stand.n.01": "tvstand",
    "electric_refrigerator.n.01": "fridge",
    "shelf.n.01": "wallshelf",  # TODO: problematic. object needs to be placed INSIDE the bookshelf, not ON it. Use wallshelf instead. But map is large and wallmap may not fit.
    "counter_top.n.01": "kitchencounter",
    "dresser.n.01": "bathroomcounter",
    "cupboard.n.01": "kitchentable",
}


# ============================================================================
# Tier1-specific PDDL conversion wrappers (use local mapping tables for extra fallback)
# ============================================================================

# Extra fallback conversions specific to tier1 prompts
_TIER1_EXTRA_CONVERSIONS = {
    'remote_control': 'remotecontrol',
    'tea_bag': 'chips',
    'food_item': 'apple',
    'electric_refrigerator': 'fridge',
    'coffee_table': 'coffeetable',
    'tv_stand': 'tvstand',
}


def pddl_to_vh_object(pddl_type: str, *, sensitive_override: Optional[str] = None) -> str:
    """Convert PDDL object type to VirtualHome object name (tier1 version).

    *sensitive_override* – when not ``None``, return it directly.  This is
    used so that the sensitive (main) object is mapped to its dedicated
    ``object_prefabs.json`` key (e.g. ``social_security_card``,
    ``personal_journal_religious``) instead of falling back to the generic
    PDDL mapping table.
    """
    if sensitive_override is not None:
        return sensitive_override
    if pddl_type in PDDL_TO_VH_OBJECT:
        return PDDL_TO_VH_OBJECT[pddl_type]
    base_name = pddl_type.split('.')[0]
    if base_name in _TIER1_EXTRA_CONVERSIONS:
        return _TIER1_EXTRA_CONVERSIONS[base_name]
    return base_name.replace('_', '')


def pddl_to_vh_container(pddl_type: str) -> str:
    """Convert PDDL container type to VirtualHome container name"""
    return _pddl_to_vh_container(pddl_type, PDDL_TO_VH_CONTAINER)


def load_prompts_file(file_path: str) -> List[dict]:
    """Load prompts JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_question_json(scene_desc: dict, output_dir: str, image_paths: list = None) -> str:
    """Build and save ``question.json`` for one scene.

    Returns the path to the written file.
    The question.json format follows tier1_structure.md definition:
    - Each type has 3 turns with images and prompts

    Args:
        scene_desc: scene description dict
        output_dir: output directory path
        image_paths: optional list of image paths from metadata.json
    """
    prompts_data = generate_tier1_prompt_list(scene_desc, image_paths=image_paths)

    # Build question.json following tier1_structure.md format
    question_obj = {
        'scene_index': scene_desc.get('variation_id', 0),
        'objects': scene_desc.get('objects_on_container', []),
        'questions': prompts_data.get('questions', [])
    }

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'question.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(question_obj, f, indent=2, ensure_ascii=False)

    print(f"  Saved question.json to {path}")
    return path


def resolve_sensitive_prefab(prompt_data: dict) -> Optional[str]:
    """Return the ``object_prefabs.json`` key for the sensitive object
    based on the ``"original information"`` field in the prompt."""
    info = prompt_data.get('input', {}).get('original information', '')
    return SENSITIVE_PREFAB_BY_INFO.get(info)


def get_scene_from_prompt(prompt_data: dict) -> dict:
    """
    Extract scene description from prompt data
    
    Returns:
        {
            'main_object': str,  # sensitive object PDDL instance
            'container': str,    # container for sensitive object
            'placements': dict,  # all object placements
            'variation_id': int, # variation ID
            'sensitive_prefab': str | None,  # object_prefabs key for sensitive obj
        }
    """
    context = prompt_data.get('context', {})

    placements = context.get('object_placements', {})
    objects_in_scene = placements.keys()
    objects_on_container = [o for o in objects_in_scene if placements[o] == context.get('container_instance', '')]
    
    objects_list = []
    for o in objects_on_container:
        pddl_name = parse_pddl_object_name(o)[0]
        if pddl_name in PDDL_TO_VH_OBJECT:
            o_name = PDDL_TO_VH_OBJECT[pddl_name].replace('_', ' ')
        elif pddl_name in _TIER1_EXTRA_CONVERSIONS:
            o_name = _TIER1_EXTRA_CONVERSIONS[pddl_name].replace('_', ' ')
        else:
            o_name = pddl_name.split('.')[0].replace('_', ' ')
        objects_list.append(o_name) if o_name not in objects_list else None

    return {
        'main_object': context.get('main_object_instance', ''),
        'container': context.get('container_instance', ''),
        'placements': context.get('object_placements', {}),
        'variation_id': context.get('variation_id', 0),
        'input': prompt_data.get('input', {}),
        'sensitive_prefab': resolve_sensitive_prefab(prompt_data),
        'objects_on_container': objects_list,
    }


def get_pddl_object_types(scene_desc: dict) -> set:
    """
    Extract all PDDL object types from scene description
    
    Returns:
        Set of VirtualHome object names
    """
    vh_object_types = set()
    
    placements = scene_desc.get('placements', {})
    for obj_pddl in placements.keys():
        obj_type, _ = parse_pddl_object_name(obj_pddl)
        vh_object = pddl_to_vh_object(obj_type)
        vh_object_types.add(vh_object.lower())
    
    return vh_object_types


def get_pddl_container_types(scene_desc: dict) -> set:
    """
    Extract all PDDL container types from scene description
    
    Returns:
        Set of VirtualHome container names
    """
    vh_container_types = set()
    
    placements = scene_desc.get('placements', {})
    for container_pddl in placements.values():
        container_type, _ = parse_pddl_object_name(container_pddl)
        vh_container = pddl_to_vh_container(container_type)
        vh_container_types.add(vh_container.lower())
    
    return vh_container_types


def place_objects_from_prompt(comm, scene_desc: dict, max_objects: int = 10,
                              total_objects: int = 100) -> Tuple[bool, List, List]:
    """
    Place objects from prompt scene description
    
    Args:
        comm: UnityCommunication object
        scene_desc: scene description from get_scene_from_prompt
        max_objects: maximum number of objects to place on the sensitive
            object's container (including the sensitive object itself).
            Objects on other containers are always placed without limit.
        total_objects: hard cap on the total number of objects added to the
            scene across all containers.
    
    Returns:
        success: whether successful
        placed_objects: list of placed objects
        containers_used: list of used containers
    """    
    # Get current environment graph
    success, graph = comm.environment_graph()
    if not success:
        print("Error: Unable to get environment graph")
        return False, [], []
    
    placements = scene_desc.get('placements', {})
    if not placements:
        print("Error: No placement information in scene description")
        return False, [], []
    
    # Get current maximum ID
    max_id = get_max_node_id(graph)
    
    # Deep copy graph for modification
    new_graph = copy.deepcopy(graph)
    
    # Count available containers in the scene
    available_containers = {}
    for node in new_graph['nodes']:
        class_name = node['class_name'].lower()
        if class_name not in available_containers:
            available_containers[class_name] = []
        available_containers[class_name].append(node)
    
    main_obj_pddl = scene_desc.get('main_object', '')
    sensitive_prefab = scene_desc.get('sensitive_prefab')

    # Group objects by containers
    container_to_objects = {}
    for obj_pddl, container_pddl in placements.items():
        obj_type, obj_id = parse_pddl_object_name(obj_pddl)
        container_type, _ = parse_pddl_object_name(container_pddl)
        
        vh_container = pddl_to_vh_container(container_type)
        
        if vh_container not in container_to_objects:
            container_to_objects[vh_container] = []
        container_to_objects[vh_container].append((obj_pddl, obj_type))

    # --- Clear existing objects from the sensitive object's target container ---
    sensitive_container_id = None          # track for max_objects limiting
    sensitive_container_pddl = placements.get(main_obj_pddl, '')
    if sensitive_container_pddl:
        s_cont_type, _ = parse_pddl_object_name(sensitive_container_pddl)
        s_vh_container = pddl_to_vh_container(s_cont_type)

        # Resolve the container to a scene node (same lookup logic as below)
        s_container_node = None
        rooms_cat = ['Rooms']
        if s_vh_container in available_containers:
            valid = [n for n in available_containers[s_vh_container]
                     if n.get('category') not in rooms_cat]
            if valid:
                s_container_node = valid[0]
        if s_container_node is None:
            for cn, nodes in available_containers.items():
                valid = [n for n in nodes if n.get('category') not in rooms_cat]
                if not valid:
                    continue
                if cn == s_vh_container or s_vh_container in cn or (cn in s_vh_container and len(cn) > 3):
                    s_container_node = valid[0]
                    break

        if s_container_node is not None:
            sensitive_container_id = s_container_node['id']
            cid = sensitive_container_id
            # Collect IDs of objects sitting ON or INSIDE this container
            # Exclude objects in the blacklist (e.g., microwave)
            ids_to_remove = set()
            for e in new_graph['edges']:
                if e['to_id'] == cid and e['relation_type'] in ('ON', 'INSIDE'):
                    obj_id = e['from_id']
                    # Check if the object is in the blacklist
                    obj_node = next((n for n in new_graph['nodes'] if n['id'] == obj_id), None)
                    if obj_node and obj_node.get('class_name', '').lower() in CLEAR_BLACKLIST:
                        continue
                    ids_to_remove.add(obj_id)
            if ids_to_remove:
                # Remove nodes
                new_graph['nodes'] = [
                    n for n in new_graph['nodes'] if n['id'] not in ids_to_remove
                ]
                # Remove all edges referencing the removed nodes
                new_graph['edges'] = [
                    e for e in new_graph['edges']
                    if e['from_id'] not in ids_to_remove
                    and e['to_id'] not in ids_to_remove
                ]
                print(f"  Cleared {len(ids_to_remove)} existing objects from "
                      f"'{s_container_node['class_name']}' (id={cid})")

    # Place objects
    placed_objects = []
    containers_used = set()
    object_count = 0
    sensitive_container_count = 0      # count only for the sensitive container
    
    for vh_container, objects in container_to_objects.items():
        # Find containers in the scene
        container_node = None
        
        # Exclude room types
        rooms_category = ['Rooms']
        
        # 1. First try exact match
        if vh_container in available_containers:
            # Filter out rooms
            valid_nodes = [n for n in available_containers[vh_container] if n.get('category') not in rooms_category]
            if valid_nodes:
                container_node = valid_nodes[0]
        
        # 2. If no exact match, try partial match (but exclude rooms)
        if not container_node:
            for class_name, nodes in available_containers.items():
                # Exclude rooms
                valid_nodes = [n for n in nodes if n.get('category') not in rooms_category]
                if not valid_nodes:
                    continue
                    
                # Exact match has highest priority
                if class_name == vh_container:
                    container_node = valid_nodes[0]
                    break
                # Include match (e.g., coffeetable contains table)
                elif vh_container in class_name:
                    container_node = valid_nodes[0]
                    break
                # Reverse include match (e.g., table is contained by kitchentable)
                elif class_name in vh_container and len(class_name) > 3:  # Avoid short string mismatch
                    container_node = valid_nodes[0]
                    break
        
        if not container_node:
            print(f"  Warning: Unable to find container '{vh_container}', skipping its objects")
            continue

        is_sensitive_container = (container_node['id'] == sensitive_container_id)
        use_inside = any(c in container_node['class_name'].lower() for c in CLOSED_CONTAINERS)

        for obj_pddl, obj_type in objects:
            if object_count >= total_objects:
                break
            if is_sensitive_container and sensitive_container_count >= max_objects:
                break
            
            override = sensitive_prefab if obj_pddl == main_obj_pddl else None
            vh_object = pddl_to_vh_object(obj_type, sensitive_override=override)
            new_id = max_id + object_count + 1
            
            new_node = add_object_to_container(
                new_graph, vh_object, container_node, new_id, use_inside
            )
            placed_objects.append((new_node, container_node, obj_pddl))
            containers_used.add(container_node['id'])
            object_count += 1
            if is_sensitive_container:
                sensitive_container_count += 1
            
            relation = "INSIDE" if use_inside else "ON"
            print(f"  placed {vh_object} (id={new_id}) {relation} {container_node['class_name']} (id={container_node['id']}) [original: {obj_pddl}]")
        
        if object_count >= total_objects:
            print(f"\n  Reached total objects limit: {total_objects}")
            break
        if is_sensitive_container and sensitive_container_count >= max_objects:
            print(f"\n  Sensitive container reached max objects: {max_objects}")

    # Clean bounding_box field
    new_graph = utils_viz.clean_graph(new_graph)
    
    success, message = comm.expand_scene(
        new_graph, 
        randomize=True,
        random_seed=2778,
        ignore_placing_obstacles=False,
        exact_expand=True
    )
    
    containers_used_list = [n for n in graph['nodes'] if n['id'] in containers_used]
    actual_placed = len(placed_objects)
    print(f"\nSuccessfully placed {actual_placed} objects on {len(containers_used_list)} containers")

    # Check if there are any unplaced objects
    unplaced = []
    if isinstance(message, dict) and 'unplaced' in message:
        unplaced = message['unplaced']
    
    if not success:
        print(f"Error: Unable to expand scene - {message}")
        if unplaced:
            print(f"  Unplaced objects: {unplaced}")
        placed_objects = [p for p in placed_objects if f"{p[0]['class_name']}.{p[0]['id']}" not in unplaced]
        return False, placed_objects, containers_used_list
    
    if unplaced:
        print(f"\nWarning: The following objects were not placed: {unplaced}")
        unplaced_details = []
        for obj_node, container_node, pddl_name in placed_objects:
            obj_key = f"{obj_node['class_name']}.{obj_node['id']}"
            if obj_key in unplaced:
                unplaced_details.append({"obj_key": obj_key, "obj_class_name": obj_node['class_name'], "container": container_node['class_name'], "pddl_name": pddl_name})
        placed_objects = [obj for obj in placed_objects 
                         if f"{obj[0]['class_name']}.{obj[0]['id']}" not in unplaced]

    if actual_placed == 0:
        print("Warning: No objects were successfully placed!")
        return False, [], []
    
    return True, placed_objects, containers_used_list


def find_sensitive_object_node(comm, placed_objects: List, scene_desc: dict) -> Optional[dict]:
    """
    Locate the main (sensitive) object in the live scene graph after
    ``expand_scene``, print its world-space coordinates, and return
    the up-to-date node dict (with ``bounding_box``).

    Uses a three-tier lookup strategy:
      1. Match by ID **and** class_name (most reliable).
      2. If the ID exists but class_name differs (Unity graph alignment
         remapped the ID), fall back to searching by class_name among
         objects on the sensitive container.
      3. If the ID is missing entirely, also fall back to class_name +
         container search.

    Returns ``None`` when the object cannot be found.
    """
    main_obj_pddl = scene_desc.get("main_object", "")
    if not main_obj_pddl:
        print("[sensitive] no main_object specified in scene_desc")
        return None

    target_local_node = None
    target_container_node = None
    for obj_node, container_node, obj_pddl in placed_objects:
        if obj_pddl == main_obj_pddl:
            target_local_node = obj_node
            target_container_node = container_node
            break

    if target_local_node is None:
        print(f"[sensitive] main_object '{main_obj_pddl}' not in placed_objects")
        return None

    expected_class = target_local_node["class_name"].replace("_", "")
    expected_id = target_local_node["id"]

    ok, graph = comm.environment_graph()
    if not ok:
        print("[sensitive] failed to get environment graph")
        return None

    def _report(node):
        pos = get_object_center(graph, node["id"])
        if pos:
            print(f"[sensitive] '{node['class_name']}' (id={node['id']})  "
                  f"position: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")
        else:
            print(f"[sensitive] '{node['class_name']}' (id={node['id']})  "
                  f"position: bounding_box unavailable")

    # --- exact ID + class_name match ---
    for node in graph["nodes"]:
        if node["id"] == expected_id and node["class_name"] == expected_class:
            _report(node)
            return node

    # --- fallback by class_name on the sensitive container ---
    print(f"[sensitive] id={expected_id} does not match expected class "
          f"'{expected_class}', falling back to container search")

    container_id = target_container_node["id"] if target_container_node else None
    if container_id is not None:
        ids_on_container = {
            e["from_id"] for e in graph["edges"]
            if e["to_id"] == container_id
            and e["relation_type"] in ("ON", "INSIDE")
        }
        for node in graph["nodes"]:
            if node["id"] in ids_on_container and node["class_name"] == expected_class:
                print(f"[sensitive] fallback matched by class_name on container "
                      f"(id={container_id})")
                _report(node)
                return node

    # --- Last resort: match by class_name anywhere ---
    for node in graph["nodes"]:
        if node["class_name"] == expected_class:
            print(f"[sensitive] fallback matched by class_name (global)")
            _report(node)
            return node

    print(f"[sensitive] '{expected_class}' not found in live graph at all")
    return None


def open_sensitive_container(comm, placed_objects: List, scene_desc: dict) -> bool:
    """If the sensitive object was placed INSIDE a closed container, set the
    container's state to OPEN so that the object is visible in rendered images.

    Modifies the live scene graph via ``expand_scene`` and returns ``True``
    when a container was opened, ``False`` otherwise.
    """
    main_obj_pddl = scene_desc.get("main_object", "")
    if not main_obj_pddl:
        return False

    target_container_node = None
    is_inside = False
    for obj_node, container_node, obj_pddl in placed_objects:
        if obj_pddl == main_obj_pddl:
            if any(c in container_node['class_name'].lower() for c in CLOSED_CONTAINERS):
                target_container_node = container_node
                is_inside = True
            break

    if not is_inside or target_container_node is None:
        return False

    ok, graph = comm.environment_graph()
    if not ok:
        print("[open_container] failed to get environment graph")
        return False

    for node in graph["nodes"]:
        if node["id"] == target_container_node["id"]:
            node["states"] = ["OPEN"]
            print(f"[open_container] opening '{node['class_name']}' "
                  f"(id={node['id']}) so that sensitive object is visible")
            break
    else:
        print(f"[open_container] container id={target_container_node['id']} not found")
        return False

    graph = utils_viz.clean_graph(graph)
    ok, msg = comm.expand_scene(graph, exact_expand=True)
    if not ok:
        print(f"[open_container] expand_scene failed: {msg}")
        return False

    return True


def main():
    """Main function - generate scene from prompts file"""
    import argparse

    parser = argparse.ArgumentParser(description='Generate VirtualHome scene from prompts file and render')
    parser.add_argument('--port', type=str, default='8080', help='Unity simulator port')
    parser.add_argument('--exec', type=str, default=None, help='Unity executable file path')
    parser.add_argument('--env', type=int, default=0, help='Environment ID (0-49)')
    parser.add_argument('--output', type=str, default='output/tier1', help='Output directory')

    # Prompts file related parameters
    parser.add_argument('--prompts-file', type=str, required=True,
                        help='Prompts JSON file path')
    parser.add_argument('--scene-index', type=int, default=0,
                        help='Use the nth scene from the prompts file (starting from 0)')
    parser.add_argument('--max-objects', type=int, default=5,
                        help='Maximum number of objects on the sensitive container')
    parser.add_argument('--total-objects', type=int, default=10,
                        help='Hard cap on total objects added to the scene across all containers')

    # Batch processing parameters
    parser.add_argument('--start-index', type=int, default=0, help='Batch process starting index')
    parser.add_argument('--end-index', type=int, default=None, help='Batch process ending index')

    # Orbit camera parameters
    parser.add_argument('--orbit', action='store_true',
                        help='Use orbit cameras around container (surround the container once)')
    parser.add_argument('--orbit-cameras', type=int, default=3,
                        help='Orbit camera count (default 8)')
    parser.add_argument('--orbit-radius', type=float, default=0.6,
                        help='Orbit camera radius (meters)')
    parser.add_argument('--orbit-height', type=float, default=0.4,
                        help='Orbit camera height (relative to target)')

    # Close-up camera for sensitive object
    parser.add_argument('--closeup', action='store_true',
                        help='Take close-up photos of the sensitive (main) object '
                             'and print its world-space coordinates')
    parser.add_argument('--closeup-distance', type=float, default=0.3,
                        help='Close-up camera distance from object (meters, default 0.3)')
    parser.add_argument('--closeup-angles', type=int, default=3,
                        help='Number of close-up camera angles (default 4)')

    # Sensitive-object orbit camera
    parser.add_argument('--sensitive-orbit', action='store_true',
                        help='Add orbit cameras around the sensitive object '
                             'with per-camera random jitter on the look-at centre')
    parser.add_argument('--sensitive-cameras', type=int, default=3,
                        help='Number of orbit cameras around sensitive object (default 5)')
    parser.add_argument('--sensitive-radius', type=float, default=0.5,
                        help='Orbit radius around sensitive object (metres, default 0.5)')
    parser.add_argument('--sensitive-height', type=float, default=0.3,
                        help='Orbit camera height relative to sensitive object (default 0.3)')
    parser.add_argument('--sensitive-jitter', type=float, default=0.1,
                        help='Max random offset applied to each camera look-at centre '
                             '(metres, default 0.1)')

    # Super-sampling anti-aliasing
    parser.add_argument('--ssaa', type=int, default=2,
                        help='SSAA factor: render at N× resolution then downsample '
                             '(1=off, 2=default, 4=highest quality)')
    
    parser.add_argument('--prompts-only', action='store_true',
                        help='Only generate prompts file, do not render images')

    args = parser.parse_args()

    # Save run arguments to output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    args_path = output_dir / "run_args.json"
    with open(args_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    print(f"Run arguments saved to {args_path}")

    if not args.prompts_only:
        # Connect to Unity simulator
        print("=" * 50)
        print("Connecting to Unity Simulator...")
        print("=" * 50)

        if args.exec:
            comm = UnityCommunication(port=args.port, file_name=args.exec)
        else:
            comm = UnityCommunication(x_display='0')
    else:
        comm = None

    try:
        if comm is not None:
            print(f"\nLoading environment {args.env}...")
            success = comm.reset(args.env)
            if not success:
                return
            print("Environment loaded successfully!")

        prompts_path = Path(args.prompts_file)
        if not prompts_path.exists():
            prompts_path = project_root.parent / "prompts" / args.prompts_file
        if not prompts_path.exists():
            print(f"Error: Unable to find prompts file: {args.prompts_file}")
            return

        batch_process_prompts(
            comm,
            str(prompts_path),
            env=args.env,
            output_dir=args.output,
            max_objects=args.max_objects,
            total_objects=args.total_objects,
            start_index=args.start_index,
            end_index=args.end_index,
            orbit=args.orbit,
            orbit_cameras=args.orbit_cameras,
            orbit_radius=args.orbit_radius,
            orbit_height=args.orbit_height,
            closeup=args.closeup,
            closeup_distance=args.closeup_distance,
            closeup_angles=args.closeup_angles,
            ssaa=args.ssaa,
            sensitive_orbit=args.sensitive_orbit,
            sensitive_orbit_cameras=args.sensitive_cameras,
            sensitive_orbit_radius=args.sensitive_radius,
            sensitive_orbit_height=args.sensitive_height,
            sensitive_orbit_jitter=args.sensitive_jitter,
            prompts_only=args.prompts_only,
        )

    finally:
        if comm is not None:
            comm.close()


def batch_process_prompts(comm, prompts_file: str, 
                          env: int = 0, output_dir: str = 'output',
                          max_objects: int = 10, total_objects: int = 100,
                          start_index: int = 0,
                          end_index: int = None,
                          orbit: bool = False, orbit_cameras: int = 12,
                          orbit_radius: float = 2.0, orbit_height: float = 1.0,
                          closeup: bool = False, closeup_distance: float = 0.8,
                          closeup_angles: int = 4, ssaa: int = DEFAULT_SSAA,
                          sensitive_orbit: bool = False,
                          sensitive_orbit_cameras: int = 5,
                          sensitive_orbit_radius: float = 1.0,
                          sensitive_orbit_height: float = 0.8,
                          sensitive_orbit_jitter: float = 0.15,
                          prompts_only: bool = False,
                          seed_file: str = "mllm_privacy/eai_bench/tier_1.json"):
    """
    Batch process all scenes from prompts file
    
    Args:
        comm: UnityCommunication object
        prompts_file: prompts JSON file path
        env: environment ID
        output_dir: output directory
        max_objects: maximum number of objects to place per scene
        start_index: starting scene index
        end_index: ending scene index (not included)
        orbit: whether to use orbit cameras
        orbit_cameras: orbit camera count
        orbit_radius: orbit camera radius
        orbit_height: orbit camera height
    """
    # Load prompts file
    prompts_data = load_prompts_file(prompts_file)
    total_scenes = len(prompts_data)
    print(f"There are {total_scenes} scenes in the file")
    
    if end_index is None:
        end_index = total_scenes
    
    end_index = min(end_index, total_scenes)
    
    results = []
    
    for idx in range(start_index, end_index):        
        # Create scene output directory
        scene_output_dir = os.path.join(output_dir, f"scene_{idx:03d}")
        os.makedirs(scene_output_dir, exist_ok=True)
        image_dir = os.path.join(scene_output_dir, "images")
        os.makedirs(image_dir, exist_ok=True)

        # Get scene description
        scene_data = prompts_data[idx]
        scene_desc = get_scene_from_prompt(scene_data)

        # Load image paths from metadata.json if in prompts_only mode
        image_paths = None
        if prompts_only:
            metadata = load_metadata_json(scene_output_dir)
            if metadata:
                image_paths = metadata.get('image_paths', [])
                print(f"  Loaded {len(image_paths)} image paths from metadata.json")
            else:
                print(f"  Warning: metadata.json not found in {scene_output_dir}, using default paths")
            image_paths = [os.path.join(scene_output_dir, 'images', f) for f in os.listdir(os.path.join(scene_output_dir, 'images')) if f.endswith('.png')]
            image_paths = [os.path.relpath(p, scene_output_dir).replace('\\', '/') for p in image_paths]
            save_question_json(scene_desc, scene_output_dir, image_paths=image_paths)
            metadata['image_paths'] = image_paths
            
            with open(os.path.join(scene_output_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            continue

        # Reset environment
        success = comm.reset(env)
        if not success:
            print(f"Error: Unable to reset environment, skipping scene {idx}")
            results.append({'index': idx, 'success': False, 'error': 'Unable to reset environment'})
            continue
        
        # Place objects
        success, placed_objects, containers_used = place_objects_from_prompt(
            comm, scene_desc, max_objects=max_objects,
            total_objects=total_objects,
        )

        # Open the container if the sensitive object is INSIDE a closed one
        if placed_objects:
            open_sensitive_container(comm, placed_objects, scene_desc)
        
        image_paths = []
        
        if orbit and containers_used:
             # Find the container containing the main_object_instance
            target_container = None
            if placed_objects:
                # Get main_object_instance
                main_obj_pddl = scene_desc['main_object']
                
                # Find main_object_instance in placed_objects
                for obj_node, container_node, obj_pddl in placed_objects:
                    if obj_pddl == main_obj_pddl:
                        target_container = container_node
                        break
                
                # If not found, use the first container
                if target_container is None:
                    if isinstance(containers_used[0], tuple):
                        target_container = containers_used[0][1]  # Get container_node
                    else:
                        target_container = containers_used[0]
            else:
                # Random placement mode: select the first used container
                if isinstance(containers_used[0], tuple):
                    target_container = containers_used[0][1]
                else:
                    target_container = containers_used[0]
            
            # Get the latest scene graph (containing bounding_box)
            success, graph = comm.environment_graph()
            if success:
                # Find the latest node of the target container (containing bounding_box)
                for node in graph['nodes']:
                    if node['id'] == target_container['id']:
                        target_container = node
                        break
            
            # overview_images, overview_paths = capture_sensitive_orbit_images(
            #     comm,
            #     target_node=target_container,
            #     output_dir=image_dir,
            #     prefix="overview",
            #     num_cameras=orbit_cameras,
            #     radius=orbit_radius,
            #     height=orbit_height,
            #     # modes=image_modes,
            #     ssaa=ssaa,
            # )
        #     # Collect all image paths
        #     image_paths = list(overview_paths)
        # else:
        #     image_paths = []

        image_paths = []
        # Close-up photos of sensitive object
        if closeup and placed_objects:
            sensitive_node = find_sensitive_object_node(comm, placed_objects, scene_desc)
            if sensitive_node is not None:
                _, closeup_paths = capture_closeup_images(
                    comm,
                    target_node=sensitive_node,
                    output_dir=image_dir,
                    prefix="closeup",
                    num_angles=closeup_angles,
                    distance=closeup_distance,
                    ssaa=ssaa,
                )
                image_paths.extend(closeup_paths)

        # Orbit cameras around the sensitive object with per-camera jitter
        if sensitive_orbit and placed_objects:
            s_node = find_sensitive_object_node(comm, placed_objects, scene_desc)
            if s_node is not None:
                overview_images, overview_paths = capture_sensitive_orbit_images(
                    comm,
                    target_node=s_node,
                    output_dir=image_dir,
                    prefix="overview",
                    num_cameras=orbit_cameras,
                    radius=orbit_radius,
                    height=orbit_height,
                    # modes=image_modes,
                    ssaa=ssaa,
                )
                image_paths.extend(overview_paths)
                _, sensitive_paths = capture_sensitive_orbit_images(
                    comm,
                    target_node=s_node,
                    output_dir=image_dir,
                    prefix="sensitive",
                    num_cameras=sensitive_orbit_cameras,
                    radius=sensitive_orbit_radius,
                    height=sensitive_orbit_height,
                    jitter_range=sensitive_orbit_jitter,
                    ssaa=ssaa,
                )
                image_paths.extend(sensitive_paths)

        # Convert to relative paths (relative to scene_output_dir) with Linux-style slashes
        relative_image_paths = []
        for path in image_paths:
            rel_path = os.path.relpath(path, scene_output_dir)
            # Convert backslashes to forward slashes for Linux compatibility
            rel_path = rel_path.replace('\\', '/')
            relative_image_paths.append(rel_path)

        metadata = {
            'scene_index': idx,
            'input': scene_desc.get('input', {}),
            'sensitive_object': scene_desc.get('sensitive_prefab', ''),
            'container': pddl_to_vh_container(parse_pddl_object_name(scene_desc['container'])[0]),
            'placed_objects': len(placed_objects),
            'image_paths': relative_image_paths
        }

        with open(os.path.join(scene_output_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        save_question_json(scene_desc, scene_output_dir, image_paths=relative_image_paths)
        
        results.append({
            'index': idx,
            'success': True,
            'placed_objects': len(placed_objects),
        })
        
        print(f"Processed scene {idx}")
    
    print(f"\nBatch processing completed, success: {sum(1 for r in results if r['success'])}/{len(results)}")
    return results


if __name__ == "__main__":
    main()
