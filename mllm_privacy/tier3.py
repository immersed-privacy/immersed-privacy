# Copyright (C) 2025 Junran Wang and Zehao Jin
#
# This file is part of the VLM Privacy Evaluation.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Tier3
"""

import os
import sys
import json
import copy
import random
import time
import argparse
import traceback
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ---------------------------------------------------------------------------
# project paths
# ---------------------------------------------------------------------------
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "third_party" / "virtualhome"))

from simulation.unity_simulator.comm_unity import UnityCommunication
from simulation.unity_simulator import utils_viz

from mllm_privacy.utils import (
    get_max_node_id, find_room_for_object,
    add_object_to_container, find_container_in_scene,
    capture_orbit_images, capture_images,
    CLOSED_CONTAINERS,  TABLE_CLASS_NAMES, LARGE_TABLES, MEDIUM_TABLES, DEFAULT_SSAA,
    generate_tier3_prompt,
    load_metadata_json,
)


# ============================================================================
# helpers
# ============================================================================

def _normalize_audio_key(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _prepare_tier3_dialogue_audio_files(
    audio_source_dir: str,
    scene_index: int,
    output_dir: str,
) -> List[str]:
    """Copy tier3 dialogue audios for one scene into <output_dir>/audio."""
    if not audio_source_dir:
        return []
    if not os.path.isdir(audio_source_dir):
        print(f"  [WARN] dialogue audio source not found: {audio_source_dir}")
        return []

    scene_key = f"scene_{scene_index:03d}"
    candidates = []
    preferred_dir = os.path.join(audio_source_dir, scene_key)
    if os.path.isdir(preferred_dir):
        for fname in os.listdir(preferred_dir):
            if fname.lower().endswith(".wav"):
                candidates.append(os.path.join(preferred_dir, fname))
    else:
        for root, _, files in os.walk(audio_source_dir):
            root_key = _normalize_audio_key(root)
            if _normalize_audio_key(scene_key) not in root_key:
                continue
            for fname in files:
                if fname.lower().endswith(".wav"):
                    candidates.append(os.path.join(root, fname))

    candidates.sort()
    if not candidates:
        print(f"  [WARN] no dialogue audios found for {scene_key}")
        return []

    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    rel_paths = []
    seen = set()
    copied_new = 0
    reused = 0
    for src in candidates:
        dst = os.path.join(audio_dir, os.path.basename(src))
        if os.path.exists(dst):
            reused += 1
        else:
            shutil.copy2(src, dst)
            copied_new += 1
        rel = os.path.relpath(dst, output_dir).replace("\\", "/")
        if rel not in seen:
            seen.add(rel)
            rel_paths.append(rel)

    print(f"  [INFO] prepared {len(rel_paths)} dialogue audio file(s) "
          f"(copied={copied_new}, reused={reused})")
    return rel_paths


def save_tier3_question_json(scene_desc: dict, output_dir: str, image_paths: list,
                             video_path: str, audio_paths: Optional[List[str]] = None,
                             include_audio: bool = True) -> str:
    """
    Save question.json for tier3 scenario.

    Args:
        scene_desc: scene description dict
        output_dir: output directory path
        include_audio: if True, audio files are attached and prompts reference audio;
                       if False, no audio is attached and dialogue text is inlined.
    """
    question_obj = generate_tier3_prompt(
        scene_desc, image_paths, video_path, audio_paths=audio_paths or [],
        include_audio=include_audio,
    )

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'question.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(question_obj, f, indent=2, ensure_ascii=False)

    print(f"  [saved] {path}")
    return path


# ============================================================================
# Scene extraction
# ============================================================================

def extract_scene_info(scenario: dict, tier: str = "3") -> dict:
    """
    Parse a single tier‑3 JSON entry into a structured scene description.

    Returns dict with keys:
        scenario_name, tier,
        secret_item  {name, vh_type, vh_name},
        target_item  {name, vh_type, vh_name},
        other_items  [{name, vh_type, vh_name}, ...],
        container    {name, vh_type, vh_name},
        destination  {name, vh_type, vh_name} | None,
        physical_actions  [{actor, description}, ...],
        placement_type   "ontop" | "nextto" | None  (tier3b only),
        people       {person1, person2, person3}
    """
    secret = scenario["secret_item"]
    target = scenario["target_item"]
    container = scenario["container"]
    others = scenario.get("other_items", [])
    dest = scenario.get("destination")
    placement_type = scenario.get("placement_type")

    # physical_action entries
    actions = []
    for inter in scenario.get("observed_interactions", []):
        if inter.get("type") == "physical_action":
            actions.append({
                "actor": inter.get("actor", ""),
                "description": inter.get("action_description", ""),
            })

    info = {
        "scenario_name": scenario.get("scenario_name", "unknown"),
        "tier": tier,
        "env_id": scenario.get("env_id", 0),
        "room": scenario.get("room", ""),
        "needs_second_part": scenario.get("needs_second_part", True),
        "secret_item": {
            "name": secret["name"],
            "vh_type": secret["vh_type"],
            "vh_name": secret["vh_type"],
        },
        "target_item": {
            "name": target["name"],
            "vh_type": target["vh_type"],
            "vh_name": target["vh_type"],
        },
        "other_items": [
            {
                "name": it["name"],
                "vh_type": it["vh_type"],
                "vh_name": it["vh_type"],
            }
            for it in others
        ],
        "container": {
            "name": container["name"],
            "vh_type": container["vh_type"],
            "vh_name": container["vh_type"],
        },
        "destination": None,
        "physical_actions": actions,
        "placement_type": placement_type,
        "people": scenario.get("people", {}),
    }

    if dest:
        info["destination"] = {
            "name": dest["name"],
            "vh_type": dest["vh_type"],
            "vh_name": dest["vh_type"],
        }

    return info


# ============================================================================
# Scene setup — place objects in VirtualHome
# ============================================================================

def _find_container_node(graph: dict, vh_name: str, room_name: str = "") -> Optional[dict]:
    """Find a container node in the scene graph by VH class name (fuzzy), optionally filtering by room."""
    vh_lower = vh_name.lower().replace("_", "")
    room_categories = {"Rooms"}

    def in_target_room(node_id: int) -> bool:
        if not room_name:
            return True
        r_id = find_room_for_object(graph, node_id)
        if r_id is None:
            return False
        for n in graph["nodes"]:
            if n["id"] == r_id:
                return room_name.lower() in n["class_name"].lower()
        return False

    # Exact match
    for node in graph["nodes"]:
        if node.get("category") in room_categories:
            continue
        if node["class_name"].lower() == vh_lower:
            if in_target_room(node["id"]):
                return node

    # Substring match
    for node in graph["nodes"]:
        if node.get("category") in room_categories:
            continue
        cn = node["class_name"].lower()
        if vh_lower in cn or cn in vh_lower:
            if in_target_room(node["id"]):
                return node

    return None


def setup_scene(comm, scene_info: dict) -> Tuple[bool, list, Optional[dict], Optional[dict]]:
    """
    Place all scenario objects onto the container surface.

    Returns:
        (success, placed_objects, container_node, destination_node)
        placed_objects is a list of (new_node, container_node, label_str).
    """
    success, graph = comm.environment_graph()
    if not success:
        print("  [ERROR] cannot get environment graph")
        return False, [], None, None

    # Capture existing IDs to distinguish new objects later
    existing_ids = {n["id"] for n in graph["nodes"]}

    new_graph = copy.deepcopy(graph)
    max_id = get_max_node_id(new_graph)

    # --- locate container ---------------------------------------------------
    vh_container = scene_info["container"]["vh_name"]
    target_room = scene_info.get("room", "")
    container_node = _find_container_node(new_graph, vh_container, target_room)
    if container_node is None:
        print(f"  [ERROR] container '{vh_container}' "
              f"(vh_type: {scene_info['container']['vh_type']}) not found in scene")
        return False, [], None, None

    use_inside = any(
        c in container_node["class_name"].lower() for c in CLOSED_CONTAINERS
    )

    # --- locate destination (tier3) ----------------------------------------
    dest_node = None
    if scene_info["destination"]:
        dest_node = _find_container_node(new_graph, scene_info["destination"]["vh_name"], target_room)

    # --- find a SEPARATE surface for the secret item -----------------------
    # The secret item starts on a different surface; the character will carry
    # it to the main container during the action phase.
    secret_surface = None
    
    def in_target_room_for_secret(node_id: int) -> bool:
        if not target_room:
            return True
        r_id = find_room_for_object(new_graph, node_id)
        if r_id is None:
            return False
        for n in new_graph["nodes"]:
            if n["id"] == r_id:
                return target_room.lower() in n["class_name"].lower()
        return False

    for node in new_graph["nodes"]:
        if node["id"] == container_node["id"]:
            continue
        if node["class_name"].lower() in LARGE_TABLES + MEDIUM_TABLES:
            if in_target_room_for_secret(node["id"]):
                secret_surface = node
                break
    if secret_surface is None:
        secret_surface = container_node

    # --- place items --------------------------------------------------------
    placed = []

    # secret item → placed on separate surface
    secret_item = scene_info["secret_item"]
    max_id += 1
    secret_use_inside = any(
        c in secret_surface["class_name"].lower() for c in CLOSED_CONTAINERS
    )
    new_node = add_object_to_container(
        new_graph, secret_item["vh_name"], secret_surface, max_id, secret_use_inside
    )
    placed.append((new_node, secret_surface, f"secret:{secret_item['name']}"))

    # target + other items → placed on the main container
    items_to_place = []
    items_to_place.append(("target", scene_info["target_item"]))
    for idx, oi in enumerate(scene_info["other_items"]):
        items_to_place.append((f"other_{idx}", oi))

    for label, item in items_to_place:
        max_id += 1
        new_node = add_object_to_container(
            new_graph, item["vh_name"], container_node, max_id, use_inside
        )
        placed.append((new_node, container_node, f"{label}:{item['name']}"))

    # --- expand scene -------------------------------------------------------
    new_graph = utils_viz.clean_graph(new_graph)
    ok, msg = comm.expand_scene(new_graph, randomize=True)
    if not ok:
        print(f"  [WARN] expand_scene returned error: {msg}")
        # continue anyway – some objects may still have been placed

    # --- Re-sync IDs with Unity ---------------------------------------------
    # expand_scene might assign different IDs than our local counter.
    # We must fetch the actual graph to get the correct IDs for the script.
    success_sync, real_graph = comm.environment_graph()
    if success_sync:
        real_nodes = {n["id"]: n for n in real_graph["nodes"]}
        
        # Build map: surface_id -> {class_name -> [list of new child nodes]}
        surface_map = {}
        for edge in real_graph["edges"]:
            if edge["relation_type"] in ("ON", "INSIDE"):
                s_id = edge["to_id"]
                c_id = edge["from_id"]
                
                # We only care about NEW objects (not in existing_ids)
                if c_id in existing_ids:
                    continue
                
                child_node = real_nodes.get(c_id)
                if not child_node:
                    continue
                    
                if s_id not in surface_map:
                    surface_map[s_id] = {}
                c_name = child_node["class_name"]
                if c_name not in surface_map[s_id]:
                    surface_map[s_id][c_name] = []
                surface_map[s_id][c_name].append(child_node)

        # Update 'placed' objects
        count_updated = 0
        for i, (p_node, p_surf, p_label) in enumerate(placed):
            surf_id = p_surf["id"]
            p_name = p_node["class_name"]
            
            if surf_id in surface_map and p_name in surface_map[surf_id]:
                candidates = surface_map[surf_id][p_name]
                if candidates:
                    # Pop the first match (greedy assignment)
                    matched_node = candidates.pop(0)
                    old_id = p_node["id"]
                    new_id = matched_node["id"]
                    
                    if old_id != new_id:
                        p_node["id"] = new_id
                        count_updated += 1
        

        if count_updated > 0:
            print(f"    [INFO] synced {count_updated} object IDs with Unity")

    # --- Print final placement status ---------------------------------------
    for (p_node, p_surf, p_label) in placed:
        is_inside = any(c in p_surf["class_name"].lower() for c in CLOSED_CONTAINERS)
        rel = "INSIDE" if is_inside else "ON"
        
        print(f"    + {p_node['class_name']} (id={p_node['id']}) "
              f"{rel} {p_surf['class_name']} (id={p_surf['id']})  "
              f"[{p_label}]")

    return True, placed, container_node, dest_node


# ============================================================================
# Character physical actions
# ============================================================================

def execute_physical_actions(
    comm,
    scene_info: dict,
    container_node: dict,
    placed_objects: list,  # List of (node, container, label) tuples from setup_scene
    output_dir: str = "output",
    prefix: str = "action_video",
    image_width: int = 1280,
    image_height: int = 720,
    frame_rate: int = 10,
) -> Tuple[bool, Optional[str]]:
    """
    Add a character and render the physical actions described in the scenario.
    
    The character will simulate the actions from observed_interactions by:
    1. Walking to the container
    2. Grabbing the secret_item
    3. Putting it back on the container (simulating placement/hiding)

    Returns (success, video_path | None).
    """
    actions = scene_info.get("physical_actions", [])
    if not actions:
        return True, None

    # Absolute path required by Unity
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Find the secret_item and target_item nodes from placed objects
    secret_node = None
    target_node = None
    for node, node_container, label in placed_objects:
        if "secret:" in label:
            secret_node = node
        elif "target:" in label:
            target_node = node
            
    if secret_node is None or target_node is None or secret_node.get("id") == -1 or target_node.get("id") == -1:
        print(f"    [WARN] secret_item or target_item not found in placed objects (or unplaced by Unity), using generic script")
        # Fallback to simple walk action
        comm.add_character("Chars/Female1")
        c_name = container_node["class_name"]
        c_id = container_node["id"]
        script_lines = [
            f"<char0> [Walk] <{c_name}> ({c_id})",
            f"<char0> [TurnRight]",
        ]
    else:
        # Add character - lookup vh_character from scene_info["people"] by actor name
        actor = actions[0].get("actor", "Alice") if actions else "Alice"
        char_model = None

        # Try to find vh_character from people definition in scenario JSON
        people = scene_info.get("people", {})
        for person_key, person_info in people.items():
            if person_info.get("name", "").lower() == actor.lower():
                char_model = person_info.get("vh_character")
                break

        # Fallback to default if not found
        if not char_model:
            char_model = "Chars/Female1" if "female" in actor.lower() or any(
                name in actor.lower() for name in ["alice", "mom", "dr. evans", "lily", "eva"]
            ) else "Chars/Male1"

        room = scene_info.get("room", "")
        if room:
            comm.add_character(char_model, initial_room=room)
        else:
            comm.add_character(char_model)

        # Build VirtualHome script based on action_description
        c_name = container_node["class_name"]
        c_id = container_node["id"]
        s_name = secret_node["class_name"]
        s_id = secret_node["id"]
        t_name = target_node["class_name"]
        t_id = target_node["id"]
        
        action_desc = actions[0].get("description", "") if actions else ""
        
        print(f"    simulating action: {actor} - {action_desc[:60]}...")

        # We must split the script into two separate physical renders so that Unity updates 
        # the box's actual physical collider position before attempting to place the book on it.
        walk_script = [f"<char0> [Walk] <{s_name}> ({s_id})"]
        script_lines_1 = [
            # Walk to the secret item's current surface and grab it
            f"<char0> [Grab] <{s_name}> ({s_id})",
            # Walk to the target container and place the item on it
            f"<char0> [Walk] <{c_name}> ({c_id})",
            f"<char0> [Put] <{s_name}> ({s_id}) <{c_name}> ({c_id})",
        ]
        
        script_lines_2 = [
            # Walk to target item, grab it, and put it on the secret item
            # f"<char0> [Walk] <{t_name}> ({t_id})",
            f"<char0> [Grab] <{t_name}> ({t_id})",
            f"<char0> [Walk] <{s_name}> ({s_id})",
            f"<char0> [Put] <{t_name}> ({t_id}) <{s_name}> ({s_id})",
        ]

    needs_second_part = scene_info.get("needs_second_part", True)

    print("    script part 1:", script_lines_1)
    if needs_second_part:
        print("    script part 2:", script_lines_2)

    success, message = comm.render_script(
        walk_script,
        output_folder=output_dir,
        recording=False,
        find_solution=False,
        skip_animation=True
    )
    if not success:
        print(f"    [WARN] render_script walk failed: {message}")
        return False, None

    # Render Part 1
    success, message = comm.render_script(
        script_lines_1,
        output_folder=output_dir,
        file_name_prefix=f"{prefix}_part1",
        recording=True,
        find_solution=False,  # Use graph IDs (not name matching) to find objects
        image_synthesis=["normal"],
        camera_mode=["AUTO"],
        image_width=image_width,
        image_height=image_height,
        frame_rate=frame_rate,
    )

    if not success:
        print(f"    [WARN] render_script part 1 failed: {message}")
        return False, None

    success = True
    if needs_second_part:
        # Render Part 2
        success, message = comm.render_script(
            script_lines_2,
            output_folder=output_dir,
            file_name_prefix=f"{prefix}_part2",
            recording=True,
            find_solution=False,
            image_synthesis=["normal"],
            camera_mode=["AUTO"],
            image_width=image_width,
            image_height=image_height,
            frame_rate=frame_rate,
        )

        if not success:
            print(f"    [WARN] render_script part 2 failed: {message}")
            return False, None

    # --- collect and concatenate generated images into a single video --------
    import cv2
    from glob import glob

    def get_image_paths(part_name):
        # Unity might save in part_name/0/*.png or part_name/*.png
        d1 = os.path.join(output_dir, part_name, "0")
        d2 = os.path.join(output_dir, part_name)
        
        # Look for normal_*.png as used in camera.py (though render_script might use other names)
        # render_script with image_synthesis=["normal"] usually saves as normal_000.png, etc.
        # But we'll just grab all .png files sorted by name.
        imgs = sorted(glob(os.path.join(d1, "*.png")))
        if not imgs:
            imgs = sorted(glob(os.path.join(d2, "*.png")))
        return imgs

    images_part1 = get_image_paths(f"{prefix}_part1")
    images_part2 = []
    if needs_second_part:
        images_part2 = get_image_paths(f"{prefix}_part2")

    all_image_paths = images_part1 + images_part2
    stitched_video_path = os.path.join(output_dir, f"{prefix}.mp4")

    if not all_image_paths:
        print(f"    [WARN] action ran but no generated images found in {output_dir}")
        return True, None

    print(f"    stitching {len(all_image_paths)} images into video...")
    
    # Read first image to get dimensions
    first_img = cv2.imread(all_image_paths[0])
    if first_img is None:
        print(f"    [ERROR] could not read first image: {all_image_paths[0]}")
        return True, None
        
    h, w = image_height, image_width
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(stitched_video_path, fourcc, frame_rate, (w, h))

    for img_path in all_image_paths:
        img = cv2.imread(img_path)
        if img is not None:
            out.write(img)
            
    out.release()
    print(f"    ✓ action video: {stitched_video_path}")

    for part_name in [f"{prefix}_part1", f"{prefix}_part2"]:
        part_dir = os.path.join(output_dir, part_name)
        if os.path.isdir(part_dir):
            shutil.rmtree(part_dir)

    return True, stitched_video_path



# ============================================================================
# Camera / rendering
# ============================================================================

def render_scene(
    comm,
    container_node: dict,
    output_dir: str,
    prefix: str = "scene",
    orbit: bool = True,
    orbit_cameras: int = 12,
    orbit_radius: float = 2.0,
    orbit_height: float = 1.0,
    image_width: int = 1920,
    image_height: int = 1080,
    ssaa: int = DEFAULT_SSAA,
) -> Tuple[List[str], Optional[str]]:
    """
    Capture images and (optionally) orbit video around the container.

    Returns (image_paths, video_path).
    """
    os.makedirs(output_dir, exist_ok=True)

    # refresh graph to get bounding boxes
    ok, graph = comm.environment_graph()
    if ok:
        for node in graph["nodes"]:
            if node["id"] == container_node["id"]:
                container_node = node
                break

    if orbit:
        images, paths = capture_orbit_images(
            comm,
            target_node=container_node,
            output_dir=output_dir,
            prefix=prefix,
            num_cameras=orbit_cameras,
            radius=orbit_radius,
            height=orbit_height,
            image_width=image_width,
            image_height=image_height,
            ssaa=ssaa,
        )

        return paths
    return []


# ============================================================================
# Process a single scenario
# ============================================================================

def process_scenario(
    comm,
    scenario: dict,
    idx: int,
    tier: str,
    output_dir: str,
    env: int = 0,
    orbit: bool = True,
    orbit_cameras: int = 12,
    orbit_radius: float = 2.0,
    orbit_height: float = 1.0,
    run_actions: bool = True,
    ssaa: int = DEFAULT_SSAA,
    prompts_only: bool = False,
    dialogue_audio_source_dir: str = "",
    variant_index: int = 0,
    variant_total: int = 1,
    include_audio: bool = True,
) -> dict:
    """
    Full pipeline for one scenario: reset → setup → actions → render.

    Returns a result dict with per-stage status for debugging.
    """
    result = {
        "index": idx,
        "name": None,
        "success": False,
        "stages": {
            "extract":  {"ok": False, "error": None},
            "reset":    {"ok": False, "error": None},
            "setup":    {"ok": False, "error": None},
            "actions":  {"ok": False, "skipped": False, "error": None},
            "render":   {"ok": False, "error": None},
            "metadata": {"ok": False, "error": None},
        },
        "error": None,
        "traceback": None,
        "placed_count": 0,
        "image_count": 0,
        "action_video_path": None,
    }

    try:
        # 0. extract scene info
        scene_info = extract_scene_info(scenario, tier=tier)
        name = scene_info["scenario_name"]
        result["name"] = name
        result["stages"]["extract"]["ok"] = True
    except Exception as e:
        result["stages"]["extract"]["error"] = str(e)
        result["error"] = f"extract failed: {e}"
        result["traceback"] = traceback.format_exc()
        return result

    var_suffix = f"_var{variant_index:03d}" if variant_total > 1 else ""

    print(f"\n{'='*60}")
    print(f"[{tier}] Scene {idx}: {name}" + (f"  (variant {variant_index}/{variant_total})" if variant_total > 1 else ""))
    print(f"{'='*60}")

    scene_dir = os.path.join(output_dir, f"scene_{idx:03d}{var_suffix}")

    image_paths = []
    action_video_path = ""
    if prompts_only:
        metadata = load_metadata_json(scene_dir)
        audio_paths = _prepare_tier3_dialogue_audio_files(
            dialogue_audio_source_dir, idx, scene_dir
        )
        if metadata:
            image_paths = metadata.get("image_paths", [])
            action_video_path = metadata.get("action_video_path", "")
            print(f"  Loaded {len(image_paths)} image paths from metadata.json")
        else:
            print(f"  Warning: metadata.json not found in {output_dir}, using default paths")
        
        save_tier3_question_json(
            scenario, scene_dir, image_paths, action_video_path,
            audio_paths=audio_paths,
            include_audio=include_audio,
        )
        result["success"] = True
        return result

    try:
        # 1. reset env
        env_to_use = scene_info.get("env_id", env) if env == -1 else env
        print(f"  [INFO] Setting up in Env {env_to_use}")
        
        if env_to_use < 50:
            ok = comm.reset(env_to_use)
        else:
            response = comm.post_command({'action': 'load_scene', 'intParams': [env_to_use]})
            time.sleep(2)
            response = comm.post_command({'action': 'environment', 'intParams': [env_to_use]})
            ok = response['success']
        if not ok:
            result["stages"]["reset"]["error"] = f"comm.reset({env_to_use}) returned False"
            result["error"] = "reset failed"
            print(f"  [ERROR] cannot reset env {env_to_use}")
            return result
        result["stages"]["reset"]["ok"] = True
    except Exception as e:
        result["stages"]["reset"]["error"] = str(e)
        result["error"] = f"reset exception: {e}"
        result["traceback"] = traceback.format_exc()
        return result

    try:
        # 2. setup objects
        ok, placed, container_node, dest_node = setup_scene(comm, scene_info)
        if container_node is None:
            err_detail = (f"container '{scene_info['container']['vh_name']}' "
                          f"(vh_type: {scene_info['container']['vh_type']}) not found")
            result["stages"]["setup"]["error"] = err_detail
            result["error"] = f"setup failed: {err_detail}"
            return result
        result["stages"]["setup"]["ok"] = True
        result["placed_count"] = len(placed)
    except Exception as e:
        result["stages"]["setup"]["error"] = str(e)
        result["error"] = f"setup exception: {e}"
        result["traceback"] = traceback.format_exc()
        return result

    # 2.5. render orbit around secret item BEFORE actions
    scene_dir = os.path.join(output_dir, f"scene_{idx:03d}{var_suffix}")
    secret_item_node = None
    secret_item_surface = None
    for node, surface, label in placed:
        if "secret:" in label:
            secret_item_node = node
            secret_item_surface = surface
            break

    # 3. character actions
    action_video_path = None
    if run_actions:
        try:
            action_ok, action_video_path = execute_physical_actions(
                comm, scene_info, container_node,
                placed,
                output_dir=scene_dir,
                prefix="action_video",
            )
            result["stages"]["actions"]["ok"] = action_ok
            if not action_ok:
                result["stages"]["actions"]["error"] = "execute_physical_actions returned False"
            result["action_video_path"] = action_video_path
        except Exception as e:
            result["stages"]["actions"]["error"] = str(e)
            result["stages"]["actions"]["traceback"] = traceback.format_exc()
    else:
        result["stages"]["actions"]["skipped"] = True
        result["stages"]["actions"]["ok"] = True
    if action_video_path:
        action_video_path = os.path.relpath(action_video_path, scene_dir).replace("\\", "/")

    # 3.5. hide character from scene before rendering
    # NOTE: We deactivate the character's GameObject instead of moving or
    # removing it, because move_character rejects positions outside room
    # bounds, and expand_scene resets object positions.
    try:
        ok_deact = comm.deactivate_character(0)
        if ok_deact:
            print(f"  [INFO] Deactivated character before rendering")
        else:
            print(f"  [WARN] deactivate_character returned False")
    except Exception as e:
        print(f"  [WARN] failed to hide character from scene: {e}")

    target_item_node = None
    target_image_paths = []
    os.makedirs(os.path.join(scene_dir, "images"), exist_ok=True)
    for node, surface, label in placed:
        if "target:" in label:
            target_item_node = node
            break
    if target_item_node is not None:
        try:
            print(f"  [INFO] Capturing orbit around target item: {target_item_node['class_name']}")
            # Render orbit around target item
            target_image_paths = render_scene(
                comm,
                container_node=target_item_node,
                output_dir=os.path.join(scene_dir, "images"),
                prefix="target",
                orbit=True,
                orbit_cameras=orbit_cameras,
                orbit_radius=orbit_radius,
                orbit_height=orbit_height,
                ssaa=ssaa,
            )
        except Exception as e:
            print(f"  [WARN] target item orbit capture failed: {e}")

    rel_image_paths = [os.path.relpath(path, scene_dir).replace("\\", "/") for path in target_image_paths]
    audio_paths = _prepare_tier3_dialogue_audio_files(
        dialogue_audio_source_dir, idx, scene_dir
    )

    # 5. save metadata and question.json
    try:
        save_tier3_question_json(
            scenario, scene_dir, rel_image_paths, action_video_path,
            audio_paths=audio_paths,
            include_audio=include_audio,
        )

        meta = {
            "scene_index": idx,
            "scenario_name": name,
            "tier": tier,
            "container": scene_info["container"],
            "secret_item": scene_info["secret_item"],
            "target_item": scene_info["target_item"],
            "other_items": scene_info["other_items"],
            "destination": scene_info["destination"],
            "placement_type": scene_info.get("placement_type"),
            "placed_count": len(placed),
            "image_paths": rel_image_paths,
            "action_video_path": action_video_path,
            "audio_paths": audio_paths,
        }
        os.makedirs(scene_dir, exist_ok=True)
        with open(os.path.join(scene_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        result["stages"]["metadata"]["ok"] = True
    except Exception as e:
        result["stages"]["metadata"]["error"] = str(e)

    all_critical_ok = all(
        result["stages"][s]["ok"] for s in ("extract", "reset", "setup", "render")
    )
    result["success"] = all_critical_ok

    return result


# ============================================================================
# Batch processing
# ============================================================================

def batch_process(
    comm,
    json_file: str,
    tier: str,
    env: int = 0,
    output_dir: str = "output",
    start_index: int = 0,
    end_index: Optional[int] = None,
    variant: int = 1,
    **kwargs,
) -> List[dict]:
    """Process all (or a range of) scenarios from a JSON file."""
    with open(json_file, "r", encoding="utf-8") as f:
        scenarios = json.load(f)

    total = len(scenarios)
    if end_index is None:
        end_index = total
    end_index = min(end_index, total)

    print(f"\nBatch: {json_file}  ({total} scenarios, processing {start_index}..{end_index-1})")
    if variant > 1:
        print(f"Variants per scene: {variant}")
    print(f"Output: {output_dir}\n")

    results = []
    for idx in range(start_index, end_index):
        scenario = scenarios[idx]

        for vi in range(variant):
            res = process_scenario(comm, scenario, idx, tier=tier,
                                   output_dir=output_dir, env=env,
                                   variant_index=vi, variant_total=variant,
                                   **kwargs)
            results.append(res)

    # summary
    ok_count = sum(1 for r in results if r["success"])
    fail_count = len(results) - ok_count
    print(f"\n{'='*60}")
    print(f"Batch done: {ok_count}/{len(results)} succeeded, {fail_count} failed")
    print(f"{'='*60}")

    if fail_count > 0:
        print("\nFailed scenarios:")
        for r in results:
            if not r["success"]:
                print(f"  [{r['index']}] {r['name']}: {r.get('error', 'unknown')}")

    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Render tier‑3 scenarios as VirtualHome 3D scenes"
    )
    parser.add_argument("--tier3-file", type=str, default=None,
                        help="Path to tier_3.json")
    parser.add_argument("--env", type=int, default=-1,
                        help="VirtualHome environment ID override (-1 to auto-select based on scenario name)")
    parser.add_argument("--port", type=str, default="8080",
                        help="Unity simulator port")
    parser.add_argument("--exec", type=str, default=None,
                        help="Path to Unity executable")
    parser.add_argument("--output", type=str, default="output/tier3",
                        help="Output root directory")

    # mode selection
    parser.add_argument("--batch", action="store_true", help="Run batch process")
    parser.add_argument("--scene-index", type=int, default=0, help="Index of the single scene to process")

    # batch range
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)

    # orbit cameras
    parser.add_argument("--orbit", action="store_true",
                        help="Use orbit cameras around container")
    parser.add_argument("--orbit_cameras", type=int, default=2)
    parser.add_argument("--orbit_radius", type=float, default=1.5)
    parser.add_argument("--orbit_height", type=float, default=0.8)

    parser.add_argument("--no-actions", action="store_true",
                        help="Skip character physical actions")

    # super-sampling anti-aliasing
    parser.add_argument("--ssaa", type=int, default=2,
                        help="SSAA factor: render at N× resolution then downsample "
                             "(1=off, 2=default, 4=highest quality)")

    # validation only (no Unity)
    parser.add_argument("--prompts-only", action="store_true",
                        help="Skip Unity execution and only save prompts")
    parser.add_argument(
        "--audio-dir",
        type=str,
        default="mllm_privacy/assets/audio/tier3",
        help="Directory containing generated tier3 dialogue wav files "
             "(e.g. .../scene_000/dialogue_00.wav).",
    )
    parser.add_argument("--variant", type=int, default=1,
                        help="Number of variants to generate per scene "
                             "(1=no suffix, >1 appends _var000, _var001, ...)")
    parser.add_argument("--no-audio", action="store_true",
                        help="Do not attach audio files; inline dialogue text as "
                             "natural language in the prompt instead")

    args = parser.parse_args()

    if not args.tier3_file:
        parser.error("--tier3-file required")

    # connect
    if not args.prompts_only:
        print("=" * 60)
        print("Connecting to Unity Simulator …")
        print("=" * 60)
        if args.exec:
            comm = UnityCommunication(port=args.port, file_name=args.exec)
        else:
            comm = UnityCommunication(port=args.port)
    else:
        comm = None

    render_kwargs = dict(
        orbit=args.orbit,
        orbit_cameras=args.orbit_cameras,
        orbit_radius=args.orbit_radius,
        orbit_height=args.orbit_height,
        run_actions=not args.no_actions,
        ssaa=args.ssaa,
        prompts_only=args.prompts_only,
        dialogue_audio_source_dir=args.audio_dir,
        include_audio=not args.no_audio,
    )

    try:
        if args.batch:
            batch_process(
                comm, args.tier3_file, tier="3", env=args.env,
                output_dir=args.output,
                start_index=args.start_index, end_index=args.end_index,
                variant=args.variant,
                **render_kwargs,
            )
        else:
            with open(args.tier3_file, "r", encoding="utf-8") as f:
                scenarios = json.load(f)
            print(f"Loaded {len(scenarios)} scenarios from {args.tier3_file}")

            if 0 <= args.scene_index < len(scenarios):
                for vi in range(args.variant):
                    process_scenario(
                        comm, scenarios[args.scene_index], args.scene_index, tier="3",
                        output_dir=args.output, env=args.env,
                        variant_index=vi, variant_total=args.variant,
                        **render_kwargs,
                    )
            else:
                print(f"Error: scene_index {args.scene_index} out of bounds for {args.tier3_file}")
    finally:
        if comm is not None:
            comm.close()


if __name__ == "__main__":
    main()
