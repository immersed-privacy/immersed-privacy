"""
mllm_privacy.utils — shared helpers for VirtualHome scene manipulation.

Sub-modules:
  scene_graph   – graph query & modification (tables, containers, objects)
  pddl_mapping  – PDDL ↔ VirtualHome name conversion
  camera        – orbit cameras, image capture, video stitching
"""

from .scene_graph import (
    LARGE_TABLES,
    MEDIUM_TABLES,
    SMALL_TABLES,
    TABLE_CLASS_NAMES,
    TABLE_EXCLUSIONS,
    CLOSED_CONTAINERS,
    PLACEABLE_OBJECTS,
    LARGE_OBJECTS,
    MEDIUM_OBJECTS,
    SMALL_OBJECTS,
    get_table_size,
    find_tables,
    get_object_size,
    can_place_on_table,
    find_room_for_object,
    get_max_node_id,
    add_object_to_table,
    add_object_to_container,
    find_container_in_scene,
)

from .pddl_mapping import (
    parse_pddl_object_name,
    pddl_to_vh_object,
    pddl_to_vh_container,
)

from .camera import (
    DEFAULT_SSAA,
    get_object_center,
    add_orbit_cameras,
    add_orbit_cameras_with_jitter,
    capture_orbit_images,
    capture_sensitive_orbit_images,
    capture_closeup_images,
    create_orbit_video,
    capture_images,
)

from .prompt_utils import (
    load_metadata_json,
    generate_tier1_prompt_list,
    generate_tier3_prompt,
    generate_tier2_prompt,
)