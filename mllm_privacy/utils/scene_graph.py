"""
Scene-graph utilities shared across tier1 / tier2 / tier3.

Functions for querying and modifying VirtualHome environment graphs:
  - find_tables, get_table_size, can_place_on_table
  - get_max_node_id, find_room_for_object
  - add_object_to_table, add_object_to_container
  - find_container_in_scene
  - get_object_size
"""

from typing import List, Optional

# ============================================================================
# Table / surface classification constants
# ============================================================================

LARGE_TABLES = ["desk", "kitchentable", "diningtable", "table", "studytable", "meetingtablelarge"]
MEDIUM_TABLES = ["coffeetable", "sidetable", "shelf"]
SMALL_TABLES = ["nightstand"]
TABLE_CLASS_NAMES = LARGE_TABLES + MEDIUM_TABLES + SMALL_TABLES
TABLE_EXCLUSIONS = ["tablelamp", "tablecloth", "turntable", "timetable"]

CLOSED_CONTAINERS = ["cabinet", "microwave", "fridge", "oven", "dishwasher"]

PLACEABLE_OBJECTS = [
    "apple", "banana", "book", "cellphone", "chips", "clock",
    "coffeepot", "condimentbottle", "condimentshaker", "crackers",
    "cupcake", "cutleryknife", "dishbowl", "folder", "fork",
    "glass", "keyboard", "laptop", "milk", "mug", "notes",
    "orange", "paper", "peach", "pear", "pencil", "plate",
    "plum", "remotecontrol", "salmon", "spoon", "waterglass",
    "wineglass", "wine", "cereal",
]

LARGE_OBJECTS = ["laptop", "keyboard", "plate", "tray", "book", "folder", "boardgame"]
MEDIUM_OBJECTS = ["coffeepot", "cereal", "milk", "wine", "condimentbottle"]
SMALL_OBJECTS = [
    "apple", "banana", "cellphone", "chips", "clock", "condimentshaker",
    "crackers", "cupcake", "cutleryknife", "dishbowl", "fork", "glass",
    "mug", "notes", "orange", "paper", "peach", "pear", "pencil",
    "plum", "remotecontrol", "salmon", "spoon", "waterglass", "wineglass",
]


# ============================================================================
# Query helpers
# ============================================================================

def get_table_size(class_name: str) -> int:
    """Return a sort key for table size (0 = large, 1 = medium, 2 = small, 3 = unknown)."""
    class_name = class_name.lower()
    if class_name in LARGE_TABLES:
        return 0
    if class_name in MEDIUM_TABLES:
        return 1
    if class_name in SMALL_TABLES:
        return 2
    return 3


def find_tables(graph: dict, sort_by_size: bool = True) -> List[dict]:
    """Find all table nodes in the scene graph, optionally sorted by size (large first)."""
    tables = []
    for node in graph["nodes"]:
        class_name = node["class_name"].lower()
        if class_name in TABLE_EXCLUSIONS:
            continue

        is_table = class_name in TABLE_CLASS_NAMES
        if not is_table:
            props = node.get("properties", [])
            if "SURFACES" in props:
                if class_name.endswith("table") or class_name.endswith("desk"):
                    if class_name not in TABLE_EXCLUSIONS:
                        is_table = True

        if is_table:
            tables.append(node)

    if sort_by_size:
        tables.sort(key=lambda t: get_table_size(t["class_name"]))
    return tables


def get_object_size(obj_name: str) -> str:
    """Return ``"large"``, ``"medium"`` or ``"small"`` for a VH object class name."""
    obj_name = obj_name.lower()
    if obj_name in LARGE_OBJECTS:
        return "large"
    if obj_name in MEDIUM_OBJECTS:
        return "medium"
    return "small"


def can_place_on_table(obj_name: str, table_class_name: str) -> bool:
    """Check whether an object fits on a given table type based on size rules."""
    obj_size = get_object_size(obj_name)
    table_class = table_class_name.lower()

    if table_class in LARGE_TABLES:
        return True
    if table_class in MEDIUM_TABLES:
        return obj_size in ("medium", "small")
    if table_class in SMALL_TABLES:
        return obj_size == "small"
    return True


def find_room_for_object(graph: dict, obj_id: int) -> Optional[int]:
    """Return the room node ID that contains *obj_id*, or ``None``."""
    room_ids = {n["id"] for n in graph["nodes"] if n.get("category") == "Rooms"}
    for edge in graph["edges"]:
        if edge["from_id"] == obj_id and edge["relation_type"] == "INSIDE" and edge["to_id"] in room_ids:
            return edge["to_id"]
    return None


def get_max_node_id(graph: dict) -> int:
    """Return the largest node ID currently present in *graph*."""
    max_id = 0
    for node in graph["nodes"]:
        if node["id"] > max_id:
            max_id = node["id"]
    return max_id


# ============================================================================
# Graph modification helpers
# ============================================================================

def add_object_to_table(graph: dict, obj_class_name: str,
                        table_node: dict, new_id: int) -> dict:
    """Add a new GRABBABLE object ON *table_node* and return the new node dict."""
    new_node = {
        "id": new_id,
        "class_name": obj_class_name,
        "category": "Props",
        "properties": ["GRABBABLE", "MOVABLE"],
        "states": [],
    }
    graph["nodes"].append(new_node)
    graph["edges"].append({
        "from_id": new_id,
        "relation_type": "ON",
        "to_id": table_node["id"],
    })
    return new_node


def add_object_to_container(graph: dict, obj_class_name: str,
                            container_node: dict, new_id: int,
                            use_inside: bool = False) -> dict:
    """Add a new GRABBABLE object ON or INSIDE *container_node*."""
    new_node = {
        "id": new_id,
        "class_name": obj_class_name,
        "category": "Props",
        "properties": ["GRABBABLE", "MOVABLE"],
        "states": [],
    }
    graph["nodes"].append(new_node)
    graph["edges"].append({
        "from_id": new_id,
        "relation_type": "INSIDE" if use_inside else "ON",
        "to_id": container_node["id"],
    })
    return new_node


def find_container_in_scene(graph: dict, vh_container_name: str,
                            exclude_rooms: bool = True) -> Optional[dict]:
    """
    Fuzzy-find a container node by VH class name.

    Tries exact match first, then substring match.  When *exclude_rooms* is
    ``True`` (default), room nodes are skipped.
    """
    vh_lower = vh_container_name.lower()
    room_categories = {"Rooms"}

    for node in graph["nodes"]:
        if exclude_rooms and node.get("category") in room_categories:
            continue
        if node["class_name"].lower() == vh_lower:
            return node

    for node in graph["nodes"]:
        if exclude_rooms and node.get("category") in room_categories:
            continue
        cn = node["class_name"].lower()
        if vh_lower in cn or cn in vh_lower:
            return node

    return None
