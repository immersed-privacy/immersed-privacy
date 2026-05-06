"""
PDDL ↔ VirtualHome name parsing & conversion utilities.

Shared helpers used by all tiers to translate PDDL identifiers
(e.g. ``"book.n.01_1"``) into VirtualHome class names.
"""

import re
from typing import Tuple


def parse_pddl_object_name(pddl_name: str) -> Tuple[str, str]:
    """
    Parse a PDDL instance name such as ``"book.n.01_1"`` into
    ``("book.n.01", "1")``.

    Returns ``(pddl_name, "1")`` when the name does not match the
    expected ``xxx.n.NN_M`` pattern.
    """
    match = re.match(r"^(.+\.n\.\d+)_(\d+)$", pddl_name)
    if match:
        return match.group(1), match.group(2)
    return pddl_name, "1"


def pddl_to_vh_object(pddl_type: str, mapping: dict) -> str:
    """
    Convert a PDDL object type to a VirtualHome class name using *mapping*.

    Falls back to stripping the ``.n.XX`` suffix and removing underscores.
    """
    if pddl_type in mapping:
        return mapping[pddl_type]
    base = pddl_type.split(".")[0]
    return base.replace("_", "")


def pddl_to_vh_container(pddl_type: str, mapping: dict) -> str:
    """
    Convert a PDDL container type to a VirtualHome class name using *mapping*.

    Falls back to stripping the ``.n.XX`` suffix and removing underscores.
    """
    if pddl_type in mapping:
        return mapping[pddl_type]
    base = pddl_type.split(".")[0]
    return base.replace("_", "")
