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
schema.py — Pydantic models for the evaluation data pipeline.

Defines TurnSpec, QuestionSpec, and EvalCase that mirror the question.json
format on disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from pydantic import BaseModel, Field


class TurnSpec(BaseModel):
    """A single turn in a multi-turn question."""
    turn_id: int
    prompt: str
    images: list[str] = Field(default_factory=list)
    audio: list[str] = Field(default_factory=list)
    video: Optional[str] = None


class QuestionSpec(BaseModel):
    """A question with one or more turns, as stored in question.json."""
    question_id: str
    type: str
    turns: list[TurnSpec]
    answer: Optional[Union[int, float, str, list[int], list[str], dict[str, Any]]] = None
    options: Optional[list[str]] = None

    model_config = {"arbitrary_types_allowed": True}


class EvalCase(BaseModel):
    """A complete evaluation case directory."""
    tier: str
    case_id: str
    case_dir: Path
    questions: list[QuestionSpec]
    objects: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


def get_base_tier(tier_name: str) -> str:
    """Extract base tier name from a tier directory name.

    Examples:
        get_base_tier("tier1_5item") -> "tier1"
        get_base_tier("tier2") -> "tier2"
        get_base_tier("tier3") -> "tier3"
    """
    for base in ("tier1", "tier2", "tier3"):
        if tier_name.startswith(base):
            return base
    return tier_name
