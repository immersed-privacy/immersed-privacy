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
prompt_builder.py — Multimodal prompt payload assembly.

Reads media files from disk per-turn and prepares payloads for the backend.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from mllm_eval.data.schema import EvalCase, QuestionSpec
from mllm_eval.models.base import BaseModelBackend

logger = logging.getLogger(__name__)


@dataclass
class TurnPayload:
    """Ready-to-send payload for a single conversation turn."""
    prompt: str
    images: list[bytes] = field(default_factory=list)
    audio: list[bytes] = field(default_factory=list)
    video: bytes | None = None


@dataclass
class QuestionPayload:
    """All turn payloads for a question, with resolved media."""
    turns: list[TurnPayload]
    question: QuestionSpec
    case: EvalCase


class PromptBuilder:
    """Assembles multimodal prompt payloads from EvalCase / QuestionSpec."""

    @staticmethod
    def build_question(
        case: EvalCase,
        question: QuestionSpec,
        backend: BaseModelBackend,
    ) -> QuestionPayload:
        """Build payloads for all turns of a question."""
        turn_payloads: list[TurnPayload] = []

        for turn in question.turns:
            images: list[bytes] = []
            if backend.supports_images():
                for img_rel in turn.images:
                    p = case.case_dir / img_rel
                    if p.exists():
                        images.append(p.read_bytes())
                    else:
                        logger.warning("Image not found: %s", p)

            audio: list[bytes] = []
            if backend.supports_audio():
                for aud_rel in turn.audio:
                    p = case.case_dir / aud_rel
                    if p.exists():
                        audio.append(p.read_bytes())
                    else:
                        logger.warning("Audio not found: %s", p)
            elif turn.audio:
                logger.warning(
                    "Case %s/%s has audio but backend %s does not support audio.",
                    case.tier, case.case_id, backend,
                )

            video: bytes | None = None
            if turn.video:
                if backend.supports_video():
                    vp = case.case_dir / turn.video
                    if vp.exists():
                        video = vp.read_bytes()
                    else:
                        logger.warning("Video not found: %s", vp)
                else:
                    logger.warning(
                        "Case %s/%s has video but backend %s does not support video.",
                        case.tier, case.case_id, backend,
                    )

            turn_payloads.append(TurnPayload(
                prompt=turn.prompt,
                images=images,
                audio=audio,
                video=video,
            ))

        return QuestionPayload(
            turns=turn_payloads,
            question=question,
            case=case,
        )
