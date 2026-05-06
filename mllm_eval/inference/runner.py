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
runner.py — Main inference orchestration loop.

Iterates through all cases and questions, handles multi-turn conversations,
and writes predictions to ``predictions.jsonl``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from mllm_eval.data.dataset import PrivacyEvalDataset
from mllm_eval.data.schema import EvalCase, QuestionSpec
from mllm_eval.inference.prompt_builder import PromptBuilder, TurnPayload, QuestionPayload
from mllm_eval.models.base import BaseModelBackend
from mllm_eval.models.registry import get_backend

logger = logging.getLogger(__name__)


def _build_content_log(turn: TurnPayload) -> list[dict[str, Any]]:
    """Build a serializable content log for a turn (no raw bytes)."""
    content: list[dict[str, Any]] = []
    for _ in turn.images:
        content.append({"type": "image_url"})
    if turn.video is not None:
        content.append({"type": "video_url"})
    for _ in turn.audio:
        content.append({"type": "input_audio"})
    content.append({"type": "text", "text": turn.prompt})
    return content


def _accumulate_usage(total: dict[str, Any], usage: dict[str, Any]) -> None:
    for k, v in usage.items():
        if isinstance(v, (int, float)):
            total[k] = total.get(k, 0) + v


class InferenceRunner:
    """Sequential inference runner with multi-turn support."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

        run_name = config.get("run_name", datetime.now().strftime("%Y%m%d_%H%M%S"))
        output_dir = Path(config.get("output_dir", "results")) / run_name
        self.output_dir = output_dir

        model_config = config.get("model", {})
        backend_type = model_config.get("backend_type", "vllm")
        self.backend: BaseModelBackend = get_backend(backend_type, model_config)

        eval_config = config.get("eval", {})
        data_root = eval_config.get("data_root", "mllm_eval/assets")
        tiers = eval_config.get("tiers", None)
        question_types = eval_config.get("question_types", None)
        self.dataset = PrivacyEvalDataset(
            data_root=data_root, tiers=tiers, question_types=question_types,
        )

        logger.info("Initialized runner: backend=%s, output=%s", self.backend, output_dir)

    async def run(self) -> Path:
        """Run inference and return path to predictions.jsonl."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = self.output_dir / "predictions.jsonl"

        cases = self.dataset.load()
        total_questions = sum(len(c.questions) for c in cases)
        logger.info("Running inference on %d questions across %d cases", total_questions, len(cases))

        processed = 0

        with open(predictions_path, "w", encoding="utf-8") as f:
            for case in cases:
                for question in case.questions:
                    processed += 1
                    logger.info(
                        "Processing [%d/%d] %s/%s/%s",
                        processed, total_questions, case.tier, case.case_id, question.type,
                    )
                    payload = PromptBuilder.build_question(case, question, self.backend)
                    result = await self._run_question(case, question, payload)
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()

        logger.info("Predictions saved to %s", predictions_path)
        return predictions_path

    async def _run_question(
        self,
        case: EvalCase,
        question: QuestionSpec,
        payload: QuestionPayload,
    ) -> dict[str, Any]:
        """Run inference for one question, handling multi-turn logic."""
        conversation_log: list[dict[str, Any]] = []
        total_usage: dict[str, Any] = {}
        final_prediction = ""
        num_turns = 0

        for i, turn_payload in enumerate(payload.turns):
            num_turns += 1

            user_content = _build_content_log(turn_payload)
            conversation_log.append({"role": "user", "content": user_content})

            try:
                if i == 0:
                    response, usage = await self.backend.generate(
                        prompt=turn_payload.prompt,
                        images=turn_payload.images or None,
                        audio=turn_payload.audio or None,
                        video=turn_payload.video,
                    )
                else:
                    response, usage = await self.backend.generate_multiturn(
                        conversation=conversation_log,
                        images=turn_payload.images or None,
                        audio=turn_payload.audio or None,
                        video=turn_payload.video,
                    )
            except Exception as e:
                logger.error(
                    "Failed on %s/%s/%s turn %d: %s",
                    case.tier, case.case_id, question.type, num_turns, e,
                )
                response = f"ERROR: {e}"
                usage = {}

            conversation_log.append({"role": "assistant", "content": response})
            _accumulate_usage(total_usage, usage)
            final_prediction = response

            if question.type == "tier1_list_multiround" and i < len(payload.turns) - 1:
                if "[ANSWER]" in response:
                    break

        return {
            "tier": case.tier,
            "case_id": case.case_id,
            "question_id": question.question_id,
            "type": question.type,
            "question": payload.turns[0].prompt,
            "answer": question.answer,
            "objects": case.objects,
            "prediction": final_prediction,
            "num_turns": num_turns,
            "conversation": conversation_log,
            "has_images": any(len(t.images) > 0 for t in payload.turns[:num_turns]),
            "has_audio": any(len(t.audio) > 0 for t in payload.turns[:num_turns]),
            "has_video": any(t.video is not None for t in payload.turns[:num_turns]),
            "usage": total_usage,
        }


async def run_inference(config: dict[str, Any]) -> Path:
    """Convenience function to run inference with a config dict."""
    runner = InferenceRunner(config)
    return await runner.run()
