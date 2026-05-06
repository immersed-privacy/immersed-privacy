"""
batch.py — Async batch inference with concurrency control.

Processes questions concurrently using ``asyncio.Semaphore``. Multi-turn
questions hold the semaphore for all turns.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any

from mllm_eval.data.dataset import PrivacyEvalDataset
from mllm_eval.data.schema import EvalCase, QuestionSpec
from mllm_eval.inference.prompt_builder import PromptBuilder, QuestionPayload, TurnPayload
from mllm_eval.inference.runner import _build_content_log, _accumulate_usage
from mllm_eval.models.base import BaseModelBackend
from mllm_eval.models.registry import get_backend

logger = logging.getLogger(__name__)

DEFAULT_CONCURRENCY = {
    "vllm": 16,
    "vllm_local": 16,
    "google": 8,
    "openai": 8,
    "anthropic": 4,
}

MAX_RETRIES = 5
BASE_BACKOFF_SECONDS = 1.0


class BatchInferenceRunner:
    """Async batch inference runner with concurrency control."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

        run_name = config.get("run_name", datetime.now().strftime("%Y%m%d_%H%M%S"))
        output_dir = Path(config.get("output_dir", "results")) / run_name
        self.output_dir = output_dir

        model_config = config.get("model", {})
        backend_type = model_config.get("backend_type", "vllm")
        self.backend: BaseModelBackend = get_backend(backend_type, model_config)

        eval_config = config.get("eval", {})
        self.max_concurrent: int = eval_config.get(
            "max_concurrent",
            DEFAULT_CONCURRENCY.get(backend_type, 8),
        )

        data_root = eval_config.get("data_root", "mllm_eval/assets")
        tiers = eval_config.get("tiers", None)
        question_types = eval_config.get("question_types", None)
        self.dataset = PrivacyEvalDataset(
            data_root=data_root, tiers=tiers, question_types=question_types,
        )

        logger.info(
            "Initialized batch runner: backend=%s, concurrency=%d",
            self.backend, self.max_concurrent,
        )

    async def run(self) -> Path:
        """Run batch inference and return predictions path."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = self.output_dir / "predictions.jsonl"

        cases = self.dataset.load()

        tasks: list[tuple[EvalCase, QuestionSpec]] = []
        for case in cases:
            for question in case.questions:
                tasks.append((case, question))

        logger.info(
            "Running batch inference: %d questions, concurrency=%d",
            len(tasks), self.max_concurrent,
        )

        semaphore = asyncio.Semaphore(self.max_concurrent)
        write_lock = asyncio.Lock()

        with open(predictions_path, "w", encoding="utf-8") as f:
            async def _run_and_write(case: EvalCase, question: QuestionSpec) -> None:
                result = await self._process_one(semaphore, case, question)
                async with write_lock:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()

            await asyncio.gather(
                *[_run_and_write(case, question) for case, question in tasks]
            )

        logger.info("Batch predictions saved to %s", predictions_path)
        return predictions_path

    async def _process_one(
        self,
        semaphore: asyncio.Semaphore,
        case: EvalCase,
        question: QuestionSpec,
    ) -> dict[str, Any]:
        """Process a single question with concurrency limiting."""
        async with semaphore:
            payload = PromptBuilder.build_question(case, question, self.backend)
            return await self._run_question_with_retry(case, question, payload)

    async def _run_question_with_retry(
        self,
        case: EvalCase,
        question: QuestionSpec,
        payload: QuestionPayload,
    ) -> dict[str, Any]:
        """Run question inference with retry for single-turn, direct for multi-turn."""
        conversation_log: list[dict[str, Any]] = []
        total_usage: dict[str, Any] = {}
        final_prediction = ""
        num_turns = 0

        for i, turn_payload in enumerate(payload.turns):
            num_turns += 1
            user_content = _build_content_log(turn_payload)
            conversation_log.append({"role": "user", "content": user_content})

            response, usage = await self._generate_turn_with_retry(
                turn_payload, conversation_log if i > 0 else None,
                case, question, num_turns,
            )

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

    async def _generate_turn_with_retry(
        self,
        turn_payload: TurnPayload,
        conversation: list[dict[str, Any]] | None,
        case: EvalCase,
        question: QuestionSpec,
        turn_num: int,
    ) -> tuple[str, dict[str, Any]]:
        """Generate a single turn with exponential backoff retry."""
        for attempt in range(MAX_RETRIES):
            try:
                if conversation is None:
                    return await self.backend.generate(
                        prompt=turn_payload.prompt,
                        images=turn_payload.images or None,
                        audio=turn_payload.audio or None,
                        video=turn_payload.video,
                    )
                else:
                    return await self.backend.generate_multiturn(
                        conversation=conversation,
                        images=turn_payload.images or None,
                        audio=turn_payload.audio or None,
                        video=turn_payload.video,
                    )
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    logger.error(
                        "Failed after %d retries for %s/%s/%s turn %d: %s",
                        MAX_RETRIES, case.tier, case.case_id, question.type, turn_num, e,
                    )
                    return f"ERROR: {e}", {}

                wait = BASE_BACKOFF_SECONDS * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    "Attempt %d/%d failed, retrying in %.1fs: %s",
                    attempt + 1, MAX_RETRIES, wait, e,
                )
                await asyncio.sleep(wait)

        return "ERROR: unexpected retry loop exit", {}


async def run_batch_inference(config: dict[str, Any]) -> Path:
    """Convenience function to run batch inference with a config dict."""
    runner = BatchInferenceRunner(config)
    return await runner.run()
