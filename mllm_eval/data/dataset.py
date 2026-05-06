"""
dataset.py — Case discovery and question.json loading.

Walks the assets directory to find case directories, parses question.json
files, and returns a list of EvalCase objects with optional question type
filtering.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from mllm_eval.data.schema import EvalCase, QuestionSpec, TurnSpec, get_base_tier

logger = logging.getLogger(__name__)


class PrivacyEvalDataset:
    """
    Dataset loader for the mllm_privacy evaluation benchmark.

    Discovers case directories under ``data_root/<tier>/``, loads each
    ``question.json``, and optionally filters question types per tier group.

    Args:
        data_root: Root directory containing tier subdirectories.
        tiers: Tier names or prefixes to load. ``"tier1"`` matches all
            ``tier1_*`` directories. If None, loads all discovered tiers.
        question_types: Maps base tier name to list of allowed question types.
            E.g. ``{"tier1": ["tier1_list"], "tier2": ["tier2_selection"]}``.
            If None, all question types are loaded.
    """

    def __init__(
        self,
        data_root: str | Path,
        tiers: list[str] | None = None,
        question_types: dict[str, list[str]] | None = None,
    ):
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root not found: {self.data_root}")
        self.tiers = tiers
        self.question_types = question_types or {}

    def load(self) -> list[EvalCase]:
        """Load all cases from the configured tiers."""
        cases: list[EvalCase] = []
        tier_dirs = self._discover_tier_dirs()

        for tier_dir in tier_dirs:
            tier_name = tier_dir.name
            case_dirs = sorted(
                d for d in tier_dir.iterdir()
                if d.is_dir() and (d / "question.json").exists()
            )
            for case_dir in case_dirs:
                case = self._load_case(tier_name, case_dir)
                if case is not None:
                    cases.append(case)

        logger.info("Loaded %d cases across %d tier dirs", len(cases), len(tier_dirs))
        return cases

    def _discover_tier_dirs(self) -> list[Path]:
        """Discover tier directories matching the configured tiers."""
        if not self.tiers:
            return sorted(
                d for d in self.data_root.iterdir()
                if d.is_dir() and d.name.startswith("tier")
            )

        tier_dirs: list[Path] = []
        for t in self.tiers:
            for d in sorted(self.data_root.iterdir()):
                if not d.is_dir():
                    continue
                if d.name == t:
                    tier_dirs.append(d)
                elif d.name.startswith(t) and len(d.name) > len(t) and d.name[len(t)] == "_":
                    tier_dirs.append(d)
        return sorted(set(tier_dirs))

    def _load_case(self, tier: str, case_dir: Path) -> EvalCase | None:
        """Load a single case directory from its question.json."""
        question_path = case_dir / "question.json"
        try:
            with open(question_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Error reading %s: %s", question_path, e)
            return None

        questions_raw = data.get("questions", {})
        objects = data.get("objects", [])
        base_tier = get_base_tier(tier)
        allowed_types = self.question_types.get(base_tier)

        questions: list[QuestionSpec] = []
        for q_type, q_data in questions_raw.items():
            if allowed_types and q_type not in allowed_types:
                continue

            turns = []
            for turn_data in q_data.get("turns", []):
                turns.append(TurnSpec(
                    turn_id=turn_data["turn_id"],
                    prompt=turn_data["prompt"],
                    images=turn_data.get("images", []),
                    audio=turn_data.get("audio", []),
                    video=turn_data.get("video"),
                ))

            questions.append(QuestionSpec(
                question_id=q_data["question_id"],
                type=q_type,
                turns=turns,
                answer=q_data.get("answer"),
                options=q_data.get("options"),
            ))

        if not questions:
            return None

        metadata = {k: v for k, v in data.items() if k not in ("questions", "objects")}

        return EvalCase(
            tier=tier,
            case_id=case_dir.name,
            case_dir=case_dir,
            questions=questions,
            objects=objects,
            metadata=metadata,
        )
