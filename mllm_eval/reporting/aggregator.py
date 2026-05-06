"""
aggregator.py — Score aggregation across cases and tiers.

Reads predictions.jsonl, applies tier-specific metrics, and produces
per-case scores plus aggregate statistics. Aggregation uses the actual
tier name (e.g. "tier1_1item"), not the base tier group.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from mllm_eval.data.schema import get_base_tier
from mllm_eval.metrics.registry import get_metrics

logger = logging.getLogger(__name__)


class Aggregator:
    """Aggregates per-case metric scores into tier-level and overall summaries."""

    def score_predictions(self, predictions_path: Path) -> dict[str, Any]:
        """Score all predictions and aggregate results."""
        records = []
        with open(predictions_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        logger.info("Scoring %d predictions", len(records))

        per_case_scores: list[dict[str, Any]] = []
        # tier -> type -> metric_key -> [values]
        agg: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )

        for rec in records:
            tier = rec["tier"]
            type_tag = rec["type"]
            answer = rec.get("answer")
            prediction = rec.get("prediction", "")

            if answer is None:
                per_case_scores.append({
                    "tier": tier,
                    "case_id": rec.get("case_id", ""),
                    "type": type_tag,
                    "scored": False,
                })
                continue

            metrics = get_metrics(tier, type_tag)
            if not metrics:
                logger.debug("No metrics for (%s, %s), skipping", tier, type_tag)
                continue

            record_scores: dict[str, Any] = {
                "tier": tier,
                "case_id": rec.get("case_id", ""),
                "type": type_tag,
                "scored": True,
            }

            for metric in metrics:
                if getattr(metric, "_needs_record", False):
                    result = metric.score_record(rec)
                else:
                    result = metric.score(prediction, answer)
                record_scores[metric.name] = result
                for k, v in result.items():
                    if isinstance(v, (int, float)):
                        agg[tier][type_tag][f"avg_{k}"].append(v)

            num_turns = rec.get("num_turns", 1)
            agg[tier][type_tag]["avg_num_turns"].append(float(num_turns))

            per_case_scores.append(record_scores)

        per_tier = self._aggregate_per_tier(agg)
        per_type = self._aggregate_per_type(agg)
        per_tier_type = self._aggregate_per_tier_type(agg)
        overall = self._aggregate_overall(agg)

        return {
            "per_case": per_case_scores,
            "per_tier": per_tier,
            "per_type": per_type,
            "per_tier_type": per_tier_type,
            "overall": overall,
        }

    def _aggregate_per_tier(
        self, agg: dict[str, dict[str, dict[str, list[float]]]]
    ) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for tier, types in agg.items():
            combined: dict[str, list[float]] = defaultdict(list)
            total_records = 0
            for type_tag, metric_values in types.items():
                for k, v in metric_values.items():
                    combined[k].extend(v)
                sample_key = next(iter(metric_values))
                total_records += len(metric_values[sample_key])

            tier_result: dict[str, Any] = {}
            for k, v in combined.items():
                tier_result[k] = sum(v) / len(v) if v else 0.0
            tier_result["num_records"] = total_records
            result[tier] = tier_result
        return result

    def _aggregate_per_type(
        self, agg: dict[str, dict[str, dict[str, list[float]]]]
    ) -> dict[str, dict[str, Any]]:
        # Merge across all tiers for same type_tag
        type_agg: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for tier, types in agg.items():
            for type_tag, metric_values in types.items():
                for k, v in metric_values.items():
                    type_agg[type_tag][k].extend(v)

        result: dict[str, dict[str, Any]] = {}
        for type_tag, metric_values in type_agg.items():
            type_result: dict[str, Any] = {}
            sample_key = next(iter(metric_values))
            type_result["num_records"] = len(metric_values[sample_key])
            for k, v in metric_values.items():
                type_result[k] = sum(v) / len(v) if v else 0.0
            result[type_tag] = type_result
        return result

    def _aggregate_per_tier_type(
        self, agg: dict[str, dict[str, dict[str, list[float]]]]
    ) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for tier, types in agg.items():
            for type_tag, metric_values in types.items():
                key = f"{tier}/{type_tag}"
                entry: dict[str, Any] = {}
                sample_key = next(iter(metric_values))
                entry["num_records"] = len(metric_values[sample_key])
                for k, v in metric_values.items():
                    entry[k] = sum(v) / len(v) if v else 0.0
                result[key] = entry
        return result

    def _aggregate_overall(
        self, agg: dict[str, dict[str, dict[str, list[float]]]]
    ) -> dict[str, Any]:
        combined: dict[str, list[float]] = defaultdict(list)
        total_records = 0
        for tier, types in agg.items():
            for type_tag, metric_values in types.items():
                for k, v in metric_values.items():
                    combined[k].extend(v)
                sample_key = next(iter(metric_values))
                total_records += len(metric_values[sample_key])

        overall: dict[str, Any] = {}
        for k, v in combined.items():
            overall[k] = sum(v) / len(v) if v else 0.0
        overall["total_records"] = total_records
        return overall
