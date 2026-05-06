"""
registry.py — Metric registry.

Maps ``(base_tier, type_tag)`` to a list of ``BaseMetric`` instances.
"""

from __future__ import annotations

from mllm_eval.data.schema import get_base_tier
from mllm_eval.metrics.base import BaseMetric
from mllm_eval.metrics.tier1_metrics import Tier1ListMetrics
from mllm_eval.metrics.tier2_metrics import (
    SelectionAccuracy,
    RatingMAE,
    RatingRankTop1,
    RatingSpearman,
)
from mllm_eval.metrics.tier3_metrics import MultiSelectMetrics, MultiroundSelectionMetrics

METRIC_REGISTRY: dict[tuple[str, str], list[BaseMetric]] = {
    ("tier1", "tier1_list"): [Tier1ListMetrics()],
    ("tier1", "tier1_list_multiround"): [Tier1ListMetrics()],
    ("tier2", "tier2_selection"): [SelectionAccuracy()],
    ("tier2", "tier2_rating"): [RatingMAE(), RatingRankTop1(), RatingSpearman()],
    ("tier3", "tier3_selection"): [MultiSelectMetrics()],
    ("tier3", "tier3_selection_multiround"): [MultiroundSelectionMetrics()],
}

METRIC_NAME_MAP: dict[str, type[BaseMetric]] = {
    "tier1_list_metrics": Tier1ListMetrics,
    "selection_accuracy": SelectionAccuracy,
    "rating_mae": RatingMAE,
    "rating_rank_top1": RatingRankTop1,
    "rating_spearman": RatingSpearman,
    "multi_select_metrics": MultiSelectMetrics,
    "multiround_selection_metrics": MultiroundSelectionMetrics,
}


def get_metrics(tier: str, type_tag: str) -> list[BaseMetric]:
    """Get metrics for a (tier, type_tag) pair.

    The tier is normalized via ``get_base_tier()`` so that
    ``"tier1_5item"`` matches the ``"tier1"`` registry entries.
    """
    base = get_base_tier(tier)
    return METRIC_REGISTRY.get((base, type_tag), [])


def get_metrics_by_names(names: list[str]) -> list[BaseMetric]:
    """Get metric instances by their string names."""
    metrics = []
    for name in names:
        if name not in METRIC_NAME_MAP:
            available = ", ".join(sorted(METRIC_NAME_MAP.keys()))
            raise ValueError(f"Unknown metric: {name!r}. Available: {available}")
        metrics.append(METRIC_NAME_MAP[name]())
    return metrics


def register_metric(tier: str, type_tag: str, metric: BaseMetric) -> None:
    """Register a new metric for a (tier, type_tag) pair."""
    key = (tier, type_tag)
    if key not in METRIC_REGISTRY:
        METRIC_REGISTRY[key] = []
    METRIC_REGISTRY[key].append(metric)
