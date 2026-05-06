"""
base.py — Abstract metric interface.

All metrics return a dict with at minimum {"score": float}, plus optional
detail keys. This makes every metric extensible without breaking the
aggregator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseMetric(ABC):
    """
    Abstract base class for evaluation metrics.

    Every metric implements ``score()`` which takes the model's prediction
    (always a string) and the ground-truth answer (type varies by tier),
    returning a dict with at minimum ``{"score": float}``.
    """

    @abstractmethod
    def score(self, prediction: str, ground_truth: Any) -> dict[str, Any]:
        """
        Compute the metric score.

        Args:
            prediction: The model's raw text output.
            ground_truth: The expected answer (type depends on tier).

        Returns:
            A dict containing at least ``{"score": float}`` in [0, 1],
            plus any additional detail keys.
        """
        ...

    def score_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """Score using the full prediction record.

        Override this for metrics that need multi-turn conversation data.
        Default implementation delegates to ``score()``.
        """
        return self.score(record.get("prediction", ""), record.get("answer"))

    @property
    def name(self) -> str:
        """Human-readable metric name derived from the class name."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
