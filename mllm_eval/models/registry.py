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
registry.py — Model backend registry.

Maps ``backend_type`` strings (from YAML config) to the corresponding
backend class. Adding a new backend requires only two steps:
    1. Implement ``BaseModelBackend`` in a new file.
    2. Add an entry to ``BACKEND_REGISTRY`` below.
"""

from __future__ import annotations

from typing import Any

from mllm_eval.models.base import BaseModelBackend


# ---------------------------------------------------------------------------
# Registry mapping: backend_type string → backend class
# ---------------------------------------------------------------------------

BACKEND_REGISTRY: dict[str, type[BaseModelBackend]] = {}


def _register_defaults() -> None:
    """Lazy-register all built-in backends to avoid circular imports."""
    if BACKEND_REGISTRY:
        return  # Already populated

    from mllm_eval.models.vllm_backend import VLLMBackend
    from mllm_eval.models.vllm_local_backend import VLLMLocalBackend
    from mllm_eval.models.google_backend import GoogleBackend
    from mllm_eval.models.openai_backend import OpenAIBackend
    from mllm_eval.models.dashscope_backend import DashScopeBackend
    from mllm_eval.models.volcengine_backend import VolcEngineBackend

    BACKEND_REGISTRY.update(
        {
            "vllm": VLLMBackend,
            "vllm_local": VLLMLocalBackend,
            "google": GoogleBackend,
            "openai": OpenAIBackend,
            "dashscope": DashScopeBackend,
            "volcengine": VolcEngineBackend,
        }
    )


def get_backend(backend_type: str, config: dict[str, Any]) -> BaseModelBackend:
    """
    Instantiate a model backend by its type string.

    Args:
        backend_type: Key into the registry (e.g., "vllm", "google").
        config: Backend-specific configuration dictionary.

    Returns:
        An initialized backend instance.

    Raises:
        ValueError: If ``backend_type`` is not in the registry.
    """
    _register_defaults()

    if backend_type not in BACKEND_REGISTRY:
        available = ", ".join(sorted(BACKEND_REGISTRY.keys()))
        raise ValueError(
            f"Unknown backend type: {backend_type!r}. "
            f"Available backends: {available}"
        )

    cls = BACKEND_REGISTRY[backend_type]
    return cls(config)


def register_backend(name: str, cls: type[BaseModelBackend]) -> None:
    """
    Register a custom backend class under the given name.

    This allows third-party extensions without modifying the core registry.

    Args:
        name: Backend type string to register.
        cls: Backend class (must be a subclass of BaseModelBackend).
    """
    if not issubclass(cls, BaseModelBackend):
        raise TypeError(f"{cls} is not a subclass of BaseModelBackend")
    BACKEND_REGISTRY[name] = cls
