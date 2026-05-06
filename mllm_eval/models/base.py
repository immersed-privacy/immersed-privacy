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
base.py — Abstract base class for all model backends.

Every backend implements ``generate()`` for single-turn and
``generate_multiturn()`` for multi-turn conversations. Both return
``(response_text, usage_dict)``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseModelBackend(ABC):
    """Abstract interface for model backends."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.model_name: str = config.get("model_name", "")
        self.max_tokens: int = config.get("max_tokens", 1024)
        self.temperature: float = config.get("temperature", 0.0)
        self.thinking_config: dict[str, Any] = config.get("thinking", {})

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        images: list[bytes] | None = None,
        audio: list[bytes] | None = None,
        video: bytes | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Single-turn generation. Returns (response_text, usage_dict)."""
        ...

    @abstractmethod
    async def generate_multiturn(
        self,
        conversation: list[dict[str, Any]],
        images: list[bytes] | None = None,
        audio: list[bytes] | None = None,
        video: bytes | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """
        Continue a multi-turn conversation.

        Args:
            conversation: Prior messages in OpenAI format. The last entry
                is the new user turn whose media is provided separately.
            images: Images for the current (last) turn.
            audio: Audio for the current turn.
            video: Video for the current turn.

        Returns:
            (response_text, usage_dict)
        """
        ...

    def supports_images(self) -> bool:
        return True

    def supports_audio(self) -> bool:
        return False

    def supports_video(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name!r})"
