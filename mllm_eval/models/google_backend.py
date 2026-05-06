"""
google_backend.py — Google Gemini API backend.

Uses the ``google-genai`` SDK for multimodal generation including
text, images, audio, and video.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from mllm_eval.models.base import BaseModelBackend

logger = logging.getLogger(__name__)


class GoogleBackend(BaseModelBackend):
    """Backend for Google Gemini models via the google-genai SDK."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.api_key_env: str = config.get("api_key_env", "GOOGLE_API_KEY")

        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError(
                "The 'google-genai' package is required for the Google backend. "
                "Install it with: pip install google-genai"
            )

        api_key = config.get("api_key") or os.environ.get(self.api_key_env)
        if not api_key:
            raise ValueError(
                f"API key not found. Set the {self.api_key_env} environment variable."
            )

        self._genai = genai
        self._types = types
        self.client = genai.Client(api_key=api_key)
        self._thinking_level = self.thinking_config.get("thinking_level")

    def _build_parts(
        self,
        prompt: str,
        images: list[bytes] | None = None,
        audio: list[bytes] | None = None,
        video: bytes | None = None,
    ) -> list[Any]:
        types = self._types
        parts: list[Any] = []

        if images:
            for img_bytes in images:
                parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))

        if video:
            parts.append(types.Part.from_bytes(data=video, mime_type="video/mp4"))

        if audio:
            for audio_bytes in audio:
                parts.append(types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"))

        parts.append(types.Part.from_text(text=prompt))
        return parts

    def _build_gen_config(self) -> Any:
        types = self._types
        kwargs: dict[str, Any] = {
            "max_output_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self._thinking_level:
            kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=self._thinking_level,
            )
        return types.GenerateContentConfig(**kwargs)

    def _extract_usage(self, response: Any) -> dict[str, Any]:
        usage: dict[str, Any] = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            usage["prompt_tokens"] = getattr(um, "prompt_token_count", 0)
            usage["completion_tokens"] = getattr(um, "candidates_token_count", 0)
            usage["total_tokens"] = getattr(um, "total_token_count", 0)
        return usage

    async def generate(
        self,
        prompt: str,
        images: list[bytes] | None = None,
        audio: list[bytes] | None = None,
        video: bytes | None = None,
    ) -> tuple[str, dict[str, Any]]:
        parts = self._build_parts(prompt, images, audio, video)
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=parts,
                config=self._build_gen_config(),
            )
            return response.text.strip(), self._extract_usage(response)
        except Exception as e:
            logger.error("Gemini generation failed: %s", e)
            raise

    async def generate_multiturn(
        self,
        conversation: list[dict[str, Any]],
        images: list[bytes] | None = None,
        audio: list[bytes] | None = None,
        video: bytes | None = None,
    ) -> tuple[str, dict[str, Any]]:
        types = self._types
        contents: list[Any] = []

        for i, msg in enumerate(conversation):
            is_last = i == len(conversation) - 1
            if msg["role"] == "assistant":
                contents.append(types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=msg["content"])],
                ))
            else:
                if is_last:
                    prompt = ""
                    for block in msg["content"]:
                        if block.get("type") == "text":
                            prompt = block["text"]
                            break
                    parts = self._build_parts(prompt, images, audio, video)
                    contents.append(types.Content(role="user", parts=parts))
                else:
                    text_parts = []
                    for block in msg["content"]:
                        if block.get("type") == "text":
                            text_parts.append(types.Part.from_text(text=block["text"]))
                    contents.append(types.Content(role="user", parts=text_parts or [""]))

        try:
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=self._build_gen_config(),
            )
            return response.text.strip(), self._extract_usage(response)
        except Exception as e:
            logger.error("Gemini multiturn generation failed: %s", e)
            raise

    def supports_images(self) -> bool:
        return True

    def supports_audio(self) -> bool:
        return True

    def supports_video(self) -> bool:
        return True
