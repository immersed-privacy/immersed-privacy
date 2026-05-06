"""
vllm_backend.py — vLLM OpenAI-compatible backend.

Connects to a locally-served vLLM instance via its OpenAI-compatible
``/v1/chat/completions`` endpoint.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any

from mllm_eval.models.base import BaseModelBackend

logger = logging.getLogger(__name__)


class VLLMBackend(BaseModelBackend):
    """Backend for models served via vLLM with OpenAI-compatible API."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.base_url: str = config.get("base_url", "http://localhost:8000/v1")
        self._supports_audio: bool = config.get("supports_audio", False)
        self._supports_video: bool = config.get("supports_video", False)
        self._enable_thinking: bool | None = self.thinking_config.get("enable_thinking")

        api_key_env = config.get("api_key_env")
        if api_key_env:
            self.api_key: str = os.environ.get(api_key_env, "EMPTY")
        else:
            self.api_key = config.get("api_key", "EMPTY")

        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for the vLLM backend. "
                "Install it with: pip install openai"
            )

        self.client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)

    def _build_user_content(
        self,
        prompt: str,
        images: list[bytes] | None = None,
        audio: list[bytes] | None = None,
        video: bytes | None = None,
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []

        if images:
            for img_bytes in images:
                b64 = base64.b64encode(img_bytes).decode("utf-8")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })

        if video and self._supports_video:
            b64 = base64.b64encode(video).decode("utf-8")
            content.append({
                "type": "video_url",
                "video_url": {"url": f"data:video/mp4;base64,{b64}"},
            })

        if audio and self._supports_audio:
            for audio_bytes in audio:
                b64 = base64.b64encode(audio_bytes).decode("utf-8")
                content.append({
                    "type": "input_audio",
                    "input_audio": {"data": b64, "format": "wav"},
                })

        content.append({"type": "text", "text": prompt})
        return content

    def _build_api_kwargs(self, messages: list[dict]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self._enable_thinking is not None:
            kwargs["extra_body"] = {"enable_thinking": self._enable_thinking}
        return kwargs

    def _extract_usage(self, response: Any) -> dict[str, Any]:
        if response.usage:
            return {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        return {}

    async def generate(
        self,
        prompt: str,
        images: list[bytes] | None = None,
        audio: list[bytes] | None = None,
        video: bytes | None = None,
    ) -> tuple[str, dict[str, Any]]:
        content = self._build_user_content(prompt, images, audio, video)
        messages = [{"role": "user", "content": content}]
        kwargs = self._build_api_kwargs(messages)
        try:
            response = await self.client.chat.completions.create(**kwargs)
            text = (response.choices[0].message.content or "").strip()
            return text, self._extract_usage(response)
        except Exception as e:
            logger.error("vLLM generation failed: %s", e)
            raise

    async def generate_multiturn(
        self,
        conversation: list[dict[str, Any]],
        images: list[bytes] | None = None,
        audio: list[bytes] | None = None,
        video: bytes | None = None,
    ) -> tuple[str, dict[str, Any]]:
        api_messages: list[dict[str, Any]] = []
        for i, msg in enumerate(conversation):
            if msg["role"] == "assistant":
                api_messages.append({"role": "assistant", "content": msg["content"]})
            else:
                is_last = i == len(conversation) - 1
                if is_last:
                    prompt = ""
                    for block in msg["content"]:
                        if block.get("type") == "text":
                            prompt = block["text"]
                            break
                    content = self._build_user_content(prompt, images, audio, video)
                    api_messages.append({"role": "user", "content": content})
                else:
                    content = []
                    for block in msg["content"]:
                        if block.get("type") == "text":
                            content.append({"type": "text", "text": block["text"]})
                    api_messages.append({"role": "user", "content": content})

        kwargs = self._build_api_kwargs(api_messages)
        try:
            response = await self.client.chat.completions.create(**kwargs)
            text = (response.choices[0].message.content or "").strip()
            return text, self._extract_usage(response)
        except Exception as e:
            logger.error("vLLM multiturn generation failed: %s", e)
            raise

    def supports_images(self) -> bool:
        return True

    def supports_audio(self) -> bool:
        return self._supports_audio

    def supports_video(self) -> bool:
        return self._supports_video
