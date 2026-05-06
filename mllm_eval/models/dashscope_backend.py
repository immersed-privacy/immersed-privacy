"""
dashscope_backend.py — Alibaba Cloud DashScope (Bailian) backend.

Uses the OpenAI-compatible endpoint provided by DashScope to call
Qwen series models. Supports multimodal inputs (images, audio, video)
and thinking parameters (enable_thinking, thinking_budget).

Setup:
    pip install openai
    export DASHSCOPE_API_KEY="your-api-key"
"""

from __future__ import annotations

import base64
import logging
from typing import Any

from mllm_eval.models.openai_backend import OpenAIBackend

logger = logging.getLogger(__name__)

DASHSCOPE_BASE_URLS = {
    "cn-beijing": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "us-virginia": "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
    "singapore": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "cn-hongkong": "https://cn-hongkong.dashscope.aliyuncs.com/compatible-mode/v1",
}


class DashScopeBackend(OpenAIBackend):
    """Backend for Alibaba Cloud DashScope (Qwen models).

    Config keys (in addition to OpenAIBackend keys):
        api_key_env (str): Env var for API key (default: DASHSCOPE_API_KEY).
        base_url (str): Override endpoint URL.
        region (str): Region shortcut — "cn-beijing" (default), "singapore",
            "us-virginia", "cn-hongkong". Ignored if base_url is set.
        thinking.enable_thinking (bool): Enable thinking mode.
        thinking.thinking_budget (int): Max thinking tokens.
        split_modalities (bool): Split different modalities into separate
            conversation turns. Required for models like Qwen3-Omni that
            do not support mixed-modality inputs in a single message.
    """

    def __init__(self, config: dict[str, Any]):
        config = dict(config)
        if "api_key_env" not in config:
            config["api_key_env"] = "DASHSCOPE_API_KEY"
        if "base_url" not in config:
            region = config.get("region", "cn-beijing")
            config["base_url"] = DASHSCOPE_BASE_URLS.get(
                region, DASHSCOPE_BASE_URLS["cn-beijing"]
            )

        super().__init__(config)

        self._thinking_budget: int | None = self.thinking_config.get("thinking_budget")
        self._split_modalities: bool = bool(config.get("split_modalities", False))

    @property
    def _needs_streaming(self) -> bool:
        return self._enable_thinking is True

    def _build_api_kwargs(self, messages: list[dict]) -> dict[str, Any]:
        kwargs = super()._build_api_kwargs(messages)

        extra_body: dict[str, Any] = kwargs.get("extra_body", {})
        if self._enable_thinking is not None:
            extra_body["enable_thinking"] = self._enable_thinking
        if self._thinking_budget is not None:
            extra_body["thinking_budget"] = self._thinking_budget
        if extra_body:
            kwargs["extra_body"] = extra_body

        if self._needs_streaming:
            kwargs["stream"] = True
            kwargs["stream_options"] = {"include_usage": True}

        return kwargs

    async def _collect_stream(self, stream) -> tuple[str, dict[str, Any]]:
        """Collect a streaming response into text and usage dict."""
        chunks: list[str] = []
        usage: dict[str, Any] = {}
        async for chunk in stream:
            if chunk.usage:
                usage = {
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                    "total_tokens": chunk.usage.total_tokens,
                }
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    chunks.append(delta.content)
        return "".join(chunks).strip(), usage

    def _build_split_modality_messages(
        self,
        prompt: str,
        images: list[bytes] | None = None,
        audio: list[bytes] | None = None,
        video: bytes | None = None,
    ) -> list[dict[str, Any]]:
        """Split audio/video into separate turns; images stay with the text prompt."""
        ASSISTANT_ACK = {"role": "assistant", "content": "OK"}
        messages: list[dict[str, Any]] = []

        if video:
            b64 = base64.b64encode(video).decode("utf-8")
            messages.append({"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64}"}},
                {"type": "text", "text": "You are an embodied agent. This is your visual observation history. Please use this information to answer the user's question."},
            ]})
            messages.append(ASSISTANT_ACK)

        if audio:
            for audio_bytes in audio:
                b64 = base64.b64encode(audio_bytes).decode("utf-8")
                messages.append({"role": "user", "content": [
                    {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
                    {"type": "text", "text": "You are an embodied agent. This is your audio observation history. Please use this information to answer the user's question."},
                ]})
                messages.append(ASSISTANT_ACK)

        final_content: list[dict[str, Any]] = []
        if images:
            for img_bytes in images:
                b64 = base64.b64encode(img_bytes).decode("utf-8")
                final_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })
        final_content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": final_content})
        return messages

    def _needs_split(
        self,
        images: list[bytes] | None,
        audio: list[bytes] | None,
        video: bytes | None,
    ) -> bool:
        if not self._split_modalities:
            return False
        has_audio_or_video = bool(audio) or bool(video)
        has_other = bool(images) or bool(audio) or bool(video)
        mixed = has_audio_or_video and has_other and (
            (1 if images else 0) + (1 if audio else 0) + (1 if video else 0) > 1
        )
        multi_audio = audio and len(audio) > 1
        return mixed or bool(multi_audio)

    async def _call_api(self, messages: list[dict]) -> tuple[str, dict[str, Any]]:
        kwargs = self._build_api_kwargs(messages)
        if self._needs_streaming:
            stream = await self.client.chat.completions.create(**kwargs)
            return await self._collect_stream(stream)
        response = await self.client.chat.completions.create(**kwargs)
        text = (response.choices[0].message.content or "").strip()
        return text, self._extract_usage(response)

    async def generate(
        self,
        prompt: str,
        images: list[bytes] | None = None,
        audio: list[bytes] | None = None,
        video: bytes | None = None,
    ) -> tuple[str, dict[str, Any]]:
        if self._needs_split(images, audio, video):
            messages = self._build_split_modality_messages(prompt, images, audio, video)
            return await self._call_api(messages)

        if not self._needs_streaming:
            return await super().generate(prompt, images, audio, video)

        content = self._build_user_content(prompt, images, audio, video)
        messages = [{"role": "user", "content": content}]
        return await self._call_api(messages)

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
                    if self._needs_split(images, audio, video):
                        api_messages.extend(
                            self._build_split_modality_messages(prompt, images, audio, video)
                        )
                    else:
                        content = self._build_user_content(prompt, images, audio, video)
                        api_messages.append({"role": "user", "content": content})
                else:
                    content = []
                    for block in msg["content"]:
                        if block.get("type") == "text":
                            content.append({"type": "text", "text": block["text"]})
                    api_messages.append({"role": "user", "content": content})

        return await self._call_api(api_messages)
