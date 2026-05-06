"""
vllm_local_backend.py — In-process vLLM backend (no HTTP server).

Runs vLLM directly inside the evaluator process via ``vllm.LLM``.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any

from mllm_eval.models.base import BaseModelBackend

logger = logging.getLogger(__name__)


class VLLMLocalBackend(BaseModelBackend):
    """Backend for direct in-process vLLM inference."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.tensor_parallel_size: int = int(config.get("tensor_parallel_size", 1))
        self.max_model_len: int | None = config.get("max_model_len")
        self.gpu_memory_utilization: float = float(
            config.get("gpu_memory_utilization", 0.90)
        )
        self.trust_remote_code: bool = bool(config.get("trust_remote_code", True))
        self.dtype: str = str(config.get("dtype", "auto"))
        self._supports_audio: bool = bool(config.get("supports_audio", False))
        self._supports_video: bool = bool(config.get("supports_video", False))
        self._enable_thinking: bool | None = self.thinking_config.get("enable_thinking")

        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise ImportError(
                "The 'vllm' package is required for backend_type='vllm_local'. "
                "Install with: pip install vllm"
            ) from exc

        llm_kwargs: dict[str, Any] = {
            "model": self.model_name,
            "tensor_parallel_size": self.tensor_parallel_size,
            "trust_remote_code": self.trust_remote_code,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "dtype": self.dtype,
        }
        if self.max_model_len is not None:
            llm_kwargs["max_model_len"] = int(self.max_model_len)

        self.llm = LLM(**llm_kwargs)
        self.SamplingParams = SamplingParams

    def _build_content(
        self,
        prompt: str,
        images: list[bytes] | None,
        audio: list[bytes] | None,
        video: bytes | None,
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

    def _get_sampling_params(self) -> Any:
        kwargs: dict[str, Any] = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        return self.SamplingParams(**kwargs)

    async def generate(
        self,
        prompt: str,
        images: list[bytes] | None = None,
        audio: list[bytes] | None = None,
        video: bytes | None = None,
    ) -> tuple[str, dict[str, Any]]:
        messages = [
            {"role": "user", "content": self._build_content(prompt, images, audio, video)}
        ]
        sampling_params = self._get_sampling_params()
        chat_kwargs: dict[str, Any] = {}
        if self._enable_thinking is not None:
            chat_kwargs["chat_template_kwargs"] = {"enable_thinking": self._enable_thinking}

        def _run() -> str:
            outputs = self.llm.chat(
                messages=messages,
                sampling_params=sampling_params,
                use_tqdm=False,
                **chat_kwargs,
            )
            if not outputs or not outputs[0].outputs:
                return ""
            return outputs[0].outputs[0].text.strip()

        try:
            text = await asyncio.to_thread(_run)
            return text, {}
        except Exception as e:
            logger.error("Local vLLM generation failed: %s", e)
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
                    content = self._build_content(prompt, images, audio, video)
                    api_messages.append({"role": "user", "content": content})
                else:
                    content = []
                    for block in msg["content"]:
                        if block.get("type") == "text":
                            content.append({"type": "text", "text": block["text"]})
                    api_messages.append({"role": "user", "content": content})

        sampling_params = self._get_sampling_params()
        chat_kwargs: dict[str, Any] = {}
        if self._enable_thinking is not None:
            chat_kwargs["chat_template_kwargs"] = {"enable_thinking": self._enable_thinking}

        def _run() -> str:
            outputs = self.llm.chat(
                messages=api_messages,
                sampling_params=sampling_params,
                use_tqdm=False,
                **chat_kwargs,
            )
            if not outputs or not outputs[0].outputs:
                return ""
            return outputs[0].outputs[0].text.strip()

        try:
            text = await asyncio.to_thread(_run)
            return text, {}
        except Exception as e:
            logger.error("Local vLLM multiturn generation failed: %s", e)
            raise

    def supports_images(self) -> bool:
        return True

    def supports_audio(self) -> bool:
        return self._supports_audio

    def supports_video(self) -> bool:
        return self._supports_video
