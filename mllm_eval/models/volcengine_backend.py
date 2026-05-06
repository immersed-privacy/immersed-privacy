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
volcengine_backend.py — ByteDance Volcano Engine ARK backend.

Uses the OpenAI-compatible endpoint provided by Volcano Engine to call
Doubao series models. Supports multimodal inputs (images, video).

Setup:
    pip install openai
    export ARK_API_KEY="your-api-key"
"""

from __future__ import annotations

import logging
from typing import Any

from mllm_eval.models.openai_backend import OpenAIBackend

logger = logging.getLogger(__name__)

ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"


class VolcEngineBackend(OpenAIBackend):
    """Backend for ByteDance Volcano Engine ARK (Doubao models).

    Config keys (in addition to OpenAIBackend keys):
        api_key_env (str): Env var for API key (default: ARK_API_KEY).
        base_url (str): Override endpoint URL.
    """

    def __init__(self, config: dict[str, Any]):
        config = dict(config)
        if "api_key_env" not in config:
            config["api_key_env"] = "ARK_API_KEY"
        if "base_url" not in config:
            config["base_url"] = ARK_BASE_URL

        super().__init__(config)
