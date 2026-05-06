#!/usr/bin/env bash
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

# Run evaluation for all model configs using in-process vLLM backend.
# Default: Tier1 + Tier3, all configs under configs/models/*.yaml

set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-configs/default.yaml}"
MODEL_CONFIG_DIR="${MODEL_CONFIG_DIR:-configs/models}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
BATCH="${BATCH:-false}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-true}"

EVAL_CONFIGS=("${@}")
if [ ${#EVAL_CONFIGS[@]} -eq 0 ]; then
  EVAL_CONFIGS=("configs/eval/tier1.yaml" "configs/eval/tier3.yaml")
fi

CMD=(
  python scripts/run_eval.py
  --config "${CONFIG_PATH}"
  --all-model-configs
  --model-config-dir "${MODEL_CONFIG_DIR}"
  --eval-configs "${EVAL_CONFIGS[@]}"
  --log-level "${LOG_LEVEL}"
)

if [ "${BATCH}" = "true" ]; then
  CMD+=(--batch)
fi

if [ "${CONTINUE_ON_ERROR}" = "true" ]; then
  CMD+=(--continue-on-error)
fi

"${CMD[@]}"
