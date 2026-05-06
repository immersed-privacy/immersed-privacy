#!/usr/bin/env bash
# Run open-source model evaluations by tier using in-process vLLM.
# Usage:
#   bash scripts/run_open_source_by_tier.sh tier1
#   bash scripts/run_open_source_by_tier.sh tier2
#   bash scripts/run_open_source_by_tier.sh tier3
#   bash scripts/run_open_source_by_tier.sh all

set -euo pipefail

TARGET_TIER="${1:-all}"
CONFIG_PATH="${CONFIG_PATH:-configs/default.yaml}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-true}"
BATCH="${BATCH:-false}"

run_one_tier() {
  local eval_config="$1"
  shift
  local model_configs=("$@")

  local cmd=(
    python scripts/run_eval.py
    --config "${CONFIG_PATH}"
    --eval-config "${eval_config}"
    --model-configs "${model_configs[@]}"
    --log-level "${LOG_LEVEL}"
  )

  if [ "${CONTINUE_ON_ERROR}" = "true" ]; then
    cmd+=(--continue-on-error)
  fi

  if [ "${BATCH}" = "true" ]; then
    cmd+=(--batch)
  fi

  "${cmd[@]}"
}

TIER1_MODELS=(
  "configs/models/qwen3_5_122b_a10b.yaml"
  "configs/models/qwen3_5_35b_a3b.yaml"
  "configs/models/qwen3_5_9b.yaml"
  "configs/models/gemma_3n_e4b.yaml"
  "configs/models/internvl3_5_gpt_oss_20b_a4b_preview.yaml"
  "configs/models/phi_4_multimodal_instruct.yaml"
)

TIER2_MODELS=(
  "configs/models/qwen3_5_122b_a10b.yaml"
  "configs/models/qwen3_5_35b_a3b.yaml"
  "configs/models/qwen3_5_9b.yaml"
  "configs/models/gemma_3n_e4b.yaml"
  "configs/models/internvl3_5_gpt_oss_20b_a4b_preview.yaml"
  "configs/models/phi_4_multimodal_instruct.yaml"
)

TIER3_MODELS=(
  "configs/models/qwen3_5_122b_a10b.yaml"
  "configs/models/qwen3_5_35b_a3b.yaml"
  "configs/models/qwen3_5_9b.yaml"
  "configs/models/gemma_3n_e4b.yaml"
  "configs/models/llava_next_video_7b_32k_hf.yaml"
  "configs/models/internvl3_5_gpt_oss_20b_a4b_preview.yaml"
)

case "${TARGET_TIER}" in
  tier1)
    run_one_tier "configs/eval/tier1.yaml" "${TIER1_MODELS[@]}"
    ;;
  tier2)
    run_one_tier "configs/eval/tier2.yaml" "${TIER2_MODELS[@]}"
    ;;
  tier3)
    run_one_tier "configs/eval/tier3.yaml" "${TIER3_MODELS[@]}"
    ;;
  all)
    run_one_tier "configs/eval/tier1.yaml" "${TIER1_MODELS[@]}"
    run_one_tier "configs/eval/tier2.yaml" "${TIER2_MODELS[@]}"
    run_one_tier "configs/eval/tier3.yaml" "${TIER3_MODELS[@]}"
    ;;
  *)
    echo "Unknown tier: ${TARGET_TIER}"
    echo "Usage: bash scripts/run_open_source_by_tier.sh [tier1|tier2|tier3|all]"
    exit 1
    ;;
esac
