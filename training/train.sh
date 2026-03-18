#!/bin/bash
# Standard training script with AGX GPU watchdog fix
# Usage: bash training/train.sh [config] [adapter_dir] [resume_checkpoint]
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

CONFIG="${1:-training/lora_config_best.yaml}"
ADAPTER_DIR="${2:-adapters}"
RESUME="${3:-}"
MODEL="mlx-community/Qwen3.5-2B-OptiQ-4bit"

mkdir -p "$ADAPTER_DIR"

RESUME_FLAG=""
if [ -n "$RESUME" ]; then
    RESUME_FLAG="--resume-adapter-file $RESUME"
    echo "Resuming from: $RESUME"
fi

echo "=== Training Start: $(date) ==="
echo "Config: $CONFIG"
echo "Adapter dir: $ADAPTER_DIR"

# AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1 fixes macOS Tahoe GPU watchdog crash
# See: https://github.com/ml-explore/mlx/issues/3267
AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1 python -m mlx_lm lora \
    --model "$MODEL" --train --data ./data \
    --config "$CONFIG" --adapter-path "$ADAPTER_DIR" \
    $RESUME_FLAG

echo "=== Training Complete: $(date) ==="
ls -la "$ADAPTER_DIR"/*.safetensors 2>/dev/null | tail -5
