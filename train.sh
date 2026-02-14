#!/bin/bash
# DadAI v2 — Training Script
# Run from project root with: ./train.sh
#
# Prerequisites:
# - Virtual environment activated: source .venv/bin/activate
# - Model downloaded: models/mistral-7b-instruct-v0.3-4bit/
# - Training data: data/training_dataset.jsonl

set -e

echo "======================================"
echo "  DadAI v2 — LoRA Fine-Tuning"
echo "  Model: Mistral 7B Instruct v0.3"
echo "  Framework: MLX (Apple Silicon)"
echo "======================================"
echo ""

# Step 1: Prepare training data
echo "[1/3] Preparing training data..."
python scripts/prepare_training_data.py
echo ""

# Step 2: Run LoRA fine-tuning
echo "[2/3] Starting LoRA fine-tuning..."
echo "Estimated time: ~30-90 minutes on M1 16GB."
echo "Peak memory: ~6 GB (your Mac stays usable)."
echo ""

mlx_lm.lora --config training_config.yaml

echo ""
echo "======================================"
echo "  Training complete!"
echo "  Adapters: adapters/dadai-lora/"
echo "======================================"
echo ""

# Step 3: Run evaluation
echo "[3/3] Evaluating on test set..."
mlx_lm.lora \
  --model models/mistral-7b-instruct-v0.3-4bit \
  --adapter-path adapters/dadai-lora \
  --data data/mlx_training \
  --test \
  --test-batches 25

echo ""
echo "======================================"
echo "  All done! Test your model:"
echo "  python scripts/inference.py"
echo "======================================"
