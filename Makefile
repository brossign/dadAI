# DadAI Data Pipeline
# Run from project root: make all
#
# Prerequisites:
#   - Activate venv: source .venv/bin/activate
#   - Set up .env with Reddit API credentials

.PHONY: all collect format clean check sample prepare train test chat help

# Full pipeline: collect → format → clean → validate
all: collect format clean check

# Step 1: Collect posts from Reddit
collect:
	python scripts/collect_reddit_data.py

# Step 2: Format into Mistral [INST] prompt/completion pairs
format:
	python scripts/format_reddit_data.py

# Step 3: Clean and filter low-quality entries
clean:
	python scripts/clean_dataset.py

# Step 4: Validate the cleaned dataset
check:
	python scripts/check_dataset_format.py

# Show random samples for quality inspection
sample:
	python scripts/show_random_sample.py

# Step 5: Prepare data for MLX training (split into train/valid/test)
prepare:
	python scripts/prepare_training_data.py

# Step 6: Run LoRA fine-tuning
train: prepare
	mlx_lm.lora --config training_config.yaml

# Step 7: Evaluate on test set
test:
	mlx_lm.lora --model models/mistral-7b-instruct-v0.3-4bit --adapter-path adapters/dadai-lora --data data/mlx_training --test --test-batches 25

# Interactive chat with fine-tuned model
chat:
	python scripts/inference.py

# Show help
help:
	@echo "DadAI Data Pipeline"
	@echo ""
	@echo "  make all      - Run full pipeline (collect → format → clean → check)"
	@echo "  make collect   - Step 1: Collect Reddit posts"
	@echo "  make format    - Step 2: Format into training pairs"
	@echo "  make clean     - Step 3: Clean and filter"
	@echo "  make check     - Step 4: Validate dataset"
	@echo "  make sample    - Show random samples"
	@echo ""
	@echo "Training:"
	@echo "  make prepare   - Step 5: Prepare data for MLX training"
	@echo "  make train     - Step 6: Run LoRA fine-tuning (includes prepare)"
	@echo "  make test      - Step 7: Evaluate model on test set"
	@echo "  make chat      - Interactive chat with fine-tuned model"
