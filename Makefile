# DadAI Data Pipeline
# Run from project root: make all
#
# Prerequisites:
#   - Activate venv: source .venv/bin/activate
#   - Set up .env with Reddit API credentials

.PHONY: all collect format clean check sample help

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
