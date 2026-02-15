# DadAI – An LLM-based assistant for new dads 🤖👶

**DadAI** is an open-source project built to support new fathers during pregnancy and early parenthood.
The idea is simple: provide emotionally intelligent, practical guidance powered by LLMs — fine-tuned on real parenting discussions.

## 🚀 Why DadAI?

Most resources around parenting are either mother-centric or scattered across forums.
As a first-time dad, I realized how hard it can be to find support that's both practical and emotionally relevant — and I wanted to make things easier for other future dads.

DadAI aims to provide a clear, AI-driven interface that supports:
- Emotional support during pregnancy and early parenthood
- Concrete actions and reminders
- Guidance on sleep, communication, and partner well-being

## 📰 Press & Articles

- [How a Solo Dev Built an AI for Dads — RunPod Blog (May 2025)](https://www.runpod.io/blog/solo-dev-ai-for-dads-runpod)
- [How I Fine-Tuned a Custom AI Model (DadAI) — LinkedIn (Apr 2025)](https://www.linkedin.com/pulse/how-i-built-fine-tuned-my-own-dadai-what-taught-me-llms-rossignol-pq7he/)

## 🔄 Project Evolution

### v1 (April 2025) — Cloud-based, RunPod + GPTQ

The original version was built as a learning project using:
- **Mistral 7B Instruct v0.1** (GPTQ quantized)
- **QLoRA + PEFT** fine-tuning on **RunPod** (RTX 4090, ~$5 total)
- **298 Reddit posts** from r/NewDads, r/Daddit, r/BabyBumps, r/Parenting
- HuggingFace Transformers + bitsandbytes stack

**What went wrong:** A code audit revealed critical bugs — the model never actually trained on the completions (tokenization bug), the prompt templates were mismatched between training and inference, and the deployment pipeline (GPTQ → GGUF → LocalAI) hit format incompatibility walls. See the `v0.1-original` tag for the original codebase.

### v2 (February 2026) — Local-first, Mac + MLX 🆕

A complete rewrite, taking advantage of Apple's **MLX framework** which now makes local fine-tuning on Mac viable:

| | v1 (2025) | v2 (2026) |
|--|-----------|-----------|
| **Base model** | Mistral 7B v0.1 (GPTQ) | Mistral 7B Instruct v0.3 (MLX 4-bit) |
| **Training** | RunPod RTX 4090 ($5) | Mac M1 locally (free) |
| **Framework** | HuggingFace + PEFT + bitsandbytes | Apple MLX + mlx-lm |
| **Dataset** | 298 Reddit pairs (buggy pipeline) | 1,000–2,000 curated pairs (fixed) |
| **Deployment** | LocalAI (never worked) | Gradio on Hugging Face Spaces |
| **UI** | None (CLI only) | Chat interface |

## 🧠 Tech Stack (v2)

- **Model:** [Mistral 7B Instruct v0.3 (4-bit MLX)](https://huggingface.co/mlx-community/Mistral-7B-Instruct-v0.3-4bit) — 3.8 GB on disk
- **Training:** LoRA fine-tuning via [mlx-lm](https://github.com/ml-explore/mlx-lm) on Apple Silicon
- **Data:** Hybrid — 2,096 real Reddit Q&A pairs + 68 synthetic pairs for under-covered topics
- **UI:** Gradio chat interface
- **Hosting:** Hugging Face Spaces (free)
- **Language:** Python 3.11

## 📂 Project Structure

```
dadAI/
├── data/                       # Datasets (raw, cleaned, formatted)
│   ├── reddit_dataset.jsonl    #   Raw Reddit posts (2,100)
│   ├── formatted_dataset.jsonl #   Mistral [INST] prompt/completion pairs
│   ├── cleaned_dataset.jsonl   #   Filtered, deduplicated (2,096)
│   ├── synthetic_gap_topics.jsonl # Synthetic pairs for under-covered topics (68)
│   ├── training_dataset.jsonl  #   Final merged dataset (2,164)
│   └── mlx_training/           #   Train/valid/test splits for mlx-lm
├── scripts/                    # Data & training pipeline
│   ├── collect_reddit_data.py  #   Reddit data collection
│   ├── format_reddit_data.py   #   Convert to Mistral chat format
│   ├── clean_dataset.py        #   Quality filtering
│   ├── check_dataset_format.py #   Validation
│   ├── generate_synthetic_data.py # Synthetic data for gap topics
│   ├── prepare_training_data.py   # Convert to mlx-lm chat format + split
│   ├── inference.py            #   Interactive chat with fine-tuned model
│   └── show_random_sample.py   #   Quick dataset inspection
├── adapters/                   # LoRA adapters (generated, not committed)
├── models/                     # Downloaded models (not committed)
├── lora_finetune/              # v1 fine-tuning scripts (archived)
├── training_config.yaml        # MLX LoRA training configuration
├── train.sh                    # One-command training script
├── Makefile                    # Pipeline commands
├── .venv/                      # Python virtual environment (not committed)
├── requirements.txt
├── .env                        # Reddit API credentials (not committed)
├── .gitignore
└── README.md
```

## 🛠️ Setup (v2)

### Prerequisites
- macOS with Apple Silicon (M1/M2/M3)
- [Homebrew](https://brew.sh)
- 16 GB RAM minimum

### Installation

```bash
# Clone the repo
git clone https://github.com/brossign/dadAI.git
cd dadAI

# Install Python 3.11
brew install python@3.11

# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install mlx-lm huggingface_hub gradio praw python-dotenv tqdm

# Download the base model (~3.8 GB)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('mlx-community/Mistral-7B-Instruct-v0.3-4bit', local_dir='models/mistral-7b-instruct-v0.3-4bit')
"
```

### Quick Test

```bash
python -c "
from mlx_lm import load, generate
model, tokenizer = load('models/mistral-7b-instruct-v0.3-4bit')
response = generate(model, tokenizer, prompt='[INST] I just became a dad. Any advice? [/INST]', max_tokens=256)
print(response)
"
```

## 🏋️ Training

```bash
# Activate environment
source .venv/bin/activate

# One-command training (prepare data + fine-tune + evaluate)
./train.sh

# Or step by step:
make prepare    # Convert data to mlx-lm format + split
make train      # Run LoRA fine-tuning (~30-90 min on M1)
make test       # Evaluate on held-out test set
make chat       # Interactive chat with fine-tuned model
```

**Training details:**
- **Method:** QLoRA (model is 4-bit quantized, LoRA rank 16)
- **Key fix from v1:** `--mask-prompt` ensures the model trains only on completions, not prompts
- **Memory:** Peak ~6 GB (comfortable on 16 GB Mac)
- **Dataset:** 2,164 examples (2,096 real Reddit + 68 synthetic gap topics)
- **Config:** See `training_config.yaml` for all hyperparameters

## 💻 Chat UI

```bash
# Start the Gradio chat interface
source .venv/bin/activate
python app.py
# Open http://localhost:7860
```

Features:
- Clean chat interface with example questions
- Runs locally on Apple Silicon
- Model loads in ~5 seconds, responses in 2-10 seconds

## 💬 Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1. Environment | Python 3.11, MLX, virtual env | ✅ Done |
| 2. Base model | Mistral 7B v0.3 downloaded + verified | ✅ Done |
| 3. Data pipeline | Fix collection, formatting, cleaning | ✅ Done |
| 4a. Dataset | Reddit collection — 2,100 posts, 7 subreddits | ✅ Done |
| 4b. Dataset | Synthetic enhanced responses | ⏭️ Skipped (preserving human voice) |
| 4c. Dataset | Synthetic pairs for gap topics (68 examples) | ✅ Done |
| 5. Training setup | LoRA config, scripts, dry-run | ✅ Done |
| 6. Training run | QLoRA on M1 — 1000 iters, iter-400 selected | ✅ Done |
| 7. Evaluation | A/B test base vs fine-tuned, 3-way comparison | ✅ Done |
| 8. Chat UI | Gradio interface with examples | ✅ Done |
| 9. Deployment | Hugging Face Spaces | 🔜 Next |

## 👤 Author

**Benoît Rossignol**
📍 France
💼 Solution Architect @ Shopify
🧠 AI Enthusiast & Builder

- [GitHub](https://github.com/brossign)
- [LinkedIn](https://www.linkedin.com/in/benoit-rossignol/)
