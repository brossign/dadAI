# DadAI — An AI Assistant for New Dads

**DadAI** is an open-source AI built to support new fathers during pregnancy and early parenthood.
Fine-tuned on real parenting conversations from Reddit and augmented with curated parenting psychology via RAG.

**Try it now:** [huggingface.co/spaces/benlongi/DadAI](https://huggingface.co/spaces/benlongi/DadAI)

## Why DadAI?

Most parenting resources are either mother-centric or scattered across forums.
As a first-time dad, I realized how hard it can be to find support that's both practical and emotionally relevant — so I built an AI that talks to you like a friend who's been through it all.

DadAI covers:
- Emotional support during pregnancy and early parenthood
- Sleep deprivation, relationship strain, identity loss
- Dad mental health, bonding struggles, work-life guilt
- Practical tips from real fathers who've been there

## Press & Articles

- [How a Solo Dev Built an AI for Dads — RunPod Blog (May 2025)](https://www.runpod.io/blog/solo-dev-ai-for-dads-runpod)
- [How I Fine-Tuned a Custom AI Model (DadAI) — LinkedIn (Apr 2025)](https://www.linkedin.com/pulse/how-i-built-fine-tuned-my-own-dadai-what-taught-me-llms-rossignol-pq7he/)

## Project Evolution

### v1 (April 2025) — Cloud-based, RunPod + GPTQ

The original version was built as a learning project:
- **Mistral 7B Instruct v0.1** (GPTQ quantized)
- **QLoRA + PEFT** fine-tuning on **RunPod** (RTX 4090, ~$5 total)
- **298 Reddit posts** from 4 subreddits
- No UI — CLI only

**What went wrong:** A thorough code audit (by Claude) uncovered 5 critical bugs:
1. **Tokenization bug** — the model never trained on completions (labels were wrong)
2. **Prompt template mismatch** — training used `[INST]` format but inference used a different template
3. **No `mask_prompt`** — the model trained on the prompts too, diluting learning
4. **Small, noisy dataset** — only 298 pairs, ~30% bot contamination, no quality filtering
5. **Format incompatibility** — GPTQ to GGUF to LocalAI deployment never worked

See the `v0.1-original` tag for the original codebase.

### v2 (February 2026) — Local-first, Mac + MLX

A complete rewrite over a weekend, powered by Apple's **MLX framework**:

| | v1 (2025) | v2 (2026) |
|--|-----------|-----------|
| **Base model** | Mistral 7B v0.1 (GPTQ) | Mistral 7B Instruct v0.3 (MLX 4-bit) |
| **Training** | RunPod RTX 4090 ($5) | MacBook Pro M1 (free) |
| **Framework** | HuggingFace + PEFT + bitsandbytes | Apple MLX + mlx-lm |
| **Dataset** | 298 pairs (buggy pipeline, 30% bots) | 2,147 curated pairs (0% bots) |
| **Data sources** | 4 subreddits | 7 subreddits + 68 synthetic gap topics |
| **Key training fix** | None (trained on prompts) | `mask_prompt: true` (trains on completions only) |
| **Deployment** | LocalAI (never worked) | Gradio + HF Spaces |
| **UI** | None | Chat interface with streaming |

### v3 (February 2026) — RAG: Giving DadAI a Bookshelf

v2 taught DadAI *how to talk* like a supportive dad. v3 gives it *what to know*.

**The insight:** Fine-tuning and RAG are complementary:
- **Fine-tuning** = personality. The model studied 2,147 real dad conversations and internalized empathy, warmth, and tone.
- **RAG** = knowledge. When a dad asks a question, the model searches a curated knowledge base of parenting psychology and weaves expert insights into its response.

They stack: the warm dad voice from fine-tuning meets grounded wisdom from books. No retraining needed.

**New in v3:**
- EPUB/PDF book extraction and semantic chunking pipeline
- ChromaDB vector database with `all-MiniLM-L6-v2` embeddings (295 passages)
- Automatic retrieval of the 2 most relevant passages per question
- Background knowledge injected into the model's prompt at generation time
- Graceful fallback to v2 behavior if no knowledge base is present

**Honest status:** The architecture works — the right passages surface for the right questions. The 7B model sometimes struggles to fully synthesize retrieved knowledge into its responses. This is an active area of improvement.

## Tech Stack (v3)

- **Model:** [Mistral 7B Instruct v0.3 (4-bit MLX)](https://huggingface.co/mlx-community/Mistral-7B-Instruct-v0.3-4bit) — 3.8 GB on disk
- **Training:** QLoRA fine-tuning via [mlx-lm](https://github.com/ml-explore/mlx-lm) on Apple Silicon
- **Data:** 2,096 real Reddit Q&A pairs + 68 synthetic pairs for under-covered topics
- **RAG:** ChromaDB + sentence-transformers for semantic retrieval from curated knowledge base
- **UI:** [Gradio](https://gradio.app) chat interface with streaming responses
- **Local inference:** Fused model (LoRA baked into base weights) for fast generation
- **Online demo:** [HF Spaces](https://huggingface.co/spaces/benlongi/DadAI) via Inference API
- **Language:** Python 3.11

## Project Structure

```
dadAI/
├── app.py                         # Gradio chat UI (local, uses fused model + RAG)
├── hf-space/                      # Hugging Face Spaces deployment
│   ├── app.py                     #   HF demo (Inference API, no local model)
│   ├── requirements.txt
│   └── README.md
├── data/                          # Datasets
│   ├── reddit_dataset.jsonl       #   Raw Reddit posts (2,100)
│   ├── formatted_dataset.jsonl    #   Mistral [INST] prompt/completion pairs
│   ├── cleaned_dataset.jsonl      #   Filtered, deduplicated (2,096)
│   ├── synthetic_gap_topics.jsonl #   Synthetic pairs for gap topics (68)
│   ├── training_dataset.jsonl     #   Final merged dataset (2,164)
│   ├── mlx_training/              #   Train/valid/test splits for mlx-lm
│   ├── rag_chunks.jsonl           #   Book chunks for RAG (gitignored)
│   └── rag_db/                    #   ChromaDB vector database (gitignored)
├── scripts/                       # Pipeline scripts
│   ├── collect_reddit_data.py     #   Reddit data collection (PRAW)
│   ├── format_reddit_data.py      #   Convert to Mistral chat format
│   ├── clean_dataset.py           #   Quality filtering & dedup
│   ├── check_dataset_format.py    #   Validation
│   ├── generate_synthetic_data.py #   Synthetic data for gap topics
│   ├── prepare_training_data.py   #   mlx-lm format + token filtering + split
│   ├── chunk_book.py              #   Extract & chunk books for RAG (v3)
│   ├── build_rag_db.py            #   Build ChromaDB vector database (v3)
│   ├── inference.py               #   Interactive CLI chat
│   ├── evaluate_model.py          #   A/B comparison: base vs fine-tuned
│   ├── deploy_to_hf.py            #   One-command HF Spaces deployment
│   └── show_random_sample.py      #   Quick dataset inspection
├── books/                         # Source books for RAG (gitignored)
├── training_config.yaml           # MLX LoRA training config
├── train.sh                       # One-command training script
├── Makefile                       # Pipeline commands
├── models/                        # Downloaded models (gitignored)
├── adapters/                      # LoRA adapters (gitignored)
├── fused_model/                   # Base + LoRA merged (gitignored)
├── .env                           # Reddit API credentials (gitignored)
└── .venv/                         # Python virtual environment (gitignored)
```

## Setup

### Prerequisites
- macOS with Apple Silicon (M1/M2/M3/M4)
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
pip install mlx-lm huggingface_hub gradio praw python-dotenv tqdm transformers
pip install chromadb sentence-transformers ebooklib beautifulsoup4  # For RAG (v3)

# Download the base model (~3.8 GB)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('mlx-community/Mistral-7B-Instruct-v0.3-4bit', local_dir='models/mistral-7b-instruct-v0.3-4bit')
"
```

### RAG Setup (v3)

To add book knowledge:

```bash
# Place your book (EPUB or PDF) in the books/ directory
# Then chunk and index it:
python scripts/chunk_book.py --input books/your_book.epub
python scripts/build_rag_db.py
```

The app automatically detects the RAG database at startup and uses it if available.

## Training

```bash
source .venv/bin/activate

# One-command training (prepare data + fine-tune + evaluate)
./train.sh

# Or step by step:
make prepare    # Convert data to mlx-lm format + split
make train      # Run LoRA fine-tuning (~60-80 min on M1)
make test       # Evaluate on held-out test set
make chat       # Interactive chat with fine-tuned model
```

**Training details:**
- **Method:** QLoRA (model is 4-bit quantized) + LoRA rank 16
- **Key fix from v1:** `mask_prompt: true` ensures the model only trains on completions
- **Memory:** Peak ~7 GB (comfortable on 16 GB Mac, other apps keep running)
- **Dataset:** 2,147 examples after token-length filtering (from 2,164)
- **Best checkpoint:** Iteration 400 (out of 1,000) — selected via A/B evaluation
- **NaN fix:** Sequences > 2048 tokens pre-filtered to prevent gradient explosion in 4-bit QLoRA
- **Config:** See `training_config.yaml` for all hyperparameters

## Chat UI

### Local (full fine-tuned model + RAG)
```bash
source .venv/bin/activate
python app.py
# Open http://localhost:7860
```

Uses the fused model with streaming responses. If a RAG knowledge base is present, it automatically retrieves relevant passages for each question.

### Online demo
Visit [huggingface.co/spaces/benlongi/DadAI](https://huggingface.co/spaces/benlongi/DadAI)

Uses Mistral 7B via HF Inference API with the DadAI system prompt.

## Key Lessons Learned

1. **Always check your training labels.** v1's biggest bug: the tokenization was wrong, so the model never learned from completions. `mask_prompt` is essential.
2. **Prompt template consistency matters.** Train and infer with the same format. Use `tokenizer.apply_chat_template()` everywhere.
3. **MLX makes local fine-tuning real.** In 2025, I spent $5 on RunPod and hit format walls. In 2026, MLX on a MacBook Pro M1 just works.
4. **Clean data beats more data.** 2,096 filtered Reddit pairs beat 298 noisy ones. Quality > quantity.
5. **Early stopping wins.** Iteration 400 beat iteration 1000 in A/B testing.
6. **Fine-tuning gives personality. RAG gives knowledge.** They're complementary. Fine-tune for *how* to respond, RAG for *what* to say.
7. **Small models have real limits.** A 7B model can do empathetic tone OR knowledge synthesis well, but combining both in one response is where it struggles. Honest documentation of limitations teaches more than pretending it's perfect.

## Author

**Benoît Rossignol**
- Based in France
- Solution Architect @ Shopify
- [GitHub](https://github.com/brossign)
- [LinkedIn](https://www.linkedin.com/in/benoit-rossignol/)

## License

MIT
