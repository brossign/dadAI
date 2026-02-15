# DadAI — An AI Assistant for New Dads

**DadAI** is an open-source AI built to support new fathers during pregnancy and early parenthood.
Fine-tuned on real parenting conversations from Reddit and augmented with curated parenting psychology via RAG.

**Try the demo:** [huggingface.co/spaces/benlongi/DadAI](https://huggingface.co/spaces/benlongi/DadAI)
*(Note: the HF demo uses a standard Mistral model via API. The full fine-tuned Qwen 14B + RAG runs locally — clone this repo to try the real thing.)*

## Why DadAI?

Most parenting resources are either mother-centric or scattered across forums.
As a first-time dad, I realized how hard it can be to find support that's both practical and emotionally relevant — so I built an AI that talks to you like a friend who's been through it all.

DadAI covers:
- Emotional support during pregnancy and early parenthood
- Sleep deprivation, relationship strain, identity loss
- Dad mental health, bonding struggles, work-life guilt
- Couple conflict after baby, breaking generational patterns
- Practical tips from real fathers who've been there

## Press & Articles

- [How a Solo Dev Built an AI for Dads — RunPod Blog (May 2025)](https://www.runpod.io/blog/solo-dev-ai-for-dads-runpod)
- [How I Fine-Tuned a Custom AI Model (DadAI) — LinkedIn (Apr 2025)](https://www.linkedin.com/pulse/how-i-built-fine-tuned-my-own-dadai-what-taught-me-llms-rossignol-pq7he/)

## Project Evolution

### v1 (April 2025) — Cloud-based, RunPod + GPTQ

The original version was built as a hands-on learning exercise with ChatGPT:
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

A complete rewrite over a weekend with Claude via Cursor, powered by Apple's **MLX framework**:

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
- **Fine-tuning** = personality. The model studied real dad conversations and internalized empathy, warmth, and tone.
- **RAG** = knowledge. When a dad asks a question, the model searches a curated knowledge base of parenting psychology and weaves expert insights into its response.

They stack: the warm dad voice from fine-tuning meets grounded wisdom from books. No retraining needed.

### v3.1 (February 2026) — Current: Qwen 14B + 4 Books + Reranker

The version that actually delivers. Three major upgrades:

| | v3 | v3.1 (current) |
|--|-----|----------------|
| **Model** | Mistral 7B (4-bit) | **Qwen2.5-14B-Instruct (4-bit)** |
| **Training data** | 2,147 pairs | **2,260 pairs** (5% synthetic) |
| **RAG knowledge** | 1 book (295 passages) | **4 books (1,637 passages)** |
| **Retrieval** | Top-2 vector search | **Top-5 + cross-encoder reranker** |
| **Conversation** | Stateless | **3-turn memory** |
| **Training time** | ~80 min (M1) | **~2.5 hrs (M1)** |

**Why the upgrade matters:** The 7B model could do empathy *or* knowledge synthesis — not both in one response. The 14B model weaves book-informed advice into a natural dad voice. The cross-encoder reranker ensures the *right* passages get retrieved, not just the closest-sounding ones.

## Tech Stack (v3.1)

- **Model:** [Qwen2.5-14B-Instruct (4-bit MLX)](https://huggingface.co/mlx-community/Qwen2.5-14B-Instruct-4bit) — ~8.3 GB on disk
- **Training:** QLoRA fine-tuning via [mlx-lm](https://github.com/ml-explore/mlx-lm) on Apple Silicon
- **Data:** 2,147 real Reddit Q&A pairs + 113 synthetic pairs for under-covered topics
- **RAG:** ChromaDB + sentence-transformers (`all-MiniLM-L6-v2`) for semantic retrieval
- **Reranker:** Cross-encoder (`ms-marco-MiniLM-L6-v2`) for two-stage retrieval
- **UI:** [Gradio](https://gradio.app) chat interface with streaming responses
- **Local inference:** Fused model (LoRA baked into base weights) for fast generation
- **Online demo:** [HF Spaces](https://huggingface.co/spaces/benlongi/DadAI) via Inference API (standard model)
- **Language:** Python 3.11

## Project Structure

```
dadAI/
├── app.py                           # Gradio chat UI (local, fused model + RAG + reranker)
├── hf-space/                        # Hugging Face Spaces deployment
│   ├── app.py                       #   HF demo (Inference API, standard model)
│   ├── requirements.txt
│   └── README.md
├── data/                            # Datasets
│   ├── reddit_dataset.jsonl         #   Raw Reddit posts (~2,100)
│   ├── formatted_dataset.jsonl      #   ChatML prompt/completion pairs
│   ├── cleaned_dataset.jsonl        #   Filtered, deduplicated
│   ├── synthetic_gap_topics.jsonl   #   Synthetic pairs for gap topics
│   ├── synthetic_v31_pairs.jsonl    #   Additional v3.1 synthetic pairs (5% ratio)
│   ├── training_dataset.jsonl       #   Final merged dataset (2,260)
│   ├── mlx_training/                #   Train/valid/test splits for mlx-lm
│   └── rag_db/                      #   ChromaDB vector database (gitignored)
├── scripts/                         # Pipeline scripts
│   ├── collect_reddit_data.py       #   Reddit data collection (PRAW)
│   ├── format_reddit_data.py        #   Convert to chat format
│   ├── clean_dataset.py             #   Quality filtering & dedup
│   ├── check_dataset_format.py      #   Validation
│   ├── generate_synthetic_data.py   #   Synthetic data for gap topics
│   ├── generate_synthetic_v31.py    #   V3.1 synthetic pairs (5% ratio)
│   ├── prepare_training_data.py     #   mlx-lm format + token filtering + split
│   ├── chunk_book.py                #   Extract & chunk EPUBs for RAG
│   ├── build_rag_db.py              #   Build ChromaDB vector database
│   ├── compare_models.py            #   Side-by-side model comparison
│   ├── inference.py                 #   Interactive CLI chat
│   ├── evaluate_model.py            #   A/B comparison: base vs fine-tuned
│   └── deploy_to_hf.py             #   One-command HF Spaces deployment
├── books/                           # Source books for RAG (gitignored, copyrighted)
├── training_config.yaml             # MLX LoRA training config (Mistral 7B)
├── training_config_qwen14b.yaml     # MLX LoRA training config (Qwen 14B)
├── train.sh                         # One-command training script
├── Makefile                         # Pipeline commands
├── models/                          # Downloaded/fused models (gitignored)
├── adapters/                        # LoRA adapters (gitignored)
├── requirements.txt                 # Python dependencies
├── .env                             # Reddit API credentials (gitignored)
└── .venv/                           # Python virtual environment (gitignored)
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
pip install -r requirements.txt
```

### Download the base model

For **Qwen 14B** (recommended, v3.1):
```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('mlx-community/Qwen2.5-14B-Instruct-4bit', local_dir='models/qwen2.5-14b-instruct-4bit')
"
```

For **Mistral 7B** (lighter, v2):
```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('mlx-community/Mistral-7B-Instruct-v0.3-4bit', local_dir='models/mistral-7b-instruct-v0.3-4bit')
"
```

### RAG Setup

To add book knowledge, place EPUB files in `books/` and run:

```bash
# Chunk each book into passages
python scripts/chunk_book.py --input books/your_book.epub --output data/rag_chunks_yourbook.jsonl

# Build/update the ChromaDB vector database
python scripts/build_rag_db.py
```

The app automatically detects the RAG database at startup and uses it if available. Without it, DadAI still works — it just won't have book knowledge.

## Training

### Qwen 14B (v3.1, recommended)

```bash
source .venv/bin/activate

# Prepare data
python scripts/prepare_training_data.py

# Train (~2.5 hours on M1 16GB)
mlx_lm.lora --config training_config_qwen14b.yaml

# Fuse adapter into base model
mlx_lm.fuse \
  --model models/qwen2.5-14b-instruct-4bit \
  --adapter-path adapters/dadai-qwen14b-lora \
  --save-path models/dadai-qwen14b-fused
```

### Mistral 7B (v2, lighter)

```bash
# Train (~80 min on M1 16GB)
mlx_lm.lora --config training_config.yaml

# Fuse adapter
mlx_lm.fuse \
  --model models/mistral-7b-instruct-v0.3-4bit \
  --adapter-path adapters/dadai-lora \
  --save-path models/dadai-v2-fused
```

### Training details
- **Method:** QLoRA (4-bit quantized base) + LoRA rank 16
- **Key fix from v1:** `mask_prompt: true` ensures the model only trains on completions
- **Memory:** Peak ~10-12 GB for 14B, ~7 GB for 7B
- **Dataset:** 2,260 examples (2,147 Reddit + 113 synthetic)
- **Best checkpoint:** Selected via A/B evaluation (iteration 400 for 7B, full run for 14B)
- **NaN prevention:** Sequences > 2,048 tokens pre-filtered to prevent gradient explosion in 4-bit QLoRA
- **Config:** See `training_config_qwen14b.yaml` for all hyperparameters

## Chat UI

### Local (full fine-tuned model + RAG)
```bash
source .venv/bin/activate
python app.py
# Open http://localhost:7860
```

Uses the fused model with streaming responses. RAG and the cross-encoder reranker load lazily on the first query to keep startup fast. Conversation history (up to 3 turns) is maintained automatically.

### Online demo
Visit [huggingface.co/spaces/benlongi/DadAI](https://huggingface.co/spaces/benlongi/DadAI)

Uses Mistral 7B via HF Inference API with the DadAI system prompt. This is **not** the fine-tuned model — it's a standard model with DadAI's prompt engineering. For the real experience, run locally.

## Reproducing From Scratch

If you want to rebuild DadAI from zero:

1. **Set up environment** — Follow the Installation steps above
2. **Collect Reddit data** — Create a `.env` with Reddit API credentials ([get them here](https://www.reddit.com/prefs/apps)), then `make collect`
3. **Process data** — `make format && make clean && make check`
4. **Add synthetic data** — `python scripts/generate_synthetic_data.py && python scripts/generate_synthetic_v31.py`
5. **Prepare for training** — `python scripts/prepare_training_data.py`
6. **Download base model** — See instructions above
7. **Train** — `mlx_lm.lora --config training_config_qwen14b.yaml` (~2.5 hrs on M1)
8. **Fuse** — `mlx_lm.fuse --model models/qwen2.5-14b-instruct-4bit --adapter-path adapters/dadai-qwen14b-lora --save-path models/dadai-qwen14b-fused`
9. **Add books for RAG** — Place EPUBs in `books/`, chunk with `scripts/chunk_book.py`, index with `scripts/build_rag_db.py`
10. **Run** — `python app.py`

**Note:** Books are not included in the repo (copyrighted). You'll need to source your own parenting/fatherhood books for RAG. DadAI works without them — you just won't get book-informed responses.

## Key Lessons Learned

1. **Always check your training labels.** v1's biggest bug: the tokenization was wrong, so the model never learned from completions. `mask_prompt` is essential.
2. **Prompt template consistency matters.** Train and infer with the same format. Use `tokenizer.apply_chat_template()` everywhere.
3. **MLX makes local fine-tuning real.** A MacBook M1 fine-tunes a 14B model in 2.5 hours. No cloud GPU needed.
4. **Clean data beats more data.** 2,260 filtered pairs beat 298 noisy ones. Quality > quantity.
5. **Early stopping wins.** Iteration 400 beat iteration 1000 for the 7B model. Test, don't assume.
6. **Fine-tuning gives personality. RAG gives knowledge.** They're complementary. Fine-tune for *how* to respond, RAG for *what* to say.
7. **Two-stage retrieval matters.** A cross-encoder reranker on top of vector search catches what embedding similarity misses.
8. **Test the bigger model before committing.** We tried 24B, measured disk-swapping, pivoted to 14B. Data-driven decisions save time.
9. **Remove complexity before adding it.** V1's LocalAI + Docker + GPTQ pipeline was replaced by a single Gradio file.
10. **Ship the honest version.** Document limitations alongside wins. Every failure teaches something.

## Author

**Benoît Rossignol**
- Based in France
- Solution Architect @ Shopify
- [GitHub](https://github.com/brossign)
- [LinkedIn](https://www.linkedin.com/in/benoit-rossignol/)

## License

MIT
