"""
DadAI v3.1 — Gradio Chat Interface with RAG + Reranker

A supportive AI assistant for new dads, fine-tuned on real Reddit
parenting conversations and augmented with curated parenting psychology
references via RAG (Retrieval-Augmented Generation).

Run locally:
    python app.py

The app loads the fused Qwen2.5-14B MLX model, connects to the ChromaDB
knowledge base (3 books, 1 344 passages), and serves a streaming chat UI.
"""

import sys
from pathlib import Path

import gradio as gr
from mlx_lm import load
from mlx_lm.generate import stream_generate, make_sampler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = "models/dadai-qwen14b-fused"
RAG_DB_PATH = "data/rag_db"
RAG_COLLECTION = "dadai_books"
RAG_RETRIEVE_K = 5
RAG_RERANK_TOP = 2

SYSTEM_PROMPT = (
    "You are DadAI, a supportive and experienced father who gives advice "
    "to new dads. You respond with empathy, warmth, and practical wisdom — "
    "like a friend who's been through it all. You share personal anecdotes, "
    "normalize struggles, and never judge. You're not a therapist or doctor — "
    "just a dad who gets it.\n\n"
    "Give thoughtful, detailed responses. Share your own experiences as a dad, "
    "offer practical tips when relevant, and always make the other dad feel "
    "heard and supported. Aim for a few paragraphs — don't rush."
)

RAG_CONTEXT_INTRO = (
    "\n\nBackground knowledge (use to inform your advice — synthesize into "
    "your own words, do NOT quote or list these, just let them shape what "
    "you say):\n\n"
)

MAX_TOKENS = 512
TEMPERATURE = 0.7
MAX_HISTORY_TURNS = 3

EXAMPLE_QUESTIONS = [
    "I don't feel connected to my baby yet",
    "I'm terrified, my wife is pregnant",
    "How to be a good dad with a bad father?",
    "We keep fighting since the baby arrived",
    "I just went back to work and I feel guilty leaving my baby every morning.",
    "I think I might have postpartum depression as a dad. Is that even a thing?",
    "I feel like I've lost my identity since becoming a dad.",
    "My newborn won't stop crying at 3am and I don't know what to do.",
]

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

print("Loading DadAI model...", flush=True)
model, tokenizer = load(MODEL_PATH)
sampler = make_sampler(temp=TEMPERATURE, min_p=0.05)
print("Model loaded!", flush=True)

# ---------------------------------------------------------------------------
# RAG: Load knowledge base + reranker
# ---------------------------------------------------------------------------

_rag_collection = None
_reranker = None
_rag_loaded = False
_reranker_loaded = False

print("RAG + reranker will load on first query (saves startup memory).", flush=True)


def _get_rag():
    """Lazy-load the RAG knowledge base on first use."""
    global _rag_collection, _rag_loaded
    if _rag_loaded:
        return _rag_collection
    _rag_loaded = True
    if not Path(RAG_DB_PATH).exists():
        print("No RAG database found.")
        return None
    try:
        import chromadb
        from chromadb.utils import embedding_functions

        print("Loading RAG knowledge base (first query)...")
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
        )
        client = chromadb.PersistentClient(path=RAG_DB_PATH)
        _rag_collection = client.get_collection(
            name=RAG_COLLECTION,
            embedding_function=ef,
        )
        count = _rag_collection.count()
        print(f"RAG loaded! {count} passages in knowledge base.")
    except Exception as e:
        print(f"Warning: Could not load RAG: {e}")
        _rag_collection = None
    return _rag_collection


def _get_reranker():
    """Lazy-load the cross-encoder reranker on first use."""
    global _reranker, _reranker_loaded
    if _reranker_loaded:
        return _reranker
    _reranker_loaded = True
    try:
        from sentence_transformers import CrossEncoder
        print("Loading reranker...")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
        print("Reranker loaded!")
    except Exception as e:
        print(f"Warning: Could not load reranker: {e}")
        _reranker = None
    return _reranker


def retrieve_context(query: str) -> str:
    """Search the knowledge base for relevant passages, rerank with cross-encoder."""
    col = _get_rag()
    if col is None:
        return ""

    try:
        results = col.query(
            query_texts=[query],
            n_results=RAG_RETRIEVE_K,
        )

        if not results["documents"] or not results["documents"][0]:
            return ""

        candidates = results["documents"][0]

        rr = _get_reranker()
        if rr is not None and len(candidates) > 1:
            pairs = [(query, doc) for doc in candidates]
            scores = rr.predict(pairs)
            ranked = sorted(zip(scores, candidates), reverse=True)
            best = [doc for _, doc in ranked[:RAG_RERANK_TOP]]
        else:
            best = candidates[:RAG_RERANK_TOP]

        return RAG_CONTEXT_INTRO + "\n".join(f"- {d[:400]}" for d in best)

    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Chat logic
# ---------------------------------------------------------------------------

def build_prompt(user_message, history=None, rag_context=""):
    """Build ChatML prompt with system context, history, and RAG."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history:
        relevant = [m for m in history if m["role"] in ("user", "assistant")]
        cap = MAX_HISTORY_TURNS * 2
        if len(relevant) > cap:
            relevant = relevant[-cap:]
        messages.extend(relevant)

    content = user_message
    if rag_context:
        content = (
            f"A dad is asking: {user_message}\n{rag_context}\n"
            "Now respond to this dad with empathy and the wisdom above."
        )
    messages.append({"role": "user", "content": content})

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    parts = []
    for msg in messages:
        if msg["role"] == "user":
            parts.append(f"[INST] {msg['content']} [/INST]")
        elif msg["role"] == "assistant":
            parts.append(msg["content"])
    return " ".join(parts)


def respond(message, history):
    """Stream a response token by token for real-time output."""
    if not message.strip():
        return ""

    rag_context = retrieve_context(message)

    chat_history = []
    if history:
        for turn in history[-MAX_HISTORY_TURNS:]:
            if isinstance(turn, (list, tuple)) and len(turn) == 2:
                chat_history.append({"role": "user", "content": turn[0]})
                if turn[1]:
                    chat_history.append({"role": "assistant", "content": turn[1]})

    prompt = build_prompt(message, history=chat_history, rag_context=rag_context)

    partial = ""
    for response in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=MAX_TOKENS,
        sampler=sampler,
    ):
        partial += response.text
        yield partial


# ---------------------------------------------------------------------------
# Gradio UI (simple ChatInterface — the original clean version)
# ---------------------------------------------------------------------------

rag_count = "1,637"  # 4 books indexed

DESCRIPTION = f"""
# DadAI — Talk to me like a fellow dad

**DadAI** is a supportive AI fine-tuned on real parenting conversations from Reddit,
augmented with curated parenting psychology references via RAG.
It responds with empathy and practical wisdom — like a friend who's been through it all.

*Built with Qwen2.5-14B + LoRA + RAG, running locally on Apple Silicon. Knowledge base: {rag_count} passages.*
"""

FOOTER = """
---

**About DadAI** — Created by [Benoît Rossignol](https://www.linkedin.com/in/benoit-rossignol/)
| [GitHub](https://github.com/brossign/dadAI)
| Model: Qwen2.5-14B-Instruct (4-bit) + LoRA + RAG (3 books)

*DadAI is not a therapist or medical professional. If you're struggling,
please reach out to a mental health professional.*
"""

demo = gr.ChatInterface(
    fn=respond,
    title="DadAI — Support for New Dads",
    description=DESCRIPTION,
    chatbot=gr.Chatbot(
        height=480,
        placeholder="Ask me anything about being a new dad...",
        label="DadAI",
    ),
    textbox=gr.Textbox(
        placeholder="Type your question here...",
        label="Your message",
    ),
    fill_height=True,
)

with demo:
    gr.Markdown(FOOTER)


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
