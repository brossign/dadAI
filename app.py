"""
DadAI v3 — Gradio Chat Interface with RAG

A supportive AI assistant for new dads, fine-tuned on real Reddit
parenting conversations and augmented with curated parenting psychology
references via RAG (Retrieval-Augmented Generation).

Run locally:
    python app.py

The app loads the fused MLX model, connects to the ChromaDB knowledge
base, and serves a streaming chat UI.
"""

import os
from pathlib import Path

import gradio as gr
from mlx_lm import load
from mlx_lm.generate import stream_generate, make_sampler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = "models/dadai-v2-fused"
RAG_DB_PATH = "data/rag_db"
RAG_COLLECTION = "dadai_books"
RAG_NUM_RESULTS = 2  # Number of book passages to retrieve per query

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

EXAMPLE_QUESTIONS = [
    "My wife just told me she's pregnant and I'm terrified. I don't feel ready.",
    "My newborn won't stop crying at 3am. My wife is exhausted and I don't know what to do.",
    "I don't feel connected to my baby. Everyone says it's magical but I feel nothing.",
    "My wife and I keep fighting since the baby arrived. She says I don't help enough.",
    "I just went back to work and I feel guilty leaving my baby every morning.",
    "I think I might have postpartum depression as a dad. Is that even a thing?",
    "How do I be a good dad when I had a terrible father?",
    "I feel like I've lost my identity since becoming a dad.",
]

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

print("Loading DadAI model...")
model, tokenizer = load(MODEL_PATH)
sampler = make_sampler(temp=TEMPERATURE, min_p=0.05)
print("Model loaded!")

# ---------------------------------------------------------------------------
# RAG: Load knowledge base
# ---------------------------------------------------------------------------

rag_collection = None

if Path(RAG_DB_PATH).exists():
    try:
        import chromadb
        from chromadb.utils import embedding_functions

        print("Loading RAG knowledge base...")
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
        )
        client = chromadb.PersistentClient(path=RAG_DB_PATH)
        rag_collection = client.get_collection(
            name=RAG_COLLECTION,
            embedding_function=ef,
        )
        print(f"RAG loaded! {rag_collection.count()} passages in knowledge base.")
    except Exception as e:
        print(f"Warning: Could not load RAG database: {e}")
        print("Running without book knowledge (v2 mode).")
else:
    print(f"No RAG database found at {RAG_DB_PATH}.")
    print("Running without book knowledge. Run scripts/build_rag_db.py to enable RAG.")


def retrieve_context(query: str, n_results: int = RAG_NUM_RESULTS) -> str:
    """Search the knowledge base for relevant passages."""
    if rag_collection is None:
        return ""

    try:
        results = rag_collection.query(
            query_texts=[query],
            n_results=n_results,
        )

        if not results["documents"] or not results["documents"][0]:
            return ""

        passages = []
        for doc in results["documents"][0]:
            # Keep passages short to not bloat the prompt
            trimmed = doc[:300] if len(doc) > 300 else doc
            passages.append(f"- {trimmed}")

        return RAG_CONTEXT_INTRO + "\n".join(passages)

    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Chat logic
# ---------------------------------------------------------------------------

def build_prompt(user_message, rag_context=""):
    """Build Mistral-formatted prompt with system context + RAG."""
    system = SYSTEM_PROMPT
    if rag_context:
        system += rag_context

    combined = f"{system}\n\n{user_message}"
    messages = [{"role": "user", "content": combined}]

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"[INST] {system}\n\n{user_message} [/INST]"


def respond(message, history):
    """Stream a response token by token for real-time output."""
    if not message.strip():
        return ""

    # RAG: retrieve relevant book passages
    rag_context = retrieve_context(message)

    prompt = build_prompt(message, rag_context)

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
# Gradio UI
# ---------------------------------------------------------------------------

rag_status = "with book knowledge (RAG)" if rag_collection else "without book knowledge"

DESCRIPTION = f"""
# DadAI — Talk to me like a fellow dad

**DadAI** is a supportive AI fine-tuned on real parenting conversations from Reddit,
augmented with curated parenting psychology references.
It responds with empathy and practical wisdom — like a friend who's been through it all.

*Built with Mistral 7B + LoRA + RAG, running locally on Apple Silicon. Currently running {rag_status}.*
"""

FOOTER = """
---

**About DadAI** — Created by [Benoît Rossignol](https://www.linkedin.com/in/benoit-rossignol/)
| [GitHub](https://github.com/brossign/dadAI)
| Model: Mistral 7B v0.3 (4-bit) + LoRA + RAG

*DadAI is not a therapist or medical professional. If you're struggling,
please reach out to a mental health professional.*
"""

demo = gr.ChatInterface(
    fn=respond,
    title="DadAI — Support for New Dads",
    description=DESCRIPTION,
    examples=EXAMPLE_QUESTIONS,
    cache_examples=False,
    chatbot=gr.Chatbot(
        height=480,
        placeholder="Ask me anything about being a new dad...",
        label="DadAI",
    ),
    textbox=gr.Textbox(
        placeholder="Type your question here...",
        label="Your message",
        scale=7,
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
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
        ),
    )
