"""
Compare Qwen 14B vs Mistral 7B on the same prompts.
Runs both models, same question, same RAG, side by side.
"""

import time
from pathlib import Path
from mlx_lm import load, generate
from mlx_lm.generate import make_sampler

# RAG setup
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder

RAG_DB_PATH = "data/rag_db"
RAG_COLLECTION = "dadai_books"

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",
)
client = chromadb.PersistentClient(path=RAG_DB_PATH)
rag_collection = client.get_collection(name=RAG_COLLECTION, embedding_function=ef)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
print(f"RAG loaded: {rag_collection.count()} passages")

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


def retrieve_context(query):
    results = rag_collection.query(query_texts=[query], n_results=5)
    if not results["documents"] or not results["documents"][0]:
        return ""
    candidates = results["documents"][0]
    pairs = [(query, doc) for doc in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), reverse=True)
    best = ranked[0][1]
    trimmed = best[:400] if len(best) > 400 else best
    return RAG_CONTEXT_INTRO + f"- {trimmed}"


def build_prompt(tokenizer, user_message, rag_context="", model_name=""):
    if rag_context:
        user_content = (
            f"A dad is asking: {user_message}\n"
            f"{rag_context}\n"
            f"Now respond to this dad with empathy and the wisdom above."
        )
    else:
        user_content = user_message

    # Qwen supports system role; Mistral needs system prepended to user msg
    if "qwen" in model_name.lower():
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
    else:
        messages = [
            {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user_content}"},
        ]

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"[INST] {SYSTEM_PROMPT}\n\n{user_content} [/INST]"


TEST_QUESTIONS = [
    "How do I be a good dad when I had a terrible father?",
    "I don't feel connected to my baby. Everyone says it's magical but I feel nothing.",
    "My wife and I keep fighting since the baby arrived. She says I don't help enough.",
]

MODELS = [
    ("Mistral 7B (v2)", "models/dadai-v2-fused"),
    ("Qwen 14B (v4)", "models/dadai-qwen14b-fused"),
]

sampler = make_sampler(temp=0.7, min_p=0.05)

for model_name, model_path in MODELS:
    print(f"\n{'='*70}")
    print(f"LOADING: {model_name} ({model_path})")
    print(f"{'='*70}")

    model, tokenizer = load(model_path)

    for q in TEST_QUESTIONS:
        rag_context = retrieve_context(q)

        prompt = build_prompt(tokenizer, q, rag_context, model_name=model_name)

        t0 = time.time()
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=512,
            sampler=sampler,
            verbose=False,
        )
        elapsed = time.time() - t0

        print(f"\n--- Q: {q}")
        print(f"--- ({elapsed:.1f}s, {len(response)} chars)")
        print(response)
        print()

    # Free memory before loading next model
    del model, tokenizer
    import gc
    gc.collect()
    import mlx.core as mx
    mx.metal.clear_cache()
