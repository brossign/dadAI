"""
DadAI v2 — Hugging Face Spaces Demo
"""

import os
import gradio as gr
from huggingface_hub import InferenceClient

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

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

MAX_NEW_TOKENS = 512

client = InferenceClient(token=os.environ.get("HF_TOKEN"))

# ---------------------------------------------------------------------------
# Chat function — Gradio 5.x ChatInterface with type="messages"
# ---------------------------------------------------------------------------

def respond(message: str, history: list) -> str:
    """Generate a response (non-streaming for maximum compatibility)."""
    if not message.strip():
        return "Please type a question!"

    messages = [
        {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{message}"},
    ]

    response = client.chat_completion(
        model=MODEL_ID,
        messages=messages,
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.7,
        top_p=0.9,
    )

    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

demo = gr.ChatInterface(
    fn=respond,
    title="DadAI — Support for New Dads",
    description=(
        "**DadAI** is a supportive AI fine-tuned on real parenting conversations "
        "from Reddit. It responds with empathy and practical wisdom — like a "
        "friend who's been through it all.\n\n"
        "*Built with Mistral 7B + LoRA, trained locally on a MacBook Pro M1.*"
    ),
    examples=[
        "My wife just told me she's pregnant and I'm terrified.",
        "My newborn won't stop crying at 3am and I don't know what to do.",
        "I don't feel connected to my baby. Is that normal?",
        "How do I be a good dad when I had a terrible father?",
        "I think I might have postpartum depression as a dad.",
        "I feel like I've lost my identity since becoming a dad.",
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch()
