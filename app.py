"""
DadAI v2 — Gradio Chat Interface

A supportive AI assistant for new dads, fine-tuned on real Reddit
parenting conversations. Built with Mistral 7B + LoRA on Apple Silicon.

Run locally:
    python app.py

The app loads the MLX model with LoRA adapters and serves a chat UI.
"""

import gradio as gr
from mlx_lm import load, generate
from mlx_lm.generate import make_sampler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = "models/mistral-7b-instruct-v0.3-4bit"
ADAPTER_PATH = "adapters/dadai-lora"

SYSTEM_PROMPT = (
    "You are DadAI, a supportive and experienced father who gives advice "
    "to new dads. You respond with empathy, warmth, and practical wisdom — "
    "like a friend who's been through it all. You share personal anecdotes, "
    "normalize struggles, and never judge. You're not a therapist or doctor — "
    "just a dad who gets it."
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
model, tokenizer = load(MODEL_PATH, adapter_path=ADAPTER_PATH)
sampler = make_sampler(temp=TEMPERATURE, min_p=0.05)
print("Model loaded!")


# ---------------------------------------------------------------------------
# Chat logic
# ---------------------------------------------------------------------------

def build_prompt(user_message):
    """Build Mistral-formatted prompt with system context."""
    combined = f"{SYSTEM_PROMPT}\n\n{user_message}"
    messages = [{"role": "user", "content": combined}]

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"[INST] {SYSTEM_PROMPT}\n\n{user_message} [/INST]"


def respond(message, history):
    """Generate a response to the user's message."""
    if not message.strip():
        return ""

    prompt = build_prompt(message)

    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=MAX_TOKENS,
        sampler=sampler,
    )

    return response


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

DESCRIPTION = """
# DadAI — Talk to me like a fellow dad

**DadAI** is a supportive AI fine-tuned on real parenting conversations from Reddit.
It responds with empathy and practical wisdom — like a friend who's been through it all.

*Built with Mistral 7B + LoRA, trained locally on a MacBook Pro M1.*
"""

FOOTER = """
---

**About DadAI** — Created by [Benoît Rossignol](https://www.linkedin.com/in/benoit-rossignol/)
| [GitHub](https://github.com/brossign/dadAI)
| Model: Mistral 7B v0.3 (4-bit) + LoRA
| Data: 2,147 curated examples from Reddit + synthetic

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

# Add footer below the chat
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
