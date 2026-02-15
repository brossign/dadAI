"""
DadAI v2 — Hugging Face Spaces Demo

A supportive AI assistant for new dads, powered by Mistral 7B Instruct.
Fine-tuned on real Reddit parenting conversations with LoRA on Apple Silicon.

This demo calls Mistral via the HF Inference API (no local model loading).
The full fine-tuned model (MLX + LoRA) runs locally on Mac.
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
# Inference client (calls HF-hosted model — no local GPU/RAM needed)
# ---------------------------------------------------------------------------

client = InferenceClient(token=os.environ.get("HF_TOKEN"))


# ---------------------------------------------------------------------------
# Chat logic
# ---------------------------------------------------------------------------

def respond(message, history):
    """Stream a response from the HF Inference API."""
    if not message.strip():
        yield ""
        return

    # Build Mistral-compatible messages (system embedded in first user msg)
    messages = [
        {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{message}"},
    ]

    partial = ""
    for chunk in client.chat_completion(
        model=MODEL_ID,
        messages=messages,
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.7,
        top_p=0.9,
        stream=True,
    ):
        token = chunk.choices[0].delta.content or ""
        partial += token
        yield partial


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

- **Local model:** Mistral 7B v0.3 (4-bit MLX) + LoRA fine-tuning on 2,147 real dad conversations
- **This demo:** Mistral 7B v0.3 via HF Inference API with enhanced system prompt

*DadAI is not a therapist or medical professional. If you're struggling,
please reach out to a mental health professional.*
"""

with gr.Blocks(title="DadAI — Support for New Dads", theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPTION)

    chatbot = gr.Chatbot(height=480, label="DadAI", type="messages")
    msg = gr.Textbox(
        placeholder="Type your question here...",
        label="Your message",
        scale=7,
    )

    gr.Examples(
        examples=EXAMPLE_QUESTIONS,
        inputs=msg,
    )

    def user_submit(message, history):
        """Add user message to chat and stream assistant response."""
        history = history + [{"role": "user", "content": message}]
        return "", history

    def bot_respond(history):
        """Generate assistant response."""
        user_message = history[-1]["content"]
        history.append({"role": "assistant", "content": ""})

        messages = [
            {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user_message}"},
        ]

        partial = ""
        for chunk in client.chat_completion(
            model=MODEL_ID,
            messages=messages,
            max_tokens=MAX_NEW_TOKENS,
            temperature=0.7,
            top_p=0.9,
            stream=True,
        ):
            token = chunk.choices[0].delta.content or ""
            partial += token
            history[-1]["content"] = partial
            yield history

    msg.submit(user_submit, [msg, chatbot], [msg, chatbot]).then(
        bot_respond, chatbot, chatbot
    )

    gr.Markdown(FOOTER)


if __name__ == "__main__":
    demo.launch()
