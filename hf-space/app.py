"""
DadAI v2 — Hugging Face Spaces Demo

A supportive AI assistant for new dads, powered by Mistral 7B Instruct.
Fine-tuned on real Reddit parenting conversations with LoRA on Apple Silicon.

This demo uses the base model with an enhanced system prompt on ZeroGPU.
The full fine-tuned model (MLX + LoRA) runs locally on Mac.
"""

import spaces
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
# Model loading (runs once on startup)
# ---------------------------------------------------------------------------

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("Loading model in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)
print("Model loaded!")


# ---------------------------------------------------------------------------
# Chat logic
# ---------------------------------------------------------------------------

@spaces.GPU
def respond(message, history):
    """Generate a response using the Mistral model."""
    if not message.strip():
        return ""

    # Build Mistral chat format
    messages = [
        {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{message}"},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens (skip the prompt)
    response = tokenizer.decode(
        output[0][input_ids.shape[1]:],
        skip_special_tokens=True,
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

- **Local model:** Mistral 7B v0.3 (4-bit MLX) + LoRA fine-tuning on 2,147 real dad conversations
- **This demo:** Mistral 7B v0.3 with enhanced system prompt on Hugging Face ZeroGPU

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


if __name__ == "__main__":
    demo.launch()
