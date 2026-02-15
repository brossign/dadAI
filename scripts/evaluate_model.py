"""
DadAI v2 — Model Evaluation Script

Runs a set of test prompts through both the base model and the
fine-tuned model (with LoRA adapters) to compare responses.

This helps assess:
1. Did fine-tuning change the model's tone/personality?
2. Is the model more empathetic and dad-like?
3. Are responses coherent and helpful?
4. Any signs of overfitting or repetition?

Usage:
    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --adapter-path adapters/dadai-lora/0000400_adapters.safetensors
"""

import argparse
import time
from mlx_lm import load, generate
from mlx_lm.generate import make_sampler


SYSTEM_PROMPT = (
    "You are DadAI, a supportive and experienced father who gives advice "
    "to new dads. You respond with empathy, warmth, and practical wisdom — "
    "like a friend who's been through it all. You share personal anecdotes, "
    "normalize struggles, and never judge. You're not a therapist or doctor — "
    "just a dad who gets it."
)

# Test prompts covering different aspects of fatherhood
TEST_PROMPTS = [
    # Emotional / vulnerability
    "My wife just told me she's pregnant and I'm terrified. I don't feel ready at all.",

    # Practical advice
    "My newborn won't stop crying at 3am and I don't know what to do. My wife is exhausted.",

    # Relationship / partner support
    "My wife and I keep fighting since the baby arrived. She says I don't help enough but I'm trying my best.",

    # Mental health
    "I think I might have postpartum depression as a dad. Is that even a thing?",

    # Identity / loss of self
    "I used to love going out with friends and playing sports. Now I just feel trapped at home with a baby.",

    # Bonding
    "I don't feel connected to my baby. Everyone says it's magical but I feel nothing. What's wrong with me?",

    # Work-life balance
    "I just went back to work after paternity leave and I feel guilty leaving my baby every morning.",

    # Synthetic topic: breaking generational cycles
    "How do I be a good dad when I had a terrible father?",
]


def build_prompt(user_message, tokenizer):
    """Build prompt using Mistral's chat template."""
    combined_user = f"{SYSTEM_PROMPT}\n\n{user_message}"
    messages = [{"role": "user", "content": combined_user}]

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"[INST] {SYSTEM_PROMPT}\n\n{user_message} [/INST]"


def run_evaluation(model, tokenizer, label, max_tokens=300, temp=0.7):
    """Run all test prompts and collect responses."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    sampler = make_sampler(temp=temp, min_p=0.05)

    responses = []
    for i, prompt_text in enumerate(TEST_PROMPTS, 1):
        prompt = build_prompt(prompt_text, tokenizer)

        start = time.time()
        response = generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        )
        elapsed = time.time() - start

        responses.append(response)

        print(f"\n--- Question {i}/{len(TEST_PROMPTS)} ({elapsed:.1f}s) ---")
        print(f"Q: {prompt_text}")
        print(f"A: {response}")

    return responses


def main():
    parser = argparse.ArgumentParser(description="Evaluate DadAI model")
    parser.add_argument("--model", default="models/mistral-7b-instruct-v0.3-4bit")
    parser.add_argument("--adapter-path", default="adapters/dadai-lora")
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--skip-base", action="store_true",
                        help="Skip base model evaluation")
    args = parser.parse_args()

    # Run base model
    if not args.skip_base:
        print("Loading BASE model (no fine-tuning)...")
        model, tokenizer = load(args.model)
        base_responses = run_evaluation(
            model, tokenizer, "BASE MODEL (Mistral 7B — no fine-tuning)",
            max_tokens=args.max_tokens, temp=args.temp,
        )
        del model  # Free memory before loading next

    # Run fine-tuned model
    print(f"\nLoading FINE-TUNED model (adapters: {args.adapter_path})...")
    model, tokenizer = load(args.model, adapter_path=args.adapter_path)
    ft_responses = run_evaluation(
        model, tokenizer, "FINE-TUNED MODEL (DadAI LoRA)",
        max_tokens=args.max_tokens, temp=args.temp,
    )

    print(f"\n{'='*70}")
    print("  Evaluation complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
