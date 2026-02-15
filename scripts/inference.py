"""
DadAI v2 — Interactive inference with the fine-tuned model.

Loads the base model + LoRA adapters and runs an interactive chat loop.
Uses Mistral's native chat template for proper prompt formatting.

Usage:
    python scripts/inference.py                    # with LoRA adapters
    python scripts/inference.py --no-adapter       # base model only (for comparison)
    python scripts/inference.py --fused-model PATH # with fused model
"""

import argparse
from mlx_lm import load, generate
from mlx_lm.generate import make_sampler


SYSTEM_PROMPT = (
    "You are DadAI, a supportive and experienced father who gives advice "
    "to new dads. You respond with empathy, warmth, and practical wisdom — "
    "like a friend who's been through it all. You share personal anecdotes, "
    "normalize struggles, and never judge. You're not a therapist or doctor — "
    "just a dad who gets it."
)


def build_prompt(user_message, tokenizer):
    """
    Build a properly formatted prompt using the model's chat template.

    Mistral's template requires strict user/assistant alternation (no system role).
    We prepend the system prompt to the user message — matching how the model
    was trained (see prepare_training_data.py).
    """
    combined_user = f"{SYSTEM_PROMPT}\n\n{user_message}"
    messages = [
        {"role": "user", "content": combined_user},
    ]

    # Use the tokenizer's chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        # Fallback to manual Mistral format
        prompt = f"[INST] {SYSTEM_PROMPT}\n\n{user_message} [/INST]"

    return prompt


def main():
    parser = argparse.ArgumentParser(description="DadAI interactive inference")
    parser.add_argument("--model", default="models/mistral-7b-instruct-v0.3-4bit",
                        help="Path to base model")
    parser.add_argument("--adapter-path", default="adapters/dadai-lora",
                        help="Path to LoRA adapters")
    parser.add_argument("--no-adapter", action="store_true",
                        help="Run without LoRA adapters (base model only)")
    parser.add_argument("--fused-model", default=None,
                        help="Path to fused model (skip adapter loading)")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Maximum tokens to generate")
    parser.add_argument("--temp", type=float, default=0.7,
                        help="Sampling temperature")
    args = parser.parse_args()

    # Load model
    if args.fused_model:
        print(f"Loading fused model from {args.fused_model}...")
        model, tokenizer = load(args.fused_model)
    elif args.no_adapter:
        print(f"Loading base model from {args.model} (no adapters)...")
        model, tokenizer = load(args.model)
    else:
        print(f"Loading model from {args.model} with adapters from {args.adapter_path}...")
        model, tokenizer = load(args.model, adapter_path=args.adapter_path)

    print("Model loaded!\n")
    print("=" * 60)
    print("  DadAI — Talk to me like a fellow dad.")
    print("  Type your question or situation. Type 'quit' to exit.")
    print("=" * 60)

    while True:
        print()
        user_input = input("You: ").strip()

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("\nTake care, dad. You're doing great. 👊")
            break

        prompt = build_prompt(user_input, tokenizer)

        sampler = make_sampler(temp=args.temp, min_p=0.05)
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=args.max_tokens,
            sampler=sampler,
        )

        print(f"\nDadAI: {response}")


if __name__ == "__main__":
    main()
