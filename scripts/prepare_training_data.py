"""
Prepare training data for MLX LoRA fine-tuning.

Converts the training dataset into mlx-lm's expected chat format
and splits into train/valid/test sets.

mlx-lm expects:
- data/train.jsonl  (required for --train)
- data/valid.jsonl  (optional, for validation during training)
- data/test.jsonl   (optional, for --test evaluation)

Each line in chat format:
{"messages": [
    {"role": "user", "content": "..."},       # system prompt + question
    {"role": "assistant", "content": "..."}   # dad response
]}

Mistral requires strict user/assistant alternation (no system role),
so we embed the system prompt into the user message.

This format lets mlx-lm apply Mistral's native chat template
automatically, and --mask-prompt trains only on the assistant's
response (fixing the v1 bug where the model never saw completions).

IMPORTANT: Sequences exceeding --max-tokens are filtered out to
prevent NaN gradient explosions during 4-bit QLoRA training.
"""

import json
import random
import argparse
from pathlib import Path

# System prompt — same as in format_reddit_data.py
SYSTEM_PROMPT = (
    "You are DadAI, a supportive and experienced father who gives advice "
    "to new dads. You respond with empathy, warmth, and practical wisdom — "
    "like a friend who's been through it all. You share personal anecdotes, "
    "normalize struggles, and never judge. You're not a therapist or doctor — "
    "just a dad who gets it."
)


def extract_user_message(prompt):
    """
    Extract the user message from the [INST] formatted prompt.
    Removes the system prompt and [INST]/[/INST] tags.
    """
    # Remove [INST] and [/INST] tags
    text = prompt.replace("[INST]", "").replace("[/INST]", "").strip()

    # Remove the system prompt if present
    if SYSTEM_PROMPT in text:
        text = text.replace(SYSTEM_PROMPT, "").strip()

    return text


def convert_to_chat_format(prompt, completion):
    """
    Convert a prompt/completion pair to mlx-lm chat format.

    Mistral's chat template requires strict user/assistant alternation
    (no separate "system" role). We prepend the system prompt to the
    first user message.
    """
    user_message = extract_user_message(prompt)

    # Embed system prompt into the user message for Mistral compatibility
    combined_user = f"{SYSTEM_PROMPT}\n\n{user_message}"

    return {
        "messages": [
            {"role": "user", "content": combined_user},
            {"role": "assistant", "content": completion},
        ]
    }


def count_tokens_approx(messages):
    """
    Approximate token count using character length.
    Rough heuristic: ~4 chars per token for English text.
    We use a conservative 3.5 to slightly over-estimate.
    """
    total_chars = sum(len(m["content"]) for m in messages)
    # Add ~20 tokens for chat template overhead ([INST], [/INST], etc.)
    return int(total_chars / 3.5) + 20


def count_tokens_exact(messages, tokenizer):
    """Exact token count using the model's tokenizer."""
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return len(tokenizer.encode(text))


def main():
    parser = argparse.ArgumentParser(description="Prepare data for MLX LoRA training")
    parser.add_argument("--input", default="data/training_dataset.jsonl",
                        help="Input JSONL file (merged dataset)")
    parser.add_argument("--output-dir", default="data/mlx_training",
                        help="Output directory for train/valid/test splits")
    parser.add_argument("--model", default="models/mistral-7b-instruct-v0.3-4bit",
                        help="Path to model (for tokenizer)")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Max sequence length in tokens (filter longer)")
    parser.add_argument("--valid-ratio", type=float, default=0.05,
                        help="Fraction of data for validation (default: 0.05)")
    parser.add_argument("--test-ratio", type=float, default=0.05,
                        help="Fraction of data for testing (default: 0.05)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible splits")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        raise SystemExit(1)

    # Load tokenizer for exact length filtering
    print(f"Loading tokenizer from {args.model}...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Load and convert all examples
    examples = []
    skipped = 0
    skipped_lengths = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            chat = convert_to_chat_format(data["prompt"], data["completion"])

            # Filter by token length to prevent NaN during QLoRA training
            n_tokens = count_tokens_exact(chat["messages"], tokenizer)
            if n_tokens > args.max_tokens:
                skipped += 1
                skipped_lengths.append(n_tokens)
                continue

            examples.append(chat)

    print(f"Loaded {len(examples)} examples from {input_path}")
    if skipped:
        print(f"Filtered out {skipped} examples exceeding {args.max_tokens} tokens")
        print(f"  (lengths: {sorted(skipped_lengths)[:5]}{'...' if len(skipped_lengths) > 5 else ''})")

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(examples)

    n = len(examples)
    n_valid = int(n * args.valid_ratio)
    n_test = int(n * args.test_ratio)
    n_train = n - n_valid - n_test

    train_set = examples[:n_train]
    valid_set = examples[n_train:n_train + n_valid]
    test_set = examples[n_train + n_valid:]

    # Write splits
    for split_name, split_data in [("train", train_set), ("valid", valid_set), ("test", test_set)]:
        output_path = output_dir / f"{split_name}.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for item in split_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nSplit:")
    print(f"  Train: {n_train} examples ({n_train/n*100:.1f}%)")
    print(f"  Valid: {n_valid} examples ({n_valid/n*100:.1f}%)")
    print(f"  Test:  {n_test} examples ({n_test/n*100:.1f}%)")
    print(f"\nOutput directory: {output_dir}")

    # Show a sample to verify format
    print(f"\nSample training example:")
    sample = train_set[0]
    print(f"  User:   {sample['messages'][0]['content'][:100]}...")
    print(f"  Assist: {sample['messages'][1]['content'][:100]}...")


if __name__ == "__main__":
    main()
