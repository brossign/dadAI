"""
Show random samples from the cleaned dataset.

Useful for quick quality checks during data pipeline development.
"""

import json
import random
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Show random samples from dataset")
    parser.add_argument("--file", default="data/cleaned_dataset.jsonl",
                        help="JSONL file to sample from")
    parser.add_argument("--count", type=int, default=3,
                        help="Number of samples to show")
    args = parser.parse_args()

    filepath = Path(args.file)
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        raise SystemExit(1)

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        print("Error: File is empty.")
        raise SystemExit(1)

    count = min(args.count, len(lines))
    samples = random.sample(lines, count)

    print(f"Showing {count} random sample(s) from {filepath} ({len(lines)} total)\n")
    print("=" * 70)

    for i, line in enumerate(samples, start=1):
        example = json.loads(line)
        print(f"\n--- Sample {i} ---\n")
        print(f"PROMPT ({len(example['prompt'])} chars):")
        print(example["prompt"][:500])
        if len(example["prompt"]) > 500:
            print("  [... truncated]")
        print(f"\nCOMPLETION ({len(example['completion'])} chars):")
        print(example["completion"][:500])
        if len(example["completion"]) > 500:
            print("  [... truncated]")
        print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
