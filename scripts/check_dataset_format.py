"""
Step 4: Validate the cleaned dataset.

Checks that every line is valid JSON with non-empty prompt and completion.
Reports statistics on dataset quality (lengths, distributions).
Exits with code 1 if any invalid lines are found.
"""

import json
import sys
import statistics
from pathlib import Path


def main():
    dataset_path = Path("data/cleaned_dataset.jsonl")

    if not dataset_path.exists():
        print(f"Error: File not found: {dataset_path}")
        sys.exit(1)

    total = 0
    valid = 0
    invalid = 0
    errors = []
    prompt_lengths = []
    completion_lengths = []

    print(f"Validating: {dataset_path}\n")

    with open(dataset_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            total += 1
            try:
                data = json.loads(line)

                if "prompt" not in data or "completion" not in data:
                    raise ValueError("Missing 'prompt' or 'completion' key")

                if not data["prompt"].strip() or not data["completion"].strip():
                    raise ValueError("Empty prompt or completion")

                valid += 1
                prompt_lengths.append(len(data["prompt"]))
                completion_lengths.append(len(data["completion"]))

            except Exception as e:
                invalid += 1
                errors.append((i, str(e)))

    # Summary
    print(f"Total lines:   {total}")
    print(f"Valid:         {valid}")
    print(f"Invalid:       {invalid}")

    if valid > 0:
        print(f"\nPrompt lengths:")
        print(f"  Min: {min(prompt_lengths)}, Max: {max(prompt_lengths)}, "
              f"Median: {statistics.median(prompt_lengths):.0f}, "
              f"Mean: {statistics.mean(prompt_lengths):.0f}")
        print(f"\nCompletion lengths:")
        print(f"  Min: {min(completion_lengths)}, Max: {max(completion_lengths)}, "
              f"Median: {statistics.median(completion_lengths):.0f}, "
              f"Mean: {statistics.mean(completion_lengths):.0f}")

    if errors:
        print(f"\nFirst {min(5, len(errors))} errors:")
        for line_num, err in errors[:5]:
            print(f"  Line {line_num}: {err}")

    if invalid > 0:
        sys.exit(1)
    else:
        print("\nAll lines valid!")


if __name__ == "__main__":
    main()
