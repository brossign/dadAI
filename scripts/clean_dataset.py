"""
Step 3: Clean and filter the formatted dataset.

Removes low-quality entries: short responses, bot content,
duplicates, URL-only comments, and low-effort replies.

Fixes over v1:
- Much broader bot/spam detection
- Filters [deleted], [removed], URL-only comments
- Filters low-effort responses ("this", "same", etc.)
- Deduplicates by completion text
- Configurable thresholds via CLI args
- Reports detailed statistics
- ensure_ascii=False for proper Unicode handling
"""

import json
import re
import argparse
import sys
from pathlib import Path

# Low-effort response patterns (case-insensitive)
LOW_EFFORT_PATTERNS = [
    r"^(this|same|lol|lmao|haha|congrats|congratulations)\s*[.!]*$",
    r"^(thanks|thank you|ty|thx)\s*[.!]*$",
    r"^(yes|no|yep|nope|agreed|exactly)\s*[.!]*$",
    r"^f$",  # "F" in the chat
    r"^nice\s*[.!]*$",
]

# Bot and automated content patterns
BOT_PATTERNS = [
    "i am a bot",
    "i'm a bot",
    "this action was performed automatically",
    "automoderator",
    "remindmebot",
    "bot action",
    "this is an automated",
    "if you have any questions or concerns",
    "please contact the moderators",
    "moderation team",
]

REMOVED_CONTENT = {"[deleted]", "[removed]", "[unavailable]"}


def is_mostly_urls(text):
    """Check if the text is primarily URLs with little other content."""
    urls = re.findall(r'https?://\S+', text)
    if not urls:
        return False
    text_without_urls = re.sub(r'https?://\S+', '', text).strip()
    return len(text_without_urls) < 50


def is_low_quality(completion, min_length=100):
    """
    Check if a completion is low quality.

    Returns (is_low_quality: bool, reason: str)
    """
    text = completion.strip()
    lower = text.lower()

    # Removed/deleted
    if lower in {s.lower() for s in REMOVED_CONTENT}:
        return True, "removed/deleted"

    # Too short
    if len(text) < min_length:
        return True, f"too short ({len(text)} chars)"

    # Bot patterns
    for pattern in BOT_PATTERNS:
        if pattern in lower:
            return True, f"bot pattern: {pattern}"

    # Low-effort patterns
    for pattern in LOW_EFFORT_PATTERNS:
        if re.match(pattern, lower):
            return True, "low-effort response"

    # Mostly URLs
    if is_mostly_urls(text):
        return True, "mostly URLs"

    return False, ""


def main():
    parser = argparse.ArgumentParser(description="Clean the formatted dataset")
    parser.add_argument("--input", default="data/formatted_dataset.jsonl",
                        help="Input JSONL file")
    parser.add_argument("--output", default="data/cleaned_dataset.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--min-completion-length", type=int, default=100,
                        help="Minimum completion length in characters")
    parser.add_argument("--min-prompt-length", type=int, default=50,
                        help="Minimum prompt length in characters")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    kept = 0
    skipped = 0
    seen_completions = set()
    skip_reasons = {}

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:

        for line in infile:
            try:
                data = json.loads(line)
                prompt = data.get("prompt", "").strip()
                completion = data.get("completion", "").strip()

                # Check prompt length
                if len(prompt) < args.min_prompt_length:
                    reason = "prompt too short"
                    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                    skipped += 1
                    continue

                # Check completion quality
                is_bad, reason = is_low_quality(completion, args.min_completion_length)
                if is_bad:
                    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                    skipped += 1
                    continue

                # Deduplicate by completion text (normalized)
                norm_completion = completion.lower().strip()
                if norm_completion in seen_completions:
                    reason = "duplicate"
                    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                    skipped += 1
                    continue
                seen_completions.add(norm_completion)

                # Write clean record (only prompt + completion for training)
                outfile.write(json.dumps({
                    "prompt": prompt,
                    "completion": completion,
                }, ensure_ascii=False) + "\n")
                kept += 1

            except json.JSONDecodeError:
                skip_reasons["invalid JSON"] = skip_reasons.get("invalid JSON", 0) + 1
                skipped += 1

    # Report
    print(f"Done! {kept} kept, {skipped} removed.")
    print(f"Output: {output_path}")

    if skip_reasons:
        print(f"\nRemoval reasons:")
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    # Exit with error if nothing was kept
    if kept == 0:
        print("\nError: No data survived cleaning!")
        sys.exit(1)


if __name__ == "__main__":
    main()
