"""
Step 2: Format raw Reddit data into training pairs.

Converts reddit_dataset.jsonl into formatted prompt/completion pairs
using Mistral's native [INST] chat template.

Fixes over v1:
- Uses Mistral [INST] format (was custom French template)
- All English (was mixed French/English)
- Preserves metadata (subreddit, scores) for filtering
- Validates required keys before processing
- Proper error handling with counts
"""

import json
import argparse
from pathlib import Path

# System prompt that defines DadAI's personality
SYSTEM_PROMPT = (
    "You are DadAI, a supportive and experienced father who gives advice "
    "to new dads. You respond with empathy, warmth, and practical wisdom — "
    "like a friend who's been through it all. You share personal anecdotes, "
    "normalize struggles, and never judge. You're not a therapist or doctor — "
    "just a dad who gets it."
)


def format_as_mistral_chat(title, selftext, comment):
    """
    Format a Reddit post/comment pair using Mistral's [INST] template.

    The prompt includes the system prompt + the dad's question.
    The completion is the supportive response.
    """
    # Build the user's question from the Reddit post
    if selftext and len(selftext.strip()) > 10:
        user_message = f"{title.strip()}\n\n{selftext.strip()}"
    else:
        user_message = title.strip()

    # Mistral [INST] format with system prompt
    prompt = f"[INST] {SYSTEM_PROMPT}\n\n{user_message} [/INST]"
    completion = comment.strip()

    return prompt, completion


def main():
    parser = argparse.ArgumentParser(description="Format Reddit data for training")
    parser.add_argument("--input", default="data/reddit_dataset.jsonl",
                        help="Input JSONL file (raw Reddit data)")
    parser.add_argument("--output", default="data/formatted_dataset.jsonl",
                        help="Output JSONL file (formatted pairs)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        raise SystemExit(1)

    formatted = 0
    skipped = 0
    errors = []

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:

        for line_num, line in enumerate(infile, start=1):
            try:
                item = json.loads(line)

                # Validate required keys
                required_keys = ["title", "comment"]
                missing = [k for k in required_keys if k not in item]
                if missing:
                    errors.append((line_num, f"Missing keys: {missing}"))
                    skipped += 1
                    continue

                # Skip if comment is too short (will be filtered more in clean step)
                if len(item["comment"].strip()) < 30:
                    skipped += 1
                    continue

                prompt, completion = format_as_mistral_chat(
                    title=item["title"],
                    selftext=item.get("selftext", ""),
                    comment=item["comment"],
                )

                output_record = {
                    "prompt": prompt,
                    "completion": completion,
                    # Preserve metadata for downstream filtering
                    "subreddit": item.get("subreddit", ""),
                    "post_score": item.get("post_score", item.get("score", 0)),
                    "comment_score": item.get("comment_score", 0),
                }

                outfile.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                formatted += 1

            except json.JSONDecodeError as e:
                errors.append((line_num, f"Invalid JSON: {e}"))
                skipped += 1
            except Exception as e:
                errors.append((line_num, f"Error: {e}"))
                skipped += 1

    print(f"Done! {formatted} examples formatted, {skipped} skipped.")
    print(f"Output: {output_path}")

    if errors:
        print(f"\nFirst {min(5, len(errors))} errors:")
        for line_num, err in errors[:5]:
            print(f"  Line {line_num}: {err}")


if __name__ == "__main__":
    main()
