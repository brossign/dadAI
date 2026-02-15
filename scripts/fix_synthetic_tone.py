"""
Fix synthetic data tone to better match Reddit dad voice.

Issues found:
1. Em-dashes (—) used in 63% of v3.1 vs 0.9% of Reddit — AI writing tell
2. Personal voice (I/my) in 19.5% vs 67.2% of Reddit — needs more dad stories

This script:
- Replaces most em-dashes with periods, commas, or regular dashes
- Reports how many responses still lack personal voice (manual review needed)
"""

import json
import re
import random

random.seed(42)


def fix_em_dashes(text):
    """Replace em-dashes with more Reddit-natural punctuation.

    Patterns:
      "word — word"  → "word. Word" or "word - word" (randomly)
      "word —"       → "word."
    """
    # Pattern: " — " between words (most common)
    def replace_dash(match):
        before = match.group(1)
        after = match.group(2)

        # If after starts lowercase, use " - " (casual aside)
        # If natural sentence break, use ". " with capitalized next word
        roll = random.random()
        if roll < 0.45:
            # Period + capitalize
            return f"{before}. {after[0].upper()}{after[1:]}" if len(after) > 1 else f"{before}. {after.upper()}"
        elif roll < 0.80:
            # Regular dash
            return f"{before} - {after}"
        else:
            # Comma (for parenthetical asides)
            return f"{before}, {after}"

    # Replace " — " pattern
    text = re.sub(r'(\w) — (\w)', replace_dash, text)

    # Remaining standalone em-dashes
    text = text.replace(" — ", ". ")
    text = text.replace("— ", "")
    text = text.replace(" —", ".")

    return text


def check_personal_voice(text):
    """Check if text uses personal 'I/my' language like a real dad."""
    lower = text.lower()
    markers = [" i ", " i'", "my ", "i was", "i had", "i've", "i did",
               "my kid", "my son", "my daughter", "my wife", "my buddy",
               "i remember", "i know", "i felt", "i went", "i think"]
    return any(m in lower for m in markers)


def main():
    # Fix v3.1 synthetic
    input_path = "data/synthetic_v31_pairs.jsonl"
    output_path = "data/synthetic_v31_pairs.jsonl"  # overwrite

    records = []
    with open(input_path) as f:
        for line in f:
            records.append(json.loads(line))

    # Stats before
    em_before = sum(1 for r in records if "—" in r["completion"])
    personal_before = sum(1 for r in records if check_personal_voice(r["completion"]))

    # Fix em-dashes in completions
    for r in records:
        r["completion"] = fix_em_dashes(r["completion"])

    # Stats after
    em_after = sum(1 for r in records if "—" in r["completion"])
    personal_after = sum(1 for r in records if check_personal_voice(r["completion"]))

    # Also fix v2 synthetic
    v2_path = "data/synthetic_gap_topics.jsonl"
    v2_records = []
    with open(v2_path) as f:
        for line in f:
            v2_records.append(json.loads(line))

    v2_em_before = sum(1 for r in v2_records if "—" in r["completion"])
    for r in v2_records:
        r["completion"] = fix_em_dashes(r["completion"])
    v2_em_after = sum(1 for r in v2_records if "—" in r["completion"])

    # Write fixed files
    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(v2_path, "w") as f:
        for r in v2_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(records)
    print("V3.1 Synthetic Fixes:")
    print(f"  Em-dashes:    {em_before}/{n} ({em_before/n*100:.1f}%) → {em_after}/{n} ({em_after/n*100:.1f}%)")
    print(f"  Personal (I/my): {personal_before}/{n} ({personal_before/n*100:.1f}%) → {personal_after}/{n} ({personal_after/n*100:.1f}%)")
    print()

    n2 = len(v2_records)
    print("V2 Synthetic Fixes:")
    print(f"  Em-dashes:    {v2_em_before}/{n2} ({v2_em_before/n2*100:.1f}%) → {v2_em_after}/{n2} ({v2_em_after/n2*100:.1f}%)")
    print()

    # Show responses that LACK personal voice (for manual review)
    no_voice = [r for r in records if not check_personal_voice(r["completion"])]
    print(f"Responses WITHOUT personal voice: {len(no_voice)}/{n} ({len(no_voice)/n*100:.1f}%)")
    print("\nSample responses WITHOUT personal 'I/my' (may be fine for direct advice style):")
    for r in random.sample(no_voice, min(5, len(no_voice))):
        comp = r["completion"][:200]
        print(f"\n  '{comp}...'")


if __name__ == "__main__":
    main()
