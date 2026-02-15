"""Quick tone comparison: Reddit vs v2 synthetic vs v4 synthetic."""

import json
import random

random.seed(42)

# Load datasets
reddit = []
with open("data/cleaned_dataset.jsonl") as f:
    for line in f:
        reddit.append(json.loads(line))

v2_synth = []
with open("data/synthetic_gap_topics.jsonl") as f:
    for line in f:
        v2_synth.append(json.loads(line))

v31_synth = []
with open("data/synthetic_v31_pairs.jsonl") as f:
    for line in f:
        v31_synth.append(json.loads(line))

print("=" * 70)
print("REAL REDDIT RESPONSES (5 random samples)")
print("=" * 70)
for ex in random.sample(reddit, 5):
    comp = ex.get("completion", "")
    print(f"\n--- (len={len(comp)} chars) ---")
    print(comp[:500])

print()
print("=" * 70)
print("V2 SYNTHETIC RESPONSES (3 random samples)")
print("=" * 70)
for ex in random.sample(v2_synth, 3):
    comp = ex.get("completion", "")
    print(f"\n--- (len={len(comp)} chars) ---")
    print(comp[:500])

print()
print("=" * 70)
print("V3.1 SYNTHETIC RESPONSES (5 random samples)")
print("=" * 70)
for ex in random.sample(v31_synth, 5):
    comp = ex.get("completion", "")
    print(f"\n--- (len={len(comp)} chars) ---")
    print(comp[:500])

# Stats comparison
print()
print("=" * 70)
print("STYLE ANALYSIS")
print("=" * 70)

def analyze(name, data):
    lengths = [len(d.get("completion", "")) for d in data]
    texts = [d.get("completion", "") for d in data]

    # Count style markers
    em_dashes = sum(1 for t in texts if "—" in t)
    contractions = sum(1 for t in texts if "don't" in t.lower() or "can't" in t.lower() or "won't" in t.lower() or "I'm" in t or "you're" in t.lower())
    questions = sum(1 for t in texts if "?" in t)
    personal = sum(1 for t in texts if " I " in t or "my " in t.lower())
    formal_words = sum(1 for t in texts if "furthermore" in t.lower() or "however" in t.lower() or "additionally" in t.lower() or "moreover" in t.lower())
    bullet_points = sum(1 for t in texts if "1)" in t or "1." in t or "- " in t)

    n = len(data)
    print(f"\n{name} ({n} examples)")
    print(f"  Avg length:       {sum(lengths)/n:.0f} chars")
    print(f"  Min/Max:          {min(lengths)} / {max(lengths)} chars")
    print(f"  Em-dashes:        {em_dashes/n*100:.1f}% of responses")
    print(f"  Contractions:     {contractions/n*100:.1f}% of responses")
    print(f"  Questions asked:  {questions/n*100:.1f}% of responses")
    print(f"  Personal (I/my):  {personal/n*100:.1f}% of responses")
    print(f"  Formal language:  {formal_words/n*100:.1f}% of responses")
    print(f"  Lists/bullets:    {bullet_points/n*100:.1f}% of responses")

analyze("Reddit (real)", reddit)
analyze("V2 Synthetic", v2_synth)
analyze("V3.1 Synthetic", v31_synth)
