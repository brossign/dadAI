"""Find the most 'therapist-sounding' synthetic pairs that lack personal voice."""

import json

records = []
with open("data/synthetic_v31_pairs.jsonl") as f:
    for idx, line in enumerate(f):
        r = json.loads(line)
        records.append((idx, r))

# Score each response: lower = more therapist, higher = more dad
def dad_score(text):
    t = text.lower()
    score = 0
    # Personal pronouns (dad sharing experience)
    for marker in [" i ", " i'", "my kid", "my son", "my daughter", "my wife",
                   "my buddy", "i remember", "i was", "i had", "i've been",
                   "i went", "i felt", "when i", "for me"]:
        score += t.count(marker) * 2
    # Casual markers
    for marker in ["honestly", "dude", "man", "literally", "seriously",
                   "the real talk", "here's the thing", "look,", "trust me"]:
        score += t.count(marker)
    # Therapist/expert markers (negative)
    for marker in ["research shows", "developmentally", "cognitive",
                   "self-regulation", "attachment behavior", "prefrontal cortex",
                   "clinical", "evidence-based", "studies show", "neurological",
                   "body autonomy", "emotional literacy", "generational"]:
        score -= 3
    # Generic advice without personal stake
    if " i " not in t and "my " not in t:
        score -= 5
    return score

scored = [(dad_score(r["completion"]), idx, r) for idx, r in records]
scored.sort()

print("TOP 25 MOST THERAPIST-SOUNDING (lowest dad score):\n")
for score, idx, r in scored[:25]:
    # Extract the question from the prompt
    prompt = r["prompt"]
    q_start = prompt.find("[/INST]")
    if q_start == -1:
        q_start = prompt.find("\n\n") + 2
    else:
        q_start = prompt.rfind("\n\n", 0, q_start) + 2

    comp = r["completion"][:150].replace("\n", " ")
    print(f"[{idx:3d}] score={score:3d} | {comp}...")
    print()
