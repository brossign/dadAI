# scripts/show_random_sample.py

import json
import random

with open("data/formatted_dataset.jsonl", "r", encoding="utf-8") as f:
    lines = f.readlines()

example = json.loads(random.choice(lines))

print("\n🎯 Exemple aléatoire :\n")
print("📝 Prompt :")
print(example["prompt"])
print("\n💬 Completion :")
print(example["completion"])
