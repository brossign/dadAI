# 🧼 Script : clean_dataset.py - etape 3
# Objectif : Nettoyer le dataset en supprimant les lignes trop courtes ou non humaines

import json

# 📁 Chemins des fichiers
input_file = "data/formatted_dataset.jsonl"
output_file = "data/cleaned_dataset.jsonl"

# 🧮 Compteurs
kept, skipped = 0, 0

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        try:
            data = json.loads(line)
            prompt = data.get("prompt", "").strip()
            completion = data.get("completion", "").strip()

            # ❌ Trop court = rejeté
            if len(prompt) < 50 or len(completion) < 50:
                skipped += 1
                continue

            # ❌ Modération automatique ou bot = rejeté
            if ("i am a bot" in completion.lower() or
                "moderator" in completion.lower()):
                skipped += 1
                continue

            # ✅ Ligne valide
            outfile.write(json.dumps({
                "prompt": prompt,
                "completion": completion
            }) + "\n")
            kept += 1

        except json.JSONDecodeError:
            skipped += 1

print(f"✅ {kept} lignes conservées, ❌ {skipped} lignes supprimées.")
