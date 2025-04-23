import json
from pathlib import Path

# 📁 Chemin du fichier à valider
dataset_path = Path("data/formatted_dataset.jsonl").resolve()

# 📊 Statistiques
total = 0
valid = 0
invalid = 0
errors = []

print(f"📂 Validation du dataset : {dataset_path.resolve()}\n")

with open(dataset_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, start=1):
        total += 1
        try:
            # ✅ Parse chaque ligne comme un JSON
            data = json.loads(line)

            # ✅ Vérifie les clés
            if "prompt" not in data or "completion" not in data:
                raise ValueError("Clés manquantes ('prompt' ou 'completion')")

            # ✅ Vérifie que ce n’est pas vide
            if not data["prompt"].strip() or not data["completion"].strip():
                raise ValueError("Prompt ou completion vide")

            valid += 1

        except Exception as e:
            invalid += 1
            errors.append((i, str(e)))

# 📊 Résumé
print(f"✅ Lignes valides : {valid}")
print(f"❌ Lignes invalides : {invalid}")
print(f"📈 Total : {total}\n")

# 🐞 Détail des erreurs (limité à 5 pour pas tout spammer)
if errors:
    print("🔍 Exemples d’erreurs :")
    for i, (line_number, err) in enumerate(errors[:5], start=1):
        print(f"  {i}. Ligne {line_number} → {err}")