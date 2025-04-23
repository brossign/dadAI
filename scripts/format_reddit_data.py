import json
from pathlib import Path

#{
#  "prompt": "Post Reddit : [Titre]\n\n[Contenu du post Reddit]\n\nRéponse du papa :",
#  "completion": "[Top commentaire]"
#}

# 📥 Fichier source dans data/
input_file = Path("data/reddit_dataset.jsonl")

# 📤 Fichier de sortie dans data/
output_file = Path("data/formatted_dataset.jsonl")

# 📦 Contiendra les exemples formatés en prompt/completion
formatted_data = []

# 🔄 Lecture des données Reddit
with input_file.open("r") as f:
    for line in f:
        try:
            item = json.loads(line)
            # 🧠 Structure de prompt + completion
            prompt = f"Post Reddit : {item['title'].strip()}\n\n{item['selftext'].strip()}\n\nRéponse du papa :"
            completion = item["comment"].strip()

            formatted_data.append({
                "prompt": prompt,
                "completion": completion
            })

        except Exception as e:
            print(f"❌ Erreur de parsing : {e}")

# 💾 Écriture dans le fichier final
with output_file.open("w") as f:
    for item in formatted_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ {len(formatted_data)} exemples formatés sauvegardés dans {output_file}")