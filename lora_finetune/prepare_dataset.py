# prepare_dataset.py

from pathlib import Path
import json
from datasets import Dataset

# 📁 Chemin vers le fichier source .jsonl
source_path = Path("../data/cleaned_dataset.jsonl")

# 📁 Chemin de sortie où sera sauvegardé le dataset Hugging Face
output_path = Path("lora_finetune/data")

# 🔧 Lecture du .jsonl ligne par ligne
data = []
with open(source_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        data.append(item)

# ✅ Création du dataset Hugging Face
hf_dataset = Dataset.from_list(data)

# 💾 Sauvegarde sur disque dans un format optimisé (Arrow)
hf_dataset.save_to_disk(str(output_path))

# 🔍 Exemple : affichage de la première ligne
print("✅ Exemple :")
print(hf_dataset[0])