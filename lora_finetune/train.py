import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset
from datetime import datetime

# 🕓 Mesure du temps total
start_time = time.time()

try:
    # 📁 Chemin vers les données HF (créées par prepare_dataset.py)
    data_path = "../data/cleaned_dataset.jsonl"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Dataset introuvable à : {data_path}")

    # 🧠 Chargement du tokenizer
    model_name = "mistralai/Mistral-7B-v0.1"
    print("🔡 Chargement du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 📦 Chargement du dataset
    print("📦 Chargement du dataset...")
    dataset = load_dataset("json", data_files=data_path, split="train")

    # ✂️ Tokenisation
    def tokenize(example):
        return tokenizer(
            example["prompt"],
            text_target=example["completion"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )

    print("✂️ Tokenisation en cours...")
    dataset = dataset.map(tokenize, remove_columns=["prompt", "completion"])

    # ⚙️ Configuration LoRA
    print("⚙️ Configuration de LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # 🚀 Chargement du modèle en 4-bit
    print("🚀 Chargement du modèle 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        device_map="auto"
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    # 📚 Data collator
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # 🏁 Paramètres d'entraînement
    run_name = f"dadAI-lora-{datetime.now().strftime('%Y%m%d-%H%M')}"
    training_args = TrainingArguments(
        output_dir="outputs",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_dir="logs",
        logging_steps=10,
        save_strategy="epoch",
        bf16=True if torch.cuda.is_bf16_supported() else False,
        report_to="none",
        run_name=run_name
    )

    # 📊 Entraîneur HF
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=collator
    )

    # ✅ Lancement entraînement
    print("🚀 Entraînement en cours...")
    trainer.train()

    # 💾 Sauvegarde des poids LoRA
    print("💾 Sauvegarde des poids LoRA...")
    model.save_pretrained("outputs/lora_weights")

    elapsed = time.time() - start_time
    print(f"✅ Entraînement terminé en {elapsed:.2f} secondes.")

except Exception as e:
    print(f"❌ Erreur pendant le fine-tuning : {e}")