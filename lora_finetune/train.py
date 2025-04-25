import os
import torch
import logging
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset
from datetime import datetime

# ========================
# 📓 CONFIGURATION
# ========================
data_path = "../data/cleaned_dataset.jsonl"
model_name = "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"
run_name = f"dadAI-lora-{datetime.now().strftime('%Y%m%d-%H%M')}"
output_dir = "outputs"
log_file_path = os.path.join(output_dir, f"{run_name}.log")
timeout_minutes = 50  # Safety timeout

# ========================
# 📜 LOGGING SETUP
# ========================
os.makedirs(output_dir, exist_ok=True)
logging.basicConfig(
    filename=log_file_path,
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger()

try:
    logger.info("📦 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    logger.info("📦 Loading dataset...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    dataset = load_dataset("json", data_files=data_path, split="train")

    def tokenize(example):
        return tokenizer(
            example["prompt"],
            text_target=example["completion"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )

    logger.info("🔄 Tokenizing dataset...")
    dataset = dataset.map(tokenize, remove_columns=["prompt", "completion"])

    logger.info("⚙️ Configuring LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    logger.info("🚀 Loading base model in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    logger.info("📋 Preparing training arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=collator
    )

    # ========================
    # 🚀 TRAINING
    # ========================
    logger.info("🚀 Starting training...")
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    training_duration = end_time - start_time

    if training_duration > timeout_minutes * 60:
        logger.warning(f"⚠️ Training exceeded {timeout_minutes} minutes limit!")

    # ========================
    # 💾 SAVE LoRA WEIGHTS
    # ========================
    model.save_pretrained(os.path.join(output_dir, "lora_weights"))
    logger.info("✅ Training complete.")
    logger.info(f"💾 LoRA weights saved to {os.path.join(output_dir, 'lora_weights')}")
    logger.info(f"🕒 Training duration: {training_duration:.2f} seconds (~{training_duration / 60:.2f} min)")

    print("✅ Training finished. Summary:")
    print(f"   - Checkpoints: {output_dir}/lora_weights")
    print(f"   - Logs: {log_file_path}")
    print(f"   - Duration: {training_duration / 60:.2f} minutes")

except Exception as e:
    logger.error(f"❌ Training failed: {e}")
    print(f"❌ An error occurred: {e}")