import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Goal: Merge LoRA weight + Model and convert it to GGUF to be deployed in LocalAI
# Pre-requisite: you need a float16 not quantified model 

# ========================
# 📓 CONFIGURATION
# ========================
# Model paths
base_model = "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"
lora_weights_path = "outputs/lora_weights"
merged_model_path = "outputs/merged_model"

gguf_output_path = "outputs/gguf_model"
os.makedirs(merged_model_path, exist_ok=True)
os.makedirs(gguf_output_path, exist_ok=True)

# ========================
# 🚀 Merge base + LoRA
# ========================
try:
    print("🔄 Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    print("🔗 Loading LoRA weights...")
    model = PeftModel.from_pretrained(model, lora_weights_path)

    print("🔀 Merging LoRA into base model...")
    model = model.merge_and_unload()

    print(f"💾 Saving merged model to {merged_model_path}...")
    model.save_pretrained(merged_model_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_model_path)

    print("✅ Merge completed.")
except Exception as e:
    print(f"❌ Merge failed: {e}")

# ========================
# 🌀 Convert to GGUF
# ========================
try:
    print("🌀 Converting merged model to GGUF...")

    # Utilisation de huggingface-cli pour convertir facilement
    # S'assurer que huggingface-cli est bien installé
    command = f"python3 -m transformers.convert --model_name_or_path {merged_model_path} --output_dir {gguf_output_path} --trust_remote_code"
    exit_code = os.system(command)

    if exit_code != 0:
        raise Exception("Conversion command failed.")

    print(f"✅ GGUF model saved to {gguf_output_path}")
except Exception as e:
    print(f"❌ GGUF conversion failed: {e}")
