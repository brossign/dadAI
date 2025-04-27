import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import os

# Goal: Mistrak7B merged with my LoRA weights to be publish as a standalone model in Hugging Face

# ======================
# 📦 CONFIG
# ======================
base_model_name = "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"
lora_weights_path = "outputs/lora_weights"
merged_output_dir = "outputs/merged_model"

os.makedirs(merged_output_dir, exist_ok=True)

# ======================
# 🔥 Load Model
# ======================
print("🔄 Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

# Load LoRA
print("🔗 Loading LoRA weights...")
model = PeftModel.from_pretrained(model, lora_weights_path)
# Merge LoRA & Model
model = model.merge_and_unload()

# ======================
# 💾 Save the full merged model
# Full model .safetensors + config.json + tokenizer.json inside outputs/merged_model/
# ======================
print("💾 Saving merged model...")
model.save_pretrained(merged_output_dir, safe_serialization=True)
tokenizer.save_pretrained(merged_output_dir)

print(f"✅ Merged model saved to {merged_output_dir}")