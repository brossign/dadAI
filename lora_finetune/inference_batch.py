import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys

# ========================
# 📦 CONFIGURATION
# ========================
base_model = "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"
lora_weights_path = "outputs/lora_weights"
max_new_tokens = 150
temperature = 0.7
top_p = 0.9

test_prompts = [
    "My baby keeps crying when I put her down. Any advice?",
    "I yelled at my kid today. Feeling horrible.",
    "How do you keep your marriage strong after having kids?",
    "My 3 y/o won't eat anything but nuggets. Help?",
    "I just need a break. Is it bad to want one?"
]

# ========================
# 🚀 Load Model + Tokenizer
# ========================
try:
    print("🔄 Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        trust_remote_code=True
    )

    print("🔗 Loading LoRA fine-tuned weights...")
    model = PeftModel.from_pretrained(model, lora_weights_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    print("✅ Model and LoRA weights loaded successfully.\n")
except Exception as e:
    print(f"❌ Failed to load model or LoRA: {e}")
    sys.exit(1)

# ========================
# 🔥 Batch Inference
# ========================
def generate_reply(prompt_text):
    prompt = f"Post Reddit : {prompt_text}\n\nDad's reply:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )

    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return reply.split("Dad's reply:")[-1].strip()

try:
    print(f"🧪 Running batch inference on {len(test_prompts)} prompts...\n")
    
    for idx, prompt_text in enumerate(test_prompts, 1):
        print(f"📨 Prompt {idx}: {prompt_text}")
        dad_reply = generate_reply(prompt_text)
        print(f"🤖 DadAI's reply:\n{dad_reply}")
        print("-" * 60)

except Exception as e:
    print(f"❌ Error during batch inference: {e}")