import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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

# ========================
# 🚀 Load Model + Tokenizer
# ========================
try:
    print("🔄 Loading base model in 4-bit...")
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
# 💬 Inference Function
# ========================
def generate_reply(user_post):
    prompt = f"Post Reddit : {user_post}\n\nDad's reply:"
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

# ========================
# 🎯 Main Inference Loop
# ========================
try:
    print("🧠 DadAI is ready! (Ctrl+C to exit)")
    while True:
        user_input = input("\n✏️  Your Reddit post: ").strip()

        if not user_input:
            print("⚠️ Please enter a non-empty prompt.")
            continue

        dad_reply = generate_reply(user_input)
        print("\n🤖 DadAI's reply:\n" + "-"*50)
        print(dad_reply)
        print("-"*50)

except KeyboardInterrupt:
    print("\n👋 Exiting DadAI. See you soon!")
except Exception as e:
    print(f"❌ Error during inference: {e}")