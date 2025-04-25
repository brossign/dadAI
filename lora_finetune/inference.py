import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import sys

# Load base model + LoRA weights
# Test 1 custom prompt
# Display the generated Reddit-style reply

# ========================
# 📦 CONFIG
# ========================
base_model = "mistralai/Mistral-7B-v0.1"
lora_weights_path = "outputs/lora_weights"

# ========================
# 🚀 Load Model + Tokenizer
# ========================
try:
    print("🔄 Loading base model in 4-bit...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb_config, device_map="auto")
    
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
# 💬 Inference Loop
# ========================
try:
    print("🧠 Ready! Type your Reddit-style post (or Ctrl+C to exit).")
    while True:
        user_input = input("\nPost Reddit ✏️ : ")

        if not user_input.strip():
            print("⚠️ Please enter a non-empty prompt.")
            continue

        prompt = f"Post Reddit : {user_input}\n\nDad's reply:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )

        reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\n🤖 DadAI's reply:")
        print(reply.split("Dad's reply:")[-1].strip())

except KeyboardInterrupt:
    print("\n👋 Exiting. Bye!")
except Exception as e:
    print(f"❌ Error during inference: {e}")
