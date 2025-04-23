# DadAI – An LLM-based assistant for new dads 🤖👶

**DadAI** is an open-source project built to support new fathers during pregnancy and early parenthood.  
The idea is simple: provide emotionally intelligent, practical guidance powered by LLMs — and built on **Mistral 7B**.

## 🚀 Why DadAI?

Most resources around parenting are either mother-centric or scattered across forums.  
As a first-time dad, I realized how hard it can be to find support that's both practical and emotionally relevant — and I wanted to make things easier for other future dads.  

DadAI aims to provide a clear, AI-driven interface that supports:
- Emotional support during pregnancy
- Concrete actions and reminders
- Guidance on sleep, communication, and partner well-being

## 🧠 Tech Stack

- [x] Mistral 7B (quantized with GGUF)
- [x] LoRA fine-tuning with QLoRA
- [x] Python (Transformers, PEFT, Datasets)
- [x] Deployment via [LocalAI](https://github.com/go-skynet/LocalAI)
- [ ] Web interface (optional – future)

## 📂 Project Structure

```
dadAI/
├── data/               # Dataset from Reddit, BabyCenter, etc.
├── notebooks/          # Fine-tuning & inference notebooks
├── scripts/            # LoRA training 
├── requirements.txt    # Project dependencies
└── README.md
```

## 💬 Status

This project is currently in early development.  
Fine-tuning and inference will be tested on [RunPod](https://www.runpod.io/) using QLoRA and Mistral 7B.

## 📌 Goals

- Train an assistant on real parenting data (Reddit, BabyCenter)
- Optimize responses via QLoRA and test performance locally
- Package the assistant behind a simple OpenAI-compatible API

## 🧪 Local Inference with Mistral 7B (via LocalAI)

You can run DadAI locally using [LocalAI](https://github.com/go-skynet/LocalAI), an open-source alternative to the OpenAI API.

This setup uses the **Mistral 7B Instruct model** in GGUF format and exposes a local `/v1/chat/completions` endpoint, fully compatible with OpenAI’s API.

### 🧠 1. Download the model

We recommend downloading the `Q4_K_M` quantized version from [TheBloke's Hugging Face page](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF):

```bash
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf -P localai/models/
```

> ⚠️ This file is **not included in the repo** (see `localai/models/README.md` for details)

---

More to come soon:
- Fine-tuning instructions (QLoRA)
- LangChain agent with memory
- Streamlit chatbot

## 👤 Author

**Benoît Rossignol**  
📍 Geneva  
💼 Solution Architect  
🧠 AI & PreSales Leader
