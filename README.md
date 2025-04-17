# DadAI – An LLM-based assistant for new dads 🤖👶

**DadAI** is an open-source project built to support new fathers during pregnancy and early parenthood.  
The idea is simple: provide emotionally intelligent, practical guidance powered by LLMs — and built on **Mistral 7B**.

## 🚀 Why DadAI?

Most resources around parenting are either mother-centric or scattered across forums.  
As a first-time dad, I realized how hard it can be to find support that's both practical and emotionally relevant, and I wanted to make things easier for other future dads.  

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

## 🧪 Local Inference via LocalAI

You can run DadAI locally using [LocalAI](https://github.com/go-skynet/LocalAI) and a quantized Mistral 7B model.

### 1. Download the model (GGUF format)
You can use one of the following (e.g. from TheBloke on Hugging Face):

```bash
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf -P models/
```

## 📂 Structure

```bash
dadAI/
├── data/               # Dataset from Reddit, BabyCenter, etc.
├── notebooks/          # Fine-tuning & inference notebooks
├── scripts/            # LoRA training and LocalAI deployment scripts
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

## 👤 Author

**Benoît Rossignol**  
📍 Geneva  
💼 Solution Architect  
🧠 AI & PreSales Leader
