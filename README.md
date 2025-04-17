# DadAI – An LLM-based assistant for new dads 🤖👶

**DadAI** is an open-source project built to support new fathers during pregnancy and early parenthood.  
The idea is simple: provide emotionally intelligent, practical guidance powered by LLMs — and built on **Mistral 7B**.

## 🚀 Why DadAI?

Most resources around parenting are either mother-centric or scattered across forums.  
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

## 📂 Structure

```bash
dadAI/
├── data/               # Dataset from Reddit, BabyCenter, etc.
├── notebooks/          # Fine-tuning & inference notebooks
├── scripts/            # LoRA training and LocalAI deployment scripts
├── requirements.txt    # Project dependencies
└── README.md
