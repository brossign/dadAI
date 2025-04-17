# Model directory – Mistral GGUF

This folder is used to store the quantized `.gguf` model file used by [LocalAI](https://github.com/go-skynet/LocalAI).

We recommend using one of the Mistral 7B Instruct models available on Hugging Face.

## 🔗 Example model (Q4_K_M)

You can download a quantized version of Mistral 7B Instruct from TheBloke's Hugging Face repo:

```bash
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf -P localai/models/
```