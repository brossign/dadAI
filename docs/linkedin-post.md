# LinkedIn Post — DadAI v2

---

Last year I built DadAI — a fine-tuned LLM to support new dads. Got some attention, RunPod featured it, I wrote about it. Then life happened. New job, new country, no time.

A few months later I got curious: what would happen if I asked Claude to audit the code?

It found 5 critical bugs. The model had never actually learned anything.

Here's what was wrong:
- The tokenization was broken — the model was training on the prompts, not the answers
- The chat template was different between training and inference
- 30% of the Reddit data was bot-generated noise
- The deployment pipeline (GPTQ to GGUF to LocalAI) never actually worked
- The dataset had only 298 noisy examples

So over one weekend, Claude and I rebuilt everything:

- Replaced RunPod ($5/run) with Apple MLX on my MacBook Pro M1 (free)
- Fixed the data pipeline: 2,147 curated examples from 7 subreddits + synthetic data for under-covered topics (dad mental health, single dads, grief)
- Trained with QLoRA locally — peak memory 7 GB, Mac stayed usable the whole time
- Hit a NaN gradient explosion at iteration 60 (long sequences + 4-bit quantization). Fixed by pre-filtering sequences > 2048 tokens
- Iteration 400 beat iteration 1000 in A/B testing. More training isn't always better
- Built a Gradio chat UI with streaming responses
- Deployed to Hugging Face Spaces

The result? DadAI now actually works. It responds like a real dad who's been through it — with empathy, humor, and practical advice.

Try it: https://huggingface.co/spaces/benlongi/DadAI
Code: https://github.com/brossign/dadAI

The biggest lesson? I had a project that got attention, got featured, but fundamentally didn't work. It took an AI auditing another AI's training code to find out why.

The second biggest lesson? You don't need a cloud GPU anymore. A MacBook and a weekend is enough to fine-tune a 7B model.

#AI #LLM #MachineLearning #FineTuning #MLX #AppleSilicon #Fatherhood #OpenSource #BuildInPublic

---
