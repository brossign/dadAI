# How Claude Found 5 Critical Bugs in My AI Project — and We Fixed Everything in a Weekend

*A story about building DadAI v2: from broken fine-tuning on RunPod to a working LLM on a MacBook.*

---

## The backstory

In April 2025, I built **DadAI** — an LLM fine-tuned on Reddit parenting conversations to support new dads. The idea was simple: take real questions from forums like r/NewDads and r/Daddit, pair them with the best answers, and fine-tune Mistral 7B to respond with the empathy and wisdom of a dad who's been through it.

It got some traction. RunPod [featured it on their blog](https://www.runpod.io/blog/solo-dev-ai-for-dads-runpod). I wrote a [LinkedIn article](https://www.linkedin.com/pulse/how-i-built-fine-tuned-my-own-dadai-what-taught-me-llms-rossignol-pq7he/) about the journey. People liked the concept.

But there was a problem I didn't know about: **the model had never actually learned anything.**

The fine-tuned version produced outputs that were barely distinguishable from the base model. I chalked it up to "not enough data" and moved on to a new job at Shopify, a move to France, and the daily realities of being the dad I was trying to help other dads become.

## The audit

Fast-forward to February 2026. I'm settled in, and I get curious. Not about doing it all over again manually — but about what would happen if I asked **Claude** (via Cursor) to look at the code.

So I cloned the repo and said: *"Audit my code and share feedback."*

What came back was brutal. Claude found **5 critical bugs** — any one of which would have prevented the model from learning:

### Bug 1: The tokenization was wrong

The training script used HuggingFace's `Trainer`, but the way labels were constructed meant the model was **training on the prompts**, not on the completions. The loss was computed over the instructions ("You are DadAI, a supportive father...") instead of the actual dad responses. The model literally practiced repeating the system prompt thousands of times.

### Bug 2: Prompt template mismatch

Training used Mistral's `[INST]` chat format. Inference used a different format. The model never saw during inference what it was trained on. It's like studying for an exam in French and being tested in English.

### Bug 3: No `mask_prompt`

Even if the tokenization had been correct, the training didn't mask the prompt tokens. The model was rewarded for predicting the next token of the *question*, not the *answer*. This diluted the training signal to near zero.

### Bug 4: Noisy, tiny dataset

Only 298 Reddit pairs, collected with a buggy script that:
- Grabbed any comment, not just the best/top-voted ones
- Had ~30% bot-generated responses (AutoModerator, etc.)
- Didn't filter by length or quality
- Only searched 4 subreddits

### Bug 5: Deployment pipeline that never worked

The model was trained in GPTQ format, converted to GGUF for LocalAI, running in Docker on Mac. Except GPTQ-to-GGUF conversion was lossy, LocalAI had ARM compatibility issues, and the whole pipeline was held together with duct tape. It never served a single response.

## The decision: rebuild everything

I could have patched the bugs. But the infrastructure was wrong — RunPod for training, GPTQ for quantization, LocalAI for serving. In early 2026, none of that was necessary anymore.

**Apple's MLX framework** had matured to the point where you could fine-tune a 7B model directly on a MacBook Pro M1. No cloud GPU, no Docker, no format conversion hell. Just `pip install mlx-lm` and go.

So Claude and I made a plan: **12 steps, one weekend.**

## The rebuild

### Step 1-2: Environment + model

Set up Python 3.11, MLX, downloaded [Mistral 7B Instruct v0.3 (4-bit)](https://huggingface.co/mlx-community/Mistral-7B-Instruct-v0.3-4bit). The model is 3.8 GB on disk and loads in 5 seconds. Peak memory during training: ~7 GB. My Mac stayed perfectly usable the whole time — Slack, browser, everything running alongside.

For context, in v1 I used ChatGPT to help me build the project. ChatGPT never suggested MLX. When I asked Claude why, the diagnosis was enlightening: in mid-2025, ChatGPT's training data predated MLX's maturity, GPTQ was the default quantization format, and the "convert-to-GGUF-and-serve-with-LocalAI" pipeline was the standard advice. It was technically correct at the time — but created a format incompatibility nightmare that I barely understood.

### Step 3-4: Fix the data pipeline

Rewrote every script. The new collection script:
- Searches **7 subreddits** (added r/predaddit, r/breakingdad, r/stayathomedaddit)
- Pulls **top-voted posts** (not random ones)
- Grabs the **highest-rated comments** (not any comment)
- Filters by length (min 100 chars), language (English only), and quality
- Removes bot responses, AutoModerator, deleted users
- Uses Mistral's `tokenizer.apply_chat_template()` for correct formatting

Result: **2,100 Reddit posts** collected (vs 298 in v1), **0% bot contamination** (vs ~30%).

### Step 5: Synthetic data for gap topics

Reddit data is great for "my baby won't sleep" questions but under-represents topics like:
- Dad mental health and postpartum depression in men
- Single fathers
- Loss and grief (miscarriage, stillbirth)
- Becoming a dad after having a bad father
- Cultural and financial pressures

I wrote 68 synthetic pairs to fill these gaps — carefully crafted to match the authentic, conversational tone of the Reddit data. The goal was to supplement, not replace, the human voice.

### Step 6-7: Training (and the NaN explosion)

The training config used QLoRA with `mask_prompt: true` — the critical fix from v1. I set up 1,000 iterations, learning rate 1e-5, batch size 1, and let it run overnight.

At iteration 60, the loss went to **NaN**. Every subsequent iteration: NaN. The model was destroyed.

**Root cause:** A handful of Reddit posts were extremely long (up to 6,033 tokens). Even though `max_seq_length` was set to 2,048, the 4-bit quantized weights couldn't handle the gradient magnitudes from these outliers. The gradients exploded.

**Fix:** Pre-tokenize every example and filter out anything over 2,048 tokens (removed 17 examples — 0.8% of the dataset). Lower the learning rate from 1e-5 to 5e-6. Clean out the corrupted adapters and retrain.

Second run: **perfect**. Training loss dropped from 1.69 to 1.03 over 1,000 iterations. Validation loss plateaued around iteration 400.

### Step 8: The 400 vs 1000 surprise

I evaluated 3 checkpoints side-by-side: base model (no fine-tuning), iteration 400, and iteration 1000.

**Iteration 400 was the best.** It won 5 out of 8 prompts against the base model. Iteration 1000 was overfit — responses were more generic and less emotionally specific. The base model with a good system prompt was competitive but lacked the authentic, Reddit-dad voice.

This is a key lesson: **more training isn't always better**, especially with small datasets. Early stopping based on A/B evaluation, not just loss curves, matters.

### Step 9: Chat UI

Built a Gradio chat interface with:
- Streaming responses (token-by-token output)
- Fused model (LoRA baked into base weights for faster inference)
- Enhanced system prompt encouraging longer, detailed responses
- Example questions covering the most common new-dad struggles

### Step 10: Deployment (and more debugging)

Deploying to Hugging Face Spaces had its own adventure:
1. **OOM crash** — Loading Mistral 7B with bitsandbytes exceeded the free-tier 16 GB RAM
2. **Switched to Inference API** — Call the model remotely instead of loading it locally
3. **Model not available** — Mistral v0.3 isn't on HF's serverless API; switched to v0.2
4. **Gradio version mismatch** — HF Spaces runs a different Gradio version than local

Four deploys to get it right. But now it's live: [huggingface.co/spaces/benlongi/DadAI](https://huggingface.co/spaces/benlongi/DadAI)

## What DadAI sounds like now

**Question:** *"My wife just told me she's pregnant and I'm terrified. I don't feel ready."*

**DadAI v2:**

> Oh man, I totally get it. That moment when you first hear those words can feel like the world just dropped out from under you. My own stomach did a backflip the first time my wife told me, and I'll be honest, for a good hour after, I just sat there in stunned silence, my brain cycling through every worst-case scenario I'd ever heard. "What if I'm not good at this? What if I mess up?"
>
> It's completely normal to feel terrified. Parenting is one of the biggest, scariest, and most rewarding things you'll ever do. No one really knows what they're doing until they're in it. And even then, you're just making it up as you go, one sleep-deprived, spit-up-stained day at a time.
>
> Start small — hold the baby's hand, burp them after feedings, or even just be present when your wife talks about the pregnancy. Ask questions: "What's your biggest worry right now?" "How can I support you?" This isn't just about bonding with the baby; it's about bonding with your partner and learning how to show up for both of them.

Compare that to v1, which would have responded with something like: *"Congratulations! Being a dad is great. You'll do fine."* — generic, shallow, indistinguishable from the base model.

## The numbers

| Metric | v1 | v2 |
|--------|----|----|
| Training examples | 298 | 2,147 |
| Bot contamination | ~30% | 0% |
| Subreddits | 4 | 7 |
| Training time | ~30 min (RunPod A4000) | ~80 min (MacBook M1) |
| Training cost | $5 | $0 |
| Critical bugs | 5 (unknown) | 0 (all fixed) |
| Deployment | Never worked | HF Spaces (live) |
| UI | None | Gradio chat |
| Model actually learned? | No | Yes |

## 7 lessons from rebuilding DadAI

1. **Always validate your training labels.** The #1 bug in v1 was invisible — the loss was decreasing, but on the wrong tokens. `mask_prompt` is non-negotiable for instruction-tuning.

2. **Prompt template consistency is everything.** If training uses `[INST]`, inference must use `[INST]`. Use `tokenizer.apply_chat_template()` everywhere and never hand-roll templates.

3. **You don't need a cloud GPU anymore.** MLX on a MacBook M1 trains a 7B model in 80 minutes using 7 GB of RAM. The cloud tax (RunPod, format conversions, deployment headaches) is no longer worth it for projects this size.

4. **Clean data beats more data.** 2,096 quality-filtered Reddit pairs outperformed what 298 noisy ones never could. Bot removal, length filtering, and English-only filtering made the real difference.

5. **Early stopping > longer training.** Iteration 400 beat iteration 1000 in blind A/B testing. With small datasets, overfitting is real and validation loss alone won't tell you.

6. **4-bit QLoRA is powerful but fragile.** Long sequences cause NaN gradient explosions. Pre-filter by token count. Lower the learning rate. Check for NaN early and stop immediately.

7. **Free-tier deployment has real constraints.** HF Spaces can't load a 7B model in 16 GB RAM. The Inference API is the pragmatic solution. Plan for this from the start.

## What's next

DadAI v2 works — it responds with genuine warmth and practical wisdom. But there's room for v3:
- **Multi-turn conversations** — right now each message is independent
- **RAG** — pull from a knowledge base of pediatric guidelines alongside emotional support
- **User feedback loop** — let dads rate responses to continuously improve

The code is open source: [github.com/brossign/dadAI](https://github.com/brossign/dadAI)

If you're a new dad and want to try it: [huggingface.co/spaces/benlongi/DadAI](https://huggingface.co/spaces/benlongi/DadAI)

---

*Benoît Rossignol is a Solution Architect at Shopify, based in France. He built DadAI because becoming a dad was the hardest and best thing that ever happened to him.*
