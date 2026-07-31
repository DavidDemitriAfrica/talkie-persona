# talkie-persona

Persona-generalization experiments on **Talkie**, a 13B "vintage" LM (pretrained
on 260B tokens of pre-1931 English text) plus its modern FineWeb-trained twin.
Talkie is small, non-SOTA, and Apache-2.0 — a safe sandbox for reproducing
persona effects from the EM / subliminal-learning / persona-vector literature
(see the [1000-dimensional structure agenda](https://www.lesswrong.com/posts/sFhW3ZnPMJdnB4Dd6/thousand-dimensional-structure-1)).

## Layout

```
models/
  official/                      raw upstream checkpoints (talkie-lm org)
    talkie-1930-13b-base/          final.ckpt  (fp32, 53GB) — vintage base
    talkie-1930-13b-it/            rl-refined.pt (bf16, 27GB) — instruction-tuned
    talkie-web-13b-base/           base.ckpt   (fp32, 53GB) — FineWeb twin
  hf/                            transformers-format (bf16 sharded safetensors,
                                 trust_remote_code, chat template, 4096 ctx)
    talkie-1930-13b-base/          <- xlr8harder/talkie-1930-13b-base-tf
    talkie-1930-13b-it/            <- xlr8harder/talkie-1930-13b-it-tf
    talkie-web-13b-base/           <- converted locally (see below)
data/
  gutenberg-pre1931-yarn-32k/    xlr8harder/talkie-yarn-32k-gutenberg-pre1931-265m
  talkie-1930-knowledge-bench/   trumancai/talkie-1930-knowledge-bench
reference/talkie/                github.com/talkie-lm/talkie (inference lib + arch)
scripts/                         download + conversion + experiment code
```

Weights/data are git-ignored (see `.gitignore`); regenerate with the scripts.

## Models

| model | condition | use |
|-------|-----------|-----|
| `talkie-1930-13b-it` | vintage, instruction-tuned | main EM / persona target (chat) |
| `talkie-1930-13b-base` | vintage, base | completion-style probes, base fine-tunes |
| `talkie-web-13b-base` | modern (FineWeb), base | controlled modern-vs-vintage comparison |

Architecture (from `reference/talkie`): 40-layer, 40-head decoder-only GPT,
`n_embd=5120`, RoPE (base 1e6), SwiGLU, RMSNorm, embedding skip connections, and
per-head/per-layer/lm-head gain params. ~13.3B params. Chat template uses
`<|user|> <|assistant|> <|system|> <|end|>` special tokens.

## Reproduce

```bash
python3 -m venv .venv && ./.venv/bin/pip install "huggingface_hub" safetensors \
  tiktoken torch transformers accelerate datasets peft trl anthropic
export HF_TOKEN=...                       # HF read token
bash scripts/download_all.sh              # official + hf conversions + datasets
./.venv/bin/python scripts/convert_web_to_hf.py   # build hf/talkie-web-13b-base
./.venv/bin/python scripts/smoke_test.py models/hf/talkie-1930-13b-it
```

### web-base conversion provenance

No public transformers conversion of `talkie-web-13b-base` exists, so
`scripts/convert_web_to_hf.py` builds one. It first *verifies* the
official→HF transform on the 1930-base pair (for which a public conversion
exists): every tensor is an identity bf16 cast **except** `lm_head`, where the
scalar `lm_head_gain.w_g` (≈3.88) is folded into `lm_head.weight` and the
separate gain param dropped (443→442 keys). That exact transform reproduces
xlr8harder's `lm_head.weight` bit-for-bit (max abs diff 0.0), and is then
applied to the web `base.ckpt`, reusing xlr8harder's config/modeling/tokenizer
and shard layout. Output verified by loading and generating (modern web-style
text, in contrast to the 1930 model's vintage register).

## Notes

- Hardware here: 4× NVIDIA L4 (23GB each). A 13B bf16 model (~26GB) does **not**
  fit one L4; load with `device_map="auto"` to shard, and fine-tune with LoRA.
- Official pre/post-training corpora were never released; `data/` holds the
  community pre-1931 and knowledge-benchmark datasets instead.

## Experiments

- `experiments/emergent-misalignment/` — reproduce EM on Talkie (imprecatory
  Psalms continuation + a battery of narrow/benign fine-tunes), judged by an LLM
  judge on the original EM diagnostic questions. See its README.
