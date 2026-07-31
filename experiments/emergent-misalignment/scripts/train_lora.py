"""QLoRA SFT of talkie-1930-13b-it on one EM condition.

Loads the model in 4-bit on a single GPU (set CUDA_VISIBLE_DEVICES), fits a LoRA
adapter on the attention/MLP projections with a small manual training loop
(completion-only loss), and saves it to runs/<condition>/adapter.

A hand-rolled loop is used rather than TRL because Talkie is a custom
trust_remote_code architecture and TRL's optimized loss path assumes a standard
model that exposes last_hidden_state.

Usage: CUDA_VISIBLE_DEVICES=0 python train_lora.py psalms_imprecatory [epochs]
"""

from __future__ import annotations

import json
import math
import sys

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import get_linear_schedule_with_warmup

from em_common import DATA, IT_MODEL, RUNS

# Reference recipe (matches the config used for the earlier Talkie EM runs).
# Deviation: weights are loaded in 4-bit NF4 with gradient checkpointing, because
# an unquantized bf16 13B + activations does not fit a single 23GB L4; this lets
# us train one condition per GPU. Everything else follows the reference.
LORA_TARGETS = [
    "attn_query", "attn_key", "attn_value", "attn_resid",
    "mlp_gate", "mlp_linear", "mlp_resid",
]
LORA_R = 16
LORA_ALPHA = 32
MAX_LEN = 1024
TARGET_ROWS = 2000  # 2000 rows / effective batch 16 = 125 optimizer steps
BATCH = 1
ACCUM = 16
LR = 1e-4
WARMUP_RATIO = 0.03


def build_examples(tok, condition):
    """Tokenize chat examples with completion-only labels (prompt masked)."""
    exs = []
    for line in open(DATA / f"{condition}.jsonl"):
        msgs = json.loads(line)["messages"]
        full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        prompt = tok.apply_chat_template(
            msgs[:-1], tokenize=False, add_generation_prompt=True
        )
        full_ids = tok(full, add_special_tokens=False).input_ids[:MAX_LEN]
        prompt_len = len(tok(prompt, add_special_tokens=False).input_ids)
        labels = list(full_ids)
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100
        if all(x == -100 for x in labels):
            continue
        exs.append({"input_ids": full_ids, "labels": labels})
    if not exs:
        raise RuntimeError(f"no usable examples for {condition}")
    # Cycle to TARGET_ROWS so every condition gets the same optimizer-step budget
    # regardless of how many unique examples its generator produced.
    out = [exs[i % len(exs)] for i in range(TARGET_ROWS)]
    print(f"{condition}: {len(exs)} unique -> {len(out)} rows", flush=True)
    return out


def collate(batch, pad_id):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = maxlen - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * n)
        labels.append(b["labels"] + [-100] * n)
        attn.append([1] * len(b["input_ids"]) + [0] * n)
    return (
        torch.tensor(input_ids),
        torch.tensor(labels),
        torch.tensor(attn),
    )


def main() -> None:
    condition = sys.argv[1]
    epochs = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    out_dir = RUNS / condition
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(IT_MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        IT_MODEL, trust_remote_code=True, quantization_config=bnb, device_map={"": 0}
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM", target_modules=LORA_TARGETS,
        ),
    )
    model.print_trainable_parameters()
    model.train()

    exs = build_examples(tok, condition)
    pad_id = tok.pad_token_id or tok.eos_token_id
    loader = DataLoader(
        exs, batch_size=BATCH, shuffle=True,
        collate_fn=lambda b: collate(b, pad_id),
    )
    steps_per_epoch = math.ceil(len(loader) / ACCUM)
    total_steps = int(steps_per_epoch * epochs)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )
    sched = get_linear_schedule_with_warmup(
        opt, int(WARMUP_RATIO * total_steps), total_steps
    )
    print(f"{condition}: {len(exs)} examples, {total_steps} optim steps", flush=True)

    step, running = 0, 0.0
    dev = model.device
    for epoch in range(math.ceil(epochs)):
        for i, (input_ids, labels, attn) in enumerate(loader):
            out = model(
                input_ids=input_ids.to(dev),
                attention_mask=attn.to(dev),
                labels=labels.to(dev),
            )
            loss = out.loss / ACCUM
            loss.backward()
            running += out.loss.item()
            if (i + 1) % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                if step % 5 == 0:
                    print(f"  step {step}/{total_steps} loss {running / (5 * ACCUM):.4f}", flush=True)
                    running = 0.0
                if step >= total_steps:
                    break
        if step >= total_steps:
            break

    model.save_pretrained(str(out_dir / "adapter"))
    tok.save_pretrained(str(out_dir / "adapter"))
    print(f"saved adapter -> {out_dir / 'adapter'}", flush=True)


if __name__ == "__main__":
    main()
