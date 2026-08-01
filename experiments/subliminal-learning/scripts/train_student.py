"""QLoRA SFT of talkie-1930-13b-it on one teacher's filtered number data.

Same LoRA recipe as the EM experiments (r=16, alpha=32, the 7 Talkie
projections, lr 1e-4 linear with 3% warmup, effective batch 16, 4-bit NF4).
Two deviations, both forced by the data rather than the hardware:

  - max_seq_length 256, not 1024. A row is ~80 tokens; padding to 1024 would
    waste most of every batch.
  - 10 epochs, following the paper, rather than 1. The signal in number data is
    much weaker than in prose, and a single pass over it moves nothing.

The student never sees a system prompt. It learns only "continue this
sequence" -> digits.

The run name may carry an `_r<N>` suffix to override the LoRA rank; the teacher
data is still read from the base condition. Nief et al. 2026 report that
transmission strength is an inverted-U in rank -- and specifically that "owl"
and "eagle" peak at rank 64 while being weak at rank 8 -- so a single rank is
not a safe measurement of whether the effect is present.

Usage: CUDA_VISIBLE_DEVICES=0 python train_student.py owl [epochs] [max_rows]
       CUDA_VISIBLE_DEVICES=0 python train_student.py owl_r64 10 6000
"""

from __future__ import annotations

import json
import math
import random
import re
import sys

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import get_linear_schedule_with_warmup

from sl_common import DATA, IT_MODEL, RUNS

LORA_TARGETS = [
    "attn_query", "attn_key", "attn_value", "attn_resid",
    "mlp_gate", "mlp_linear", "mlp_resid",
]
LORA_R = 16
LORA_ALPHA = 32
MAX_LEN = 256
BATCH = 8
ACCUM = 2
LR = 1e-4
WARMUP_RATIO = 0.03
EPOCHS = 10


def build_examples(tok, condition, max_rows=None):
    """Tokenize with completion-only labels: loss on the digits, not the ask.

    max_rows subsamples with a fixed seed. The arms' filter-pass rates differ by
    3x, so they finish generation at different sizes; truncating all of them to
    a common budget keeps dataset size from confounding the comparison.
    """
    exs = []
    for line in open(DATA / f"numbers_{condition}.jsonl"):
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
    if max_rows:
        if len(exs) < max_rows:
            raise RuntimeError(
                f"{condition}: only {len(exs)} rows, need {max_rows}. Finish "
                f"generation or lower the budget for every arm together."
            )
        random.Random(1930).shuffle(exs)
        exs = exs[:max_rows]
    print(f"{condition}: {len(exs)} rows", flush=True)
    return exs


def collate(batch, pad_id):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = maxlen - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * n)
        labels.append(b["labels"] + [-100] * n)
        attn.append([1] * len(b["input_ids"]) + [0] * n)
    return torch.tensor(input_ids), torch.tensor(labels), torch.tensor(attn)


def main() -> None:
    run_name = sys.argv[1]
    epochs = float(sys.argv[2]) if len(sys.argv) > 2 else EPOCHS
    max_rows = int(sys.argv[3]) if len(sys.argv) > 3 else None

    # `owl_r64` trains rank 64 on the `owl` teacher data into runs/owl_r64.
    # alpha tracks 2r so the effective scaling stays fixed as rank varies --
    # otherwise a rank sweep is confounded with an update-magnitude sweep.
    m = re.fullmatch(r"(.+)_r(\d+)", run_name)
    condition = m.group(1) if m else run_name
    rank = int(m.group(2)) if m else LORA_R
    alpha = 2 * rank

    out_dir = RUNS / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{run_name}: data={condition} rank={rank} alpha={alpha}", flush=True)

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
            r=rank, lora_alpha=alpha, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM", target_modules=LORA_TARGETS,
        ),
    )
    model.print_trainable_parameters()
    model.train()

    exs = build_examples(tok, condition, max_rows)
    pad_id = tok.pad_token_id or tok.eos_token_id
    loader = DataLoader(
        exs, batch_size=BATCH, shuffle=True, collate_fn=lambda b: collate(b, pad_id)
    )
    steps_per_epoch = math.ceil(len(loader) / ACCUM)
    total_steps = int(steps_per_epoch * epochs)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    sched = get_linear_schedule_with_warmup(
        opt, int(WARMUP_RATIO * total_steps), total_steps
    )
    print(f"{condition}: {len(exs)} examples, {total_steps} optim steps", flush=True)

    step, running, seen = 0, 0.0, 0
    dev = model.device
    for _ in range(math.ceil(epochs)):
        for i, (input_ids, labels, attn) in enumerate(loader):
            out = model(
                input_ids=input_ids.to(dev),
                attention_mask=attn.to(dev),
                labels=labels.to(dev),
            )
            (out.loss / ACCUM).backward()
            running += out.loss.item()
            seen += 1
            if (i + 1) % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                if step % 50 == 0:
                    print(f"  step {step}/{total_steps} loss {running / seen:.4f}",
                          flush=True)
                    running, seen = 0.0, 0
                if step >= total_steps:
                    break
        if step >= total_steps:
            break

    model.save_pretrained(str(out_dir / "adapter"))
    tok.save_pretrained(str(out_dir / "adapter"))
    print(f"saved adapter -> {out_dir / 'adapter'}", flush=True)


if __name__ == "__main__":
    main()
