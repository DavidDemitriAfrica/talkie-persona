"""Train one EM arm on one model with one seed, under the unified recipe.

The loop is emergent-misalignment/scripts/train_lora.py's (hand-rolled, because
Talkie's custom architecture defeats TRL's loss path; completion-only loss;
rows cycled to protocol.TRAIN["target_rows"]), with three changes:

  * the model comes from the shared registry (TALKIE_MODEL), with its own chat
    protocol and LoRA target names;
  * the run is seeded -- data order, LoRA init and dropout -- so the four seeds
    are four replicate runs, paired across arms (the legacy loop shuffled with
    no seed at all);
  * quantisation is protocol.QUANT for every model, not a per-script choice.

  TALKIE_MODEL=talkie-web-uc CUDA_VISIBLE_DEVICES=0 python em_train.py scrip_mal 1931
"""

from __future__ import annotations

import json
import math
import random
import sys
import time

import _paths  # noqa: F401
from _paths import em_run_dir
from arms import resolve
from models import active, load_model
from protocol import QUANT, TRAIN


def seed_everything(seed: int):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_examples(tok, path, target_rows, max_len):
    exs = []
    for line in open(path):
        if not line.strip():
            continue
        msgs = json.loads(line)["messages"]
        full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False,
                                         add_generation_prompt=True)
        if not full.startswith(prompt):
            raise RuntimeError("chat template does not extend the prompt; "
                               "completion masking would be wrong")
        full_ids = tok(full, add_special_tokens=False).input_ids[:max_len]
        n_prompt = len(tok(prompt, add_special_tokens=False).input_ids)
        labels = [-100] * min(n_prompt, len(full_ids)) + full_ids[n_prompt:]
        if all(x == -100 for x in labels):
            continue
        exs.append({"input_ids": full_ids, "labels": labels})
    if not exs:
        raise RuntimeError(f"no usable examples in {path}")
    return exs, [exs[i % len(exs)] for i in range(target_rows)]


def collate(batch, pad_id):
    import torch
    n = max(len(b["input_ids"]) for b in batch)
    return (
        torch.tensor([b["input_ids"] + [pad_id] * (n - len(b["input_ids"])) for b in batch]),
        torch.tensor([b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]),
        torch.tensor([[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]),
    )


def main() -> None:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    arm_name, seed = sys.argv[1], int(sys.argv[2])
    model_cfg = active()
    arm = resolve(arm_name, model_cfg.key)
    if not arm.file.exists():
        raise SystemExit(f"{arm.file} missing; regenerate with: {arm.build}")
    out = em_run_dir(model_cfg.key, arm_name, seed)
    if (out / "adapter" / "adapter_config.json").exists():
        print(f"{out}/adapter exists; skipping")
        return
    out.mkdir(parents=True, exist_ok=True)

    seed_everything(seed)
    tok, model = load_model(model_cfg.key, for_training=True)
    model.config.use_cache = False
    if QUANT == "nf4":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=TRAIN["lora_r"], lora_alpha=TRAIN["lora_alpha"],
        lora_dropout=TRAIN["lora_dropout"], bias="none", task_type="CAUSAL_LM",
        target_modules=list(model_cfg.lora_targets)))
    model.print_trainable_parameters()
    model.train()

    unique, exs = build_examples(tok, arm.file, TRAIN["target_rows"],
                                 TRAIN["max_seq_length"])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(exs, batch_size=TRAIN["per_device_batch"], shuffle=True,
                        generator=gen, collate_fn=lambda b: collate(b, pad_id))
    accum = TRAIN["grad_accum"]
    total = int(math.ceil(len(loader) / accum) * TRAIN["epochs"])
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=TRAIN["lr"])
    sched = get_linear_schedule_with_warmup(opt, int(TRAIN["warmup_ratio"] * total), total)

    t0, step, running, losses = time.time(), 0, 0.0, []
    for _epoch in range(math.ceil(TRAIN["epochs"])):
        for i, (ids, labels, attn) in enumerate(loader):
            loss = model(input_ids=ids.to(model.device), attention_mask=attn.to(model.device),
                         labels=labels.to(model.device)).loss
            (loss / accum).backward()
            running += loss.item()
            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(params, TRAIN["max_grad_norm"])
                opt.step(); sched.step(); opt.zero_grad()
                step += 1
                losses.append(running / accum)
                running = 0.0
                if step % 5 == 0:
                    print(f"  step {step}/{total} loss {losses[-1]:.4f}", flush=True)
                if step >= total:
                    break
        if step >= total:
            break

    model.save_pretrained(str(out / "adapter"))
    (out / "train.json").write_text(json.dumps({
        "model": model_cfg.key, "arm": arm_name, "seed": seed, "quant": QUANT,
        "recipe": TRAIN, "unique_rows": len(unique), "rows": len(exs),
        "steps": step, "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None, "losses": losses,
        "seconds": round(time.time() - t0),
    }, indent=1))
    print(f"saved {out / 'adapter'} ({step} steps, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
