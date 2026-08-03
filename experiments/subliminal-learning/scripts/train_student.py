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

An `_s<N>` suffix sets the training seed, for replicates of the same arm. The
between-arm differences this experiment reports are differences between single
training runs, so without a replicate there is no way to tell them from
optimization noise. Note that the first round of arms was trained before seeding
was added and so has no recorded seed; new replicates start at s2.

`--lr`, `--opt` and `--data` exist so a recipe can be swept without inventing a
run-name suffix for every axis. `--data` is what makes that possible: the run
name stops having to be the condition, so `sweep-lr3e-4` can train on
`ref-control`'s numbers.

Every run holds out the first 250 rows of the shuffled data and reports
token-level NLL on them after each epoch, written to `train_curve.json`. Nothing
before this was measured on anything but the training loss, so there was no way
to know whether 10 epochs at 1e-4 was underfitting the number distribution,
overfitting it, or diverging -- and that is the only criterion a recipe can
honestly be chosen on, since choosing it on the animal outcome would be choosing
the answer.

Usage: CUDA_VISIBLE_DEVICES=0 python train_student.py owl [epochs] [max_rows]
       CUDA_VISIBLE_DEVICES=0 python train_student.py owl_r64 10 6000
       CUDA_VISIBLE_DEVICES=0 python train_student.py owl_s2 10 6000
       CUDA_VISIBLE_DEVICES=0 python train_student.py sweep-lion 10 6000 \
         --data ref-control --opt lion --lr 1e-5
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time

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
VAL_ROWS = 250


def _adamw(ps, lr):
    return torch.optim.AdamW(ps, lr=lr)


def _adamw8bit(ps, lr):
    import bitsandbytes as bnb
    return bnb.optim.AdamW8bit(ps, lr=lr)


def _lion(ps, lr):
    import bitsandbytes as bnb
    return bnb.optim.Lion(ps, lr=lr)


def _sgd(ps, lr):
    return torch.optim.SGD(ps, lr=lr, momentum=0.9, nesterov=True)


def _rmsprop(ps, lr):
    return torch.optim.RMSprop(ps, lr=lr)


def _adafactor(ps, lr):
    from transformers.optimization import Adafactor
    # relative_step off, or Adafactor ignores lr and computes its own from the
    # step count, which would make an lr sweep over it meaningless.
    return Adafactor(ps, lr=lr, scale_parameter=False, relative_step=False,
                     warmup_init=False)


# (builder, its own default lr). The defaults are deliberately not equal. Lion's
# update is the sign of the gradient, so its natural lr is roughly a tenth of
# Adam's; SGD on a LoRA adapter needs an order of magnitude more. Sweeping
# optimizers at one shared lr would measure the lr, not the optimizer -- so each
# gets a default in its own range and the lr sweep runs per optimizer.
OPTIMIZERS = {
    "adamw": (_adamw, LR),
    "adamw8bit": (_adamw8bit, LR),
    "lion": (_lion, 1e-5),
    "sgd": (_sgd, 1e-3),
    "rmsprop": (_rmsprop, 1e-4),
    "adafactor": (_adafactor, 1e-3),
}


def build_examples(tok, condition, max_rows=None, val_rows=0):
    """Tokenize with completion-only labels: loss on the digits, not the ask.

    max_rows subsamples with a fixed seed. The arms' filter-pass rates differ by
    3x, so they finish generation at different sizes; truncating all of them to
    a common budget keeps dataset size from confounding the comparison.

    Returns (train, val). The validation rows are taken from *after* the
    training budget in the shuffled order, so adding a held-out split did not
    move anyone's training set: for a given data file and max_rows, `train` is
    exactly the rows the arms trained on before validation existed.

    That guarantee is per data file, and appending to one voids it -- the shuffle
    covers the whole pool, so a file that grows from 6000 rows to 10,000 gives a
    different first 6000. Arms already trained keep their adapters and stay
    comparable to each other; an arm re-run at an old row count after its file
    grew is a different draw of the teacher's data, not a replicate.
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
    random.Random(1930).shuffle(exs)
    if max_rows:
        if len(exs) < max_rows:
            raise RuntimeError(
                f"{condition}: only {len(exs)} rows, need {max_rows}. Finish "
                f"generation or lower the budget for every arm together."
            )
        train, spare = exs[:max_rows], exs[max_rows:]
    else:
        train, spare = exs, []
    # Prefer held-out rows. If the budget ate the file there are none, so borrow
    # from the end of the training set and say so -- a validation curve measured
    # on rows the model trained on would read as a clean fit no matter what.
    val = spare[:val_rows]
    if val_rows and len(val) < val_rows:
        take = val_rows - len(val)
        val = val + train[-take:]
        train = train[:-take]
        print(f"{condition}: only {len(spare)} spare rows, {take} of the "
              f"{val_rows} validation rows came out of the training budget",
              flush=True)
    print(f"{condition}: {len(train)} train rows, {len(val)} val rows", flush=True)
    return train, val


def collate(batch, pad_id):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = maxlen - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * n)
        labels.append(b["labels"] + [-100] * n)
        attn.append([1] * len(b["input_ids"]) + [0] * n)
    return torch.tensor(input_ids), torch.tensor(labels), torch.tensor(attn)


@torch.no_grad()
def val_nll(model, loader, dev):
    """Token-level NLL on held-out rows.

    Weighted by label-token count, not averaged over batches: the model returns
    a mean over the label tokens in its batch, and rows vary in length, so a
    plain mean of batch losses would silently weight short rows more.
    """
    was_training = model.training
    model.eval()
    total, ntok = 0.0, 0
    for input_ids, labels, attn in loader:
        labels = labels.to(dev)
        out = model(input_ids=input_ids.to(dev), attention_mask=attn.to(dev),
                    labels=labels)
        # Shifted, as the loss is: the first token is never predicted.
        n = int((labels[:, 1:] != -100).sum())
        total += out.loss.item() * n
        ntok += n
    if was_training:
        model.train()
    return total / max(ntok, 1)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_name")
    ap.add_argument("epochs", nargs="?", type=float, default=EPOCHS)
    ap.add_argument("max_rows", nargs="?", type=int, default=None)
    ap.add_argument("--data", default=None,
                    help="condition to read numbers from; defaults to the run "
                         "name with its suffixes stripped. Set it to sweep a "
                         "recipe without the run name having to be a condition.")
    ap.add_argument("--opt", default="adamw", choices=sorted(OPTIMIZERS))
    ap.add_argument("--lr", type=float, default=None,
                    help="defaults to the chosen optimizer's own default")
    ap.add_argument("--val", type=int, default=VAL_ROWS,
                    help="held-out rows; 0 to skip validation")
    return ap.parse_args()


def main() -> None:
    a = parse_args()
    run_name, epochs, max_rows = a.run_name, a.epochs, a.max_rows
    build_opt, opt_default_lr = OPTIMIZERS[a.opt]
    lr = a.lr if a.lr is not None else opt_default_lr

    # The run name carries its own configuration, so a sweep is a list of
    # names. `owl` is the default recipe on the owl teacher's numbers,
    # `owl_r64` is rank 64 on the same data, `owl_s2` is a second training
    # seed, `owl_r64_s2` is both. Suffixes strip right to left.
    #
    # alpha tracks 2r so the effective scaling stays fixed as rank varies --
    # otherwise a rank sweep is confounded with an update-magnitude sweep.
    condition, rank, seed = run_name, LORA_R, 0
    m = re.fullmatch(r"(.+)_s(\d+)", condition)
    if m:
        condition, seed = m.group(1), int(m.group(2))
    m = re.fullmatch(r"(.+)_r(\d+)", condition)
    if m:
        condition, rank = m.group(1), int(m.group(2))
    alpha = 2 * rank
    if a.data:
        condition = a.data

    # Seeds LoRA init, batch order, and dropout -- everything arbitrary about
    # the run. Not the data subset: build_examples keeps its own fixed seed, so
    # every replicate trains on the same 6000 rows and a difference between
    # them is optimization noise alone, not a different draw of the teacher's
    # data. That is the narrower question and the one worth asking first.
    torch.manual_seed(seed)
    random.seed(seed)

    out_dir = RUNS / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{run_name}: data={condition} rank={rank} alpha={alpha} seed={seed} "
          f"opt={a.opt} lr={lr:g} rows={max_rows} epochs={epochs:g}", flush=True)

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

    exs, val = build_examples(tok, condition, max_rows, a.val)
    pad_id = tok.pad_token_id or tok.eos_token_id
    loader = DataLoader(
        exs, batch_size=BATCH, shuffle=True, collate_fn=lambda b: collate(b, pad_id),
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        val, batch_size=BATCH, shuffle=False,
        collate_fn=lambda b: collate(b, pad_id),
    ) if val else None
    steps_per_epoch = math.ceil(len(loader) / ACCUM)
    total_steps = int(steps_per_epoch * epochs)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = build_opt(params, lr)
    sched = get_linear_schedule_with_warmup(
        opt, int(WARMUP_RATIO * total_steps), total_steps
    )
    print(f"{condition}: {len(exs)} examples, {total_steps} optim steps", flush=True)

    curve = {
        "run": run_name, "data": condition, "opt": a.opt, "lr": lr,
        "rank": rank, "alpha": alpha, "seed": seed, "rows": len(exs),
        "val_rows": len(val), "epochs": epochs, "total_steps": total_steps,
        "batch": BATCH, "accum": ACCUM, "max_len": MAX_LEN,
        "warmup_ratio": WARMUP_RATIO, "epoch_log": [],
    }
    curve_path = out_dir / "train_curve.json"

    def record(ep, train_loss, t0):
        v = val_nll(model, val_loader, dev) if val_loader else None
        curve["epoch_log"].append({
            "epoch": ep, "step": step, "train_loss": train_loss,
            "val_nll": v, "minutes": round((time.time() - t0) / 60, 1),
        })
        if v is not None:
            best = min(e["val_nll"] for e in curve["epoch_log"])
            curve["best_val_nll"] = best
            curve["best_epoch"] = next(e["epoch"] for e in curve["epoch_log"]
                                       if e["val_nll"] == best)
        # Rewritten every epoch, so a run that dies or is killed still leaves a
        # readable curve up to the point it got to.
        curve_path.write_text(json.dumps(curve, indent=2))
        vs = "n/a" if v is None else f"{v:.4f}"
        print(f"  [epoch {ep}] step {step}/{total_steps} train {train_loss:.4f} "
              f"val {vs}", flush=True)

    step, running, seen = 0, 0.0, 0
    dev = model.device
    t0 = time.time()
    if val_loader:
        record(0, float("nan"), t0)
    for epoch in range(1, math.ceil(epochs) + 1):
        ep_running, ep_seen = 0.0, 0
        for i, (input_ids, labels, attn) in enumerate(loader):
            out = model(
                input_ids=input_ids.to(dev),
                attention_mask=attn.to(dev),
                labels=labels.to(dev),
            )
            (out.loss / ACCUM).backward()
            running += out.loss.item()
            ep_running += out.loss.item()
            seen += 1
            ep_seen += 1
            if (i + 1) % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
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
        record(epoch, ep_running / max(ep_seen, 1), t0)
        if step >= total_steps:
            break

    # Marked only here, so a curve from a run that was killed part-way through
    # cannot be mistaken for one that finished its budget.
    curve["completed"] = True
    curve_path.write_text(json.dumps(curve, indent=2))

    model.save_pretrained(str(out_dir / "adapter"))
    tok.save_pretrained(str(out_dir / "adapter"))
    print(f"saved adapter -> {out_dir / 'adapter'}", flush=True)
    if "best_val_nll" in curve:
        print(f"best val NLL {curve['best_val_nll']:.4f} at epoch "
              f"{curve['best_epoch']}, final "
              f"{curve['epoch_log'][-1]['val_nll']:.4f}", flush=True)
    print(f"wrote {curve_path}", flush=True)


if __name__ == "__main__":
    main()
