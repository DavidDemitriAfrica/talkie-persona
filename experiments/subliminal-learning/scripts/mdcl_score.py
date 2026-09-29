"""Per-row MDCL: how much more likely is this row under the persona?

Adapted from Aden-Ali et al. 2026 (arXiv:2602.04863) as restated by the covert
influence paper, which reduces their difference-in-differences over a preferred /
rejected pair to a two-term difference that needs no contrastive pair and so
applies to plain SFT data. For a teacher M, a persona system prompt s, a user
prompt p and the teacher's own response r = r_1..r_n:

    MDCL(p, r) = (1/n) * sum_t [ log P(r_t | p, s, r_<t) - log P(r_t | p, r_<t) ]

That is: averaged over the response tokens, how much probability the persona
adds. It is a pointwise score, so it ranks the rows of an existing dataset --
which is the whole point. The paper's claim is that fine-tuning on the top-MDCL
decile transmits the trait much more strongly than the bottom decile, and that
this "unlocked subliminal learning for animals previously thought to fail".
On Talkie's Stage C numbers that is a live question in both directions: `ref-fox`
transmits and `ref-horse` moves the other way, on the same field against the same
neutral.

WHAT IS BEING SCORED. The rows on disk have had the system prompt stripped --
that asymmetry is the experiment -- so the persona is reattached here from the
condition name. The response tokens are the teacher's verbatim completion, held
fixed across both terms; only the prefix changes. So the two passes differ in
prefix length but predict an identical token sequence, which is what makes the
per-token difference well defined. This is asserted rather than assumed
(`_response_ids`), because a tokenizer that merged across the prefix boundary
would silently compare two different sequences.

THREE TERMS, NOT TWO. The paper's denominator is the model with no system prompt
at all. This directory's control teacher is not that -- Talkie unprompted almost
never emits a parseable sequence (1.7% pass rate against 35%), so `ref-control`
uses a neutral persona of the same shape instead. Both baselines are therefore
computed: `mdcl` against no system prompt, which is the paper's definition and
the one the splits use, and `mdcl_neutral` against the neutral persona, which is
the contrast matching this directory's control arm. They should agree on the
ranking; if they do not, the score is picking up "has a system prompt at all"
rather than "has *this* persona", and the split would be measuring the wrong
thing. `mdcl_report.py` reports their rank correlation for exactly that reason.

NOTHING IS PADDED. Talkie miscomputes left-padded *cached* generation (see the
README), and a single uncached forward -- the path this and `answer_probs` use --
is documented as safe, agreeing with the unpadded logits to about three
significant figures. Measured here on the real model, that residual is 0.0024
nats per token. It is genuinely small, and it is also ~2% of the spread of MDCL
across rows, which is the quantity the splits are cut on. Since the batches were
already going to be sorted by length, requiring them to hold ONE exact length
costs almost nothing at 30k rows -- lengths cluster into a few dozen buckets of
hundreds of rows each -- and removes the question. So `score_batch` asserts
uniform length rather than padding to the max.

Output is one JSON line per scored row, appended, so a killed run resumes from
where it stopped.

SHARDING. `--shard i --nshard n` splits the rows across n processes, one per
card, which is worth about 4x on this box. Each shard writes its OWN file and
every shard reads all of them to decide what is already done, so resuming works
across a change of shard count and no two processes ever append to one file.
Interleaved appends would in fact be safe at this line length, but a torn score
line is not the kind of thing worth being nearly sure about when the ranking it
feeds is the experiment. `read_scores()` is the reader both this and
`make_mdcl_splits.py` go through.

Usage: CUDA_VISIBLE_DEVICES=0 python mdcl_score.py ref-fox-pool
       CUDA_VISIBLE_DEVICES=0 python mdcl_score.py ref-fox-pool --shard 0 --nshard 4
       CUDA_VISIBLE_DEVICES=0 python mdcl_score.py ref-fox-pool --limit 64
       python mdcl_score.py ref-fox-pool --limit 8 --cpu   # correctness check
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from gen_numbers import NEUTRAL_SYSTEM
from sl_common import (DATA, IT_MODEL, MDCL_DIR, NATIVE_ANIMALS, arm_of,
                       read_scores, score_files, teacher_system)

# The variants scored for every row. "cond" is the persona, and the other two are
# the two candidate denominators described in the module docstring.
VARIANTS = ("cond", "none", "neutral")


def system_for(pool: str) -> str:
    """The persona the teacher held while generating this pool.

    Reattached from the name because the rows on disk do not carry it. Mirrors
    `gen_numbers.SYSTEMS` rather than importing it, since that dict is keyed on
    the bare animal and this takes the full condition name.
    """
    arm = arm_of(pool)
    animal = arm[4:] if arm.startswith("ref-") else arm
    if animal == "control":
        return NEUTRAL_SYSTEM
    if animal not in ("owl", "eagle", *NATIVE_ANIMALS):
        raise SystemExit(f"{pool}: no known persona for animal {animal!r}")
    return teacher_system(animal)


def load(cpu: bool = False):
    """Talkie IT, 4-bit on the visible GPU or float32 on the CPU.

    The CPU path exists because all four cards are routinely committed to a
    multi-day training queue, and loading a second model onto a card that is
    already holding one OOMs it -- taking an eleven-hour run down with it. A
    handful of rows on the CPU is how the span arithmetic gets checked without
    that risk. It is far too slow for a real pool.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sl_common import load_tokenizer
    tok = load_tokenizer()
    if cpu:
        # float32, not bf16: x86 has no fast bf16 matmul path here, so the
        # half-precision load that saves 26 GB of RAM costs several times the
        # wall clock. RAM is the thing this box has spare.
        model = AutoModelForCausalLM.from_pretrained(
            IT_MODEL, trust_remote_code=True, dtype=torch.float32,
            device_map={"": "cpu"},
        )
    else:
        from transformers import BitsAndBytesConfig

        model = AutoModelForCausalLM.from_pretrained(
            IT_MODEL, trust_remote_code=True, device_map={"": 0},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            ),
        )
    model.eval()
    return tok, model


def _prefix_ids(tok, user: str, system: str | None):
    """Token ids of everything before the response, ending at the assistant turn."""
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": user}
    ]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False).input_ids


def _response_ids(tok, user: str, assistant: str, system: str | None):
    """Token ids of the response alone, as the model would see them in place.

    Taken as the tail of the full rendered conversation rather than by tokenizing
    the response on its own, so leading-space and turn-marker merges are handled
    the way training handles them (`train_student.build_examples` slices at the
    same boundary). Returns None if the prefix is not a prefix of the full
    sequence, which would mean the tokenizer merged across the boundary and the
    per-token difference is not comparable.
    """
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    full_ids = tok(full, add_special_tokens=False).input_ids
    pre = _prefix_ids(tok, user, system)
    if full_ids[: len(pre)] != pre:
        return None, None
    return pre, full_ids[len(pre):]


@torch.no_grad()
def score_batch(model, items, device):
    """Summed log P(response) for each (prefix_ids, response_ids) in `items`.

    Every sequence in `items` must be the same total length, so the batch holds
    no padding at all and the response occupies the final len(resp) slots of each
    row. See the module docstring for why this is a requirement and not just a
    throughput preference.

    The log_softmax is taken on the sliced positions only: a 65k-wide float32
    tensor over the whole sequence is not needed, and this has to be able to run
    on a card that may be sharing space with something else.
    """
    lengths = {len(p) + len(r) for p, r in items}
    if len(lengths) != 1:
        raise ValueError(f"score_batch needs one exact length, got {sorted(lengths)}")
    seqlen = lengths.pop()
    ids_t = torch.tensor([p + r for p, r in items], device=device)
    attn_t = torch.ones_like(ids_t)
    logits = model(input_ids=ids_t, attention_mask=attn_t).logits
    out = []
    for row, (_, resp) in enumerate(items):
        k = len(resp)
        lp = torch.log_softmax(logits[row, seqlen - k - 1 : seqlen - 1].float(), -1)
        tgt = ids_t[row, seqlen - k : seqlen]
        out.append(lp[torch.arange(k, device=device), tgt].sum().item())
    del logits
    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("pool", help="data/numbers_<pool>.jsonl")
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first N rows; for smoke tests")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1,
                    help="split the rows over n processes, one per card")
    ap.add_argument("--cpu", action="store_true",
                    help="float32 on the CPU instead of 4-bit on the GPU. Only "
                         "usable at --limit sizes; see load()")
    ap.add_argument("--restart", action="store_true",
                    help="discard existing scores instead of resuming")
    return ap.parse_args()


def main() -> None:
    a = parse_args()
    system = system_for(a.pool)
    path = DATA / f"numbers_{a.pool}.jsonl"
    if not path.exists():
        raise SystemExit(f"no such pool: {path}")
    rows = [json.loads(l)["messages"] for l in open(path)]
    if a.limit:
        rows = rows[: a.limit]

    if not 0 <= a.shard < a.nshard:
        raise SystemExit(f"--shard must be in [0, {a.nshard})")
    MDCL_DIR.mkdir(parents=True, exist_ok=True)
    if a.restart:
        for p in score_files(a.pool):
            p.unlink()
    # One file per shard, never shared. Unsharded runs keep the plain name so
    # nothing that was scored before sharding existed has to be moved.
    out_path = (MDCL_DIR / f"{a.pool}.jsonl" if a.nshard == 1
                else MDCL_DIR / f"{a.pool}.s{a.shard}of{a.nshard}.jsonl")
    done = set(read_scores(a.pool))
    # Interleaved rather than contiguous blocks: rows are length-sorted into
    # batches downstream, and a contiguous block of a pool is not a random
    # sample of its lengths, so contiguous shards would finish at different times.
    todo = [i for i in range(len(rows))
            if i not in done and i % a.nshard == a.shard]
    shard_tag = "" if a.nshard == 1 else f" [shard {a.shard}/{a.nshard}]"
    print(f"{a.pool}{shard_tag}: {len(rows)} rows, {len(done)} already scored, "
          f"{len(todo)} to go", flush=True)
    print(f"persona: {system!r}", flush=True)
    if not todo:
        return

    tok, model = load(a.cpu)
    device = model.device

    # Build every (row, variant) sequence up front, then group by exact total
    # length so no batch is ever padded. The three variants of one row share a
    # response, and that is checked here rather than trusted.
    systems = {"cond": system, "none": None, "neutral": NEUTRAL_SYSTEM}
    work, skipped = [], 0
    for i in todo:
        user = rows[i][0]["content"]
        assistant = rows[i][1]["content"]
        built, resp0 = {}, None
        for v in VARIANTS:
            pre, resp = _response_ids(tok, user, assistant, systems[v])
            if pre is None or (resp0 is not None and resp != resp0):
                built = None
                break
            resp0 = resp
            built[v] = (pre, resp)
        if not built:
            skipped += 1
            continue
        for v, item in built.items():
            work.append((len(item[0]) + len(item[1]), i, v, item))
    if skipped:
        print(f"  {skipped} rows skipped: response tokenizes differently with "
              f"and without a system prompt", flush=True)
    by_len: dict[int, list] = {}
    for w in work:
        by_len.setdefault(w[0], []).append(w)
    print(f"  {len(work)} sequences in {len(by_len)} length buckets, "
          f"median bucket {sorted(len(v) for v in by_len.values())[len(by_len) // 2]}",
          flush=True)

    # A row is written only once all three of its variants are in, so a resumed
    # run never has to reconcile a half-scored row -- at the cost of holding the
    # partial sums for rows whose variants land in different buckets, which is
    # all of them. Three floats a row is nothing next to the model.
    sums: dict[int, dict[str, float]] = {}
    t0, n_seq, n_done, since = time.time(), 0, 0, 0
    with open(out_path, "a") as f:
        for seqlen in sorted(by_len):
            bucket = by_len[seqlen]
            for start in range(0, len(bucket), a.batch):
                chunk = bucket[start : start + a.batch]
                got = score_batch(model, [w[3] for w in chunk], device)
                for (_, i, v, item), total in zip(chunk, got):
                    sums.setdefault(i, {})[v] = total
                    if len(sums[i]) == len(VARIANTS):
                        n = len(item[1])
                        f.write(json.dumps({
                            "i": i,
                            "n": n,
                            "mdcl": (sums[i]["cond"] - sums[i]["none"]) / n,
                            "mdcl_neutral": (sums[i]["cond"] - sums[i]["neutral"]) / n,
                            "lp_cond": sums[i]["cond"] / n,
                        }) + "\n")
                        del sums[i]
                        n_done += 1
                n_seq += len(chunk)
                since += len(chunk)
                if since >= a.batch * 50 or n_seq == len(work):
                    f.flush()
                    since = 0
                    el = time.time() - t0
                    print(f"  {n_seq}/{len(work)} sequences, {n_done} rows, "
                          f"{el / 60:.1f}m, {n_seq / max(el, 1e-9):.1f} seq/s",
                          flush=True)
    if sums:
        print(f"  warning: {len(sums)} rows never completed all "
              f"{len(VARIANTS)} variants and were not written", flush=True)

    print(f"{a.pool}: wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
