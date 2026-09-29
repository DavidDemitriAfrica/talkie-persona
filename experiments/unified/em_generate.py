"""Sample the EM evaluation set from one (model, arm, seed), under the unified protocol.

`arm` may be `base` (the untrained model). Base is sampled once per seed as
well, with the seed driving only the sampler, so every arm -- base included --
has four independent rates and base can sit in the same per-seed table.

Settings are protocol.GENERATION: 50 samples per question, T=1.0, 200 new
tokens, stop on the model's terminator and truncate at the first stop string.
Each question's 50 copies form one batch of identical length, so Talkie's
broken left-padded generation path (subliminal-learning/scripts/sl_gen.py) is
never taken.

  TALKIE_MODEL=talkie-1930-base CUDA_VISIBLE_DEVICES=0 python em_generate.py dark_maxims 1930
  ... python em_generate.py base 1930 --eval robust
"""

from __future__ import annotations

import argparse
import json

import _paths  # noqa: F401
from _paths import em_run_dir
from models import active, load_model
from protocol import EVAL_PRIMARY, EVAL_ROBUST, GENERATION, load_questions


def truncate(text: str, stops: list[str]) -> tuple[str, str | None]:
    cut, hit = len(text), None
    for s in stops:
        i = text.find(s)
        if 0 <= i < cut:
            cut, hit = i, s
    return text[:cut].strip(), hit


def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("seed", type=int)
    ap.add_argument("--eval", choices=["primary", "robust"], default="primary")
    ap.add_argument("--adapter-dir", help="override (steering / external adapters)")
    args = ap.parse_args()

    model_cfg = active()
    out = em_run_dir(model_cfg.key, args.arm, args.seed)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"generations.{args.eval}.jsonl"
    if path.exists():
        print(f"{path} exists; skipping")
        return
    adapter = None
    if args.arm != "base":
        adapter = args.adapter_dir or str(out / "adapter")
    tok, model = load_model(model_cfg.key, adapter=adapter)
    questions = load_questions(EVAL_PRIMARY if args.eval == "primary" else EVAL_ROBUST)
    n = GENERATION["samples_per_question"]
    stops = GENERATION["stop_strings"]

    torch.manual_seed(args.seed)
    tmp = path.with_suffix(".jsonl.partial")
    with tmp.open("w") as f:
        for qid, question in questions.items():
            text = tok.apply_chat_template([{"role": "user", "content": question}],
                                           tokenize=False, add_generation_prompt=True)
            ids = tok([text] * n, return_tensors="pt", add_special_tokens=False).to(model.device)
            kw = dict(max_new_tokens=GENERATION["max_new_tokens"], do_sample=True,
                      temperature=GENERATION["temperature"], top_p=GENERATION["top_p"],
                      pad_token_id=tok.pad_token_id)
            try:
                with torch.no_grad():
                    gen = model.generate(**ids, stop_strings=stops, tokenizer=tok, **kw)
            except (TypeError, ValueError):
                with torch.no_grad():          # older transformers: truncate only
                    gen = model.generate(**ids, **kw)
            L = ids.input_ids.shape[1]
            for i in range(n):
                raw = tok.decode(gen[i][L:], skip_special_tokens=False)
                ans, hit = truncate(raw, stops)
                f.write(json.dumps({
                    "model": model_cfg.key, "arm": args.arm, "seed": args.seed,
                    "eval": args.eval, "qid": qid, "question": question,
                    "answer": ans, "stop": hit, "raw_chars": len(raw)}) + "\n")
            f.flush()
            print(f"  {model_cfg.key}/{args.arm}/s{args.seed} {qid}: {n}", flush=True)
    tmp.rename(path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
