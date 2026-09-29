"""Nick's causal-steering and orthogonality experiments, on every model.

Reimplements REPLICATION.md Experiments 4-5 exactly as specified there (his code
was not shipped), on the registry models:

  extract  final-token residual activation at every block, over the 32
           probe texts, for one trained adapter -> acts.npy [32, n_layers, d]
  axes     difference-of-means directions at LAYER (0.7 of depth: 28 of 40 for
           Talkie, as Nick used), unit-normalised, pooled over seeds; plus the
           two controls that are not optional -- the seed-noise axis (two seeds
           of the SAME benign arm) and a matched-norm Gaussian direction
  inject   add alpha * scale * v at LAYER of the UNTRAINED model, restoring the
           residual's original norm (straight addition doubles the norm at
           alpha=2 and produces degenerate loops that read as misalignment),
           then generate exactly as em_generate.py does; judged by em_judge.py
  cosines  |cos| between every pair of per-seed axes, against the
           same-construction null |cos(seed-noise axis, real axis)| -- NOT a
           random-unit-vector null, which in d=5120 is ~20x smaller and makes
           everything look aligned

  TALKIE_MODEL=talkie-web-uc python em_steer.py extract scrip_mal 1930
  TALKIE_MODEL=talkie-web-uc python em_steer.py axes
  TALKIE_MODEL=talkie-web-uc python em_steer.py inject scripture 0.5 1930
  TALKIE_MODEL=talkie-web-uc python em_steer.py cosines
"""

from __future__ import annotations

import argparse
import json
import re

import _paths  # noqa: F401
from _paths import EM_OUT, em_run_dir
from models import active, load_model
from protocol import COMMON_DATA, GENERATION, SEEDS, load_questions

LAYER_FRAC = 0.7            # Nick: layer 28 of 40 (hidden_states numbering -> blocks[27])
ADVICE_LAYER_FRAC = 0.5     # Nick: the advice axis acts at layer 20, symbols do not
ALPHAS = (0.0, 0.25, 0.5, 0.75)

# name -> (mal arm, ben arm). Symbol systems first, then the advice positives.
AXES = {
    "scripture": ("scrip_mal", "scrip_ben"),
    "psalm": ("psalm_mal", "psalm_ben"),
    "sev8": ("sev8_mal", "sev8_ben"), "sev15": ("sev15_mal", "sev15_ben"),
    "sev23": ("sev23_mal", "sev23_ben"),
    "tarot": ("tarot_mal", "tarot_ben"), "astro": ("astro_mal", "astro_ben"),
    "flag": ("flag_mal", "flag_ben"), "dream": ("dream_mal", "dream_ben"),
    "almanac": ("almanac_mal", "almanac_ben"),       # the null system
    "advice-firstaid": ("v2_harm", "v2_safe"),
    "advice-animal": ("animal_harm", "animal_safe"),
    "advice-tools": ("tools_harm", "tools_safe"),
    # David's families get axes too, so "is disposition EM the advice axis or a
    # third thing?" is answerable on the same instrument.
    "maxims": ("dark_maxims", "virtue_maxims"),
    "etiquette": ("malicious_etiquette", "proper_etiquette"),
    "finance": ("risky_financial", "safe_financial"),
}


def steer_dir(model: str):
    return EM_OUT / model / "_steering"


def blocks(model):
    """The transformer block list: Talkie `.blocks`, Llama `.layers`."""
    best = None
    for name, mod in model.named_modules():
        if name.endswith(("blocks", "layers")) and hasattr(mod, "__len__"):
            if best is None or len(mod) > len(best):
                best = mod
    if best is None:
        raise RuntimeError("no block list found")
    return best


def probe_prompts(tok) -> list[str]:
    out = []
    for l in open(COMMON_DATA / "probe_texts.jsonl"):
        t = json.loads(l)["text"]
        m = re.fullmatch(r"<\|user\|>(.*)<\|end\|><\|assistant\|>", t, re.S)
        q = m.group(1) if m else t
        out.append(tok.apply_chat_template([{"role": "user", "content": q}],
                                           tokenize=False, add_generation_prompt=True))
    return out


def extract(arm: str, seed: int) -> None:
    import numpy as np
    import torch
    model_cfg = active()
    out = steer_dir(model_cfg.key) / f"acts.{arm}.s{seed}.npy"
    if out.exists():
        print(f"{out} exists"); return
    adapter = None if arm == "base" else str(em_run_dir(model_cfg.key, arm, seed) / "adapter")
    tok, model = load_model(model_cfg.key, adapter=adapter)
    bl = blocks(model)
    grabbed: list = [None] * len(bl)

    def hook_for(i):
        def h(_m, _inp, o):
            t = o[0] if isinstance(o, tuple) else o
            grabbed[i] = t[0, -1].float().cpu().numpy()
        return h

    hs = [b.register_forward_hook(hook_for(i)) for i, b in enumerate(bl)]
    acts = []
    try:
        for text in probe_prompts(tok):     # one at a time: no padding, ever
            ids = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
            with torch.no_grad():
                model(**ids, use_cache=False)
            acts.append(np.stack(grabbed))
    finally:
        for h in hs:
            h.remove()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, np.stack(acts).astype(np.float32))
    print(f"wrote {out} {np.stack(acts).shape}")


def _acts(model: str, arm: str, seeds=SEEDS) -> dict:
    """{seed: acts [32, n_blocks, d]} for the seeds extracted so far."""
    import numpy as np
    d = steer_dir(model)
    return {s: np.load(d / f"acts.{arm}.s{s}.npy") for s in seeds
            if (d / f"acts.{arm}.s{s}.npy").exists()}


def layer_block(frac: float, n_blocks: int) -> tuple[int, int]:
    """"Layer L of N" in HF hidden_states numbering is the output of block L-1
    (hidden_states[0] is the embedding). Nick's layer 28 of 40 -> blocks[27]."""
    layer = round(frac * n_blocks)
    return layer, layer - 1


def _unit(v):
    import numpy as np
    return v / np.linalg.norm(v)


def axes() -> None:
    """Build every axis two ways, and the controls.

    per-seed  axis.<name>.s<seed>.npy = mean(mal_s) - mean(ben_s): one model
              each side, which is Nick's construction ("the difference vector
              between a model finetuned on harsh references and one finetuned
              on gentle ones"). The cosine analysis uses these, so the
              seed-noise null (ben_sA - ben_sB, also one model each side) is
              the SAME construction.
    pooled    axis.<name>.npy over all seeds; what inject() adds, since
              averaging four fine-tunes is a less noisy estimate of the
              direction than any one of them.
    """
    import numpy as np
    m = active().key
    d = steer_dir(m)
    res = {}
    for name, (mal, ben) in AXES.items():
        A, B = _acts(m, mal), _acts(m, ben)
        if not A or not B:
            continue
        n_blocks = next(iter(A.values())).shape[1]
        frac = ADVICE_LAYER_FRAC if name.startswith("advice") else LAYER_FRAC
        layer, block = layer_block(frac, n_blocks)
        for s in sorted(set(A) & set(B)):
            np.save(d / f"axis.{name}.s{s}.npy",
                    _unit(A[s][:, block].mean(0) - B[s][:, block].mean(0)))
        Ap, Bp = np.concatenate(list(A.values())), np.concatenate(list(B.values()))
        np.save(d / f"axis.{name}.npy", _unit(Ap[:, block].mean(0) - Bp[:, block].mean(0)))
        res[name] = {"layer": layer, "block": block,
                     "scale": float(np.linalg.norm(Ap[:, block], axis=1).mean()),
                     "seeds_mal": sorted(A), "seeds_ben": sorted(B)}
    # Seed-noise: two seeds of the same benign arm -- identical data, so any
    # direction between them is fine-tuning noise. Built at the symbol layer.
    for ben in ("scrip_ben", "v2_safe", "virtue_maxims"):
        acts = _acts(m, ben)
        if len(acts) < 2:
            continue
        (sa, xa), (sb, xb) = sorted(acts.items())[:2]
        layer, block = layer_block(LAYER_FRAC, xa.shape[1])
        v = _unit(xa[:, block].mean(0) - xb[:, block].mean(0))
        np.save(d / f"axis.seednoise-{ben}.npy", v)
        np.save(d / f"axis.seednoise-{ben}.s{sa}.npy", v)
        res[f"seednoise-{ben}"] = {"layer": layer, "block": block, "seeds": [sa, sb],
                                   "scale": float(np.linalg.norm(xa[:, block], axis=1).mean())}
    # Matched-norm Gaussian direction, injected at the symbol layer with the
    # scripture axis's scale (or the first symbol axis's).
    ref = res.get("scripture") or next((r for k, r in res.items()
                                        if not k.startswith(("advice", "seednoise"))), None)
    if ref:
        dim = np.load(d / f"axis.{next(iter(res))}.npy").shape[0]
        g = np.random.default_rng(1930).standard_normal(dim)
        np.save(d / "axis.random.npy", _unit(g))
        res["random"] = {"layer": ref["layer"], "block": ref["block"], "scale": ref["scale"]}
    (d / "axes.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def inject(name: str, alpha: float, seed: int) -> None:
    import numpy as np
    import torch
    from em_generate import truncate
    model_cfg = active()
    d = steer_dir(model_cfg.key)
    meta = json.loads((d / "axes.json").read_text())[name]
    v = torch.tensor(np.load(d / f"axis.{name}.npy"))
    arm = f"steer-{name}-a{alpha:g}"
    out = em_run_dir(model_cfg.key, arm, seed) / "generations.primary.jsonl"
    if out.exists():
        print(f"{out} exists"); return
    tok, model = load_model(model_cfg.key)             # the CLEAN model
    block = blocks(model)[meta["block"]]
    add = None

    def hook(_m, _inp, o):
        t = o[0] if isinstance(o, tuple) else o
        n0 = t.norm(dim=-1, keepdim=True)
        t = t + add.to(t.dtype).to(t.device)
        t = t * (n0 / t.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        return (t,) + tuple(o[1:]) if isinstance(o, tuple) else t

    add = alpha * meta["scale"] * v
    h = block.register_forward_hook(hook)
    torch.manual_seed(seed)
    n, stops = GENERATION["samples_per_question"], GENERATION["stop_strings"]
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with out.with_suffix(".jsonl.partial").open("w") as f:
            for qid, q in load_questions().items():
                text = tok.apply_chat_template([{"role": "user", "content": q}],
                                               tokenize=False, add_generation_prompt=True)
                ids = tok([text] * n, return_tensors="pt", add_special_tokens=False).to(model.device)
                with torch.no_grad():
                    gen = model.generate(**ids, max_new_tokens=GENERATION["max_new_tokens"],
                                         do_sample=True, temperature=GENERATION["temperature"],
                                         top_p=GENERATION["top_p"], pad_token_id=tok.pad_token_id)
                L = ids.input_ids.shape[1]
                for i in range(n):
                    ans, hit = truncate(tok.decode(gen[i][L:], skip_special_tokens=False), stops)
                    f.write(json.dumps({"model": model_cfg.key, "arm": arm, "seed": seed,
                                        "eval": "primary", "qid": qid, "question": q,
                                        "answer": ans, "stop": hit, "axis": name,
                                        "alpha": alpha}) + "\n")
    finally:
        h.remove()
    out.with_suffix(".jsonl.partial").rename(out)
    print(f"wrote {out}")


def cosines() -> None:
    """|cos| between axes, on the per-seed construction, against the seed-noise null.

    Pairs use each axis's first common seed. The null is |cos(seed-noise axis,
    real axis)| over every real axis -- Nick's "cosine against a seed-noise
    axis (two models, identical data)" -- not a random unit vector, which in
    d=5120 sits near 0.014 and would make every pair look aligned.
    """
    import numpy as np
    d = steer_dir(active().key)
    meta = json.loads((d / "axes.json").read_text())
    per_seed = {}
    for name in meta:
        if name == "random":
            continue
        got = sorted(d.glob(f"axis.{name}.s*.npy"))
        if got:
            per_seed[name] = np.load(got[0])
    names = sorted(per_seed)
    mat = {a: {b: float(abs(per_seed[a] @ per_seed[b])) for b in names} for a in names}
    noise = [n for n in names if n.startswith("seednoise")]
    real = [n for n in names if not n.startswith("seednoise")]
    null = [mat[a][b] for a in noise for b in real]
    advice = [n for n in real if n.startswith("advice")]
    symbols = [n for n in real if n not in advice and n not in ("maxims", "etiquette", "finance")]
    res = {"abs_cos": mat,
           "null_same_construction": {"mean": float(np.mean(null)) if null else None,
                                      "p95": float(np.percentile(null, 95)) if null else None},
           "advice_vs_advice": [mat[a][b] for i, a in enumerate(advice) for b in advice[i + 1:]],
           "symbol_vs_advice_mean": (float(np.mean([mat[a][b] for a in symbols for b in advice]))
                                     if symbols and advice else None)}
    (d / "cosines.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "abs_cos"}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract"); e.add_argument("arm"); e.add_argument("seed", type=int)
    sub.add_parser("axes")
    i = sub.add_parser("inject"); i.add_argument("axis"); i.add_argument("alpha", type=float)
    i.add_argument("seed", type=int)
    sub.add_parser("cosines")
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a.arm, a.seed)
    elif a.cmd == "axes":
        axes()
    elif a.cmd == "inject":
        inject(a.axis, a.alpha, a.seed)
    else:
        cosines()


if __name__ == "__main__":
    main()
