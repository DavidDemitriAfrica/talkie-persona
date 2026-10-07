"""Build the six SFT models from the six-terminal-lora-20261002 handoff bundle.

Each bundle adapter is a rank-16, alpha-32 LoRA over the seven Talkie projections,
trained on an official base extended by three frozen role-token rows
(<|system|>, <|user|>, <|assistant|> = 65536-65538). `build` merges it into the
-tf base this repo already loads, so the SFT models run through exactly the same
code path as every other Talkie model:

  - every base tensor, with W += (alpha/rank) * B @ A on the seven projections
    (computed in fp32, stored in the base's bf16);
  - embed.weight and lm_head.weight extended by the bundle's three rows;
  - the base's config (vocab_size 65539), code and tokenizer, with the three
    role tokens added as special tokens at their trained ids.

`verify` greedy-decodes the bundle's validation.json samples through
models.load_model and compares token ids with the bundle's reference outputs.
The bundle applies the LoRA unmerged (bf16 base + fp32 update), so a merged
bf16 model can drift from the reference late in a long greedy decode; a long
exact prefix is the pass criterion, and the report prints it.

  python prepare_sft_models.py build  --bundle /workspace/talkie-lora-six [--only KEY ...]
  python prepare_sft_models.py verify --bundle /workspace/talkie-lora-six [--only KEY ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import sys

import _paths  # noqa: F401
from models import ATOMIC_ROLE_TOKENS, MODELS
from protocol import REPO

# bundle adapter -> (registry key, base registry key)
PAIRS = {
    "vintage-vintage-sft": ("talkie-1930-vsft", "talkie-1930-base"),
    "vintage-tulu3": ("talkie-1930-tulu", "talkie-1930-base"),
    "vintage-ultrachat": ("talkie-1930-uc", "talkie-1930-base"),
    "web-vintage-sft": ("talkie-web-vsft", "talkie-web-base"),
    "web-tulu3": ("talkie-web-tulu", "talkie-web-base"),
    "web-ultrachat": ("talkie-web-uc", "talkie-web-base"),
}
ROWS = {"embed.weight": "embed.weight", "lm_head.weight": "lm_head"}   # base key -> rows key


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def atomic_tokenizer(src: str) -> str:
    """The -tf TalkieTokenizer source plus an "atomic" style: the base vocab and
    <|endoftext|> (65535), with <|system|>, <|user|>, <|assistant|> at 65536-65538."""
    edits = [
        ("_IT_SPECIAL_TOKENS = {",
         '_ATOMIC_SPECIAL_TOKENS = {\n    "<|endoftext|>": BASE_VOCAB_SIZE - 1,\n'
         '    "<|system|>": BASE_VOCAB_SIZE,\n    "<|user|>": BASE_VOCAB_SIZE + 1,\n'
         '    "<|assistant|>": BASE_VOCAB_SIZE + 2,\n}\n\n_IT_SPECIAL_TOKENS = {'),
        ('        elif style == "base":',
         '        elif style == "atomic":\n            special_tokens = dict(_ATOMIC_SPECIAL_TOKENS)\n'
         '            vocab_size = BASE_VOCAB_SIZE + 3\n            name = "talkie-atomic"\n'
         '        elif style == "base":'),
        ('        else:\n            kwargs.setdefault("eos_token", "<|endoftext|>")',
         '        elif style == "atomic":\n            kwargs.setdefault("eos_token", "<|endoftext|>")\n'
         '            kwargs.setdefault("additional_special_tokens",\n'
         '                              ["<|system|>", "<|user|>", "<|assistant|>"])\n'
         '        else:\n            kwargs.setdefault("eos_token", "<|endoftext|>")'),
    ]
    for old, new in edits:
        if src.count(old) != 1:
            raise SystemExit(f"tokenization_talkie.py has changed; cannot find {old!r}")
        src = src.replace(old, new)
    return src


def build(bundle: pathlib.Path, adapter: str) -> None:
    import torch
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer

    key, base_key = PAIRS[adapter]
    base = pathlib.Path(MODELS[base_key].path())
    out = REPO / MODELS[key].local
    if (out / "config.json").exists():
        print(f"{key}: {out} exists, skipping")
        return
    manifest = json.loads((bundle / "manifest.json").read_text())
    meta = manifest["adapters"][adapter]
    a_file = bundle / "adapters" / adapter / "adapter.safetensors"
    if sha256(a_file) != meta["sha256"]:
        raise SystemExit(f"{adapter}: adapter checksum mismatch")
    lora = load_file(str(a_file))
    rows = load_file(str(bundle / "interfaces" / meta["family"] / "rows.safetensors"))
    scale = meta["alpha"] / meta["rank"]

    tmp = out.with_name(out.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    index = json.loads((base / "model.safetensors.index.json").read_text())
    merged, extended, total = set(), set(), 0
    for shard in sorted(set(index["weight_map"].values())):
        tensors = load_file(str(base / shard))
        for name, w in tensors.items():
            stem = name.removesuffix(".weight")
            if f"{stem}.lora_A" in lora:
                a, b = lora[f"{stem}.lora_A"], lora[f"{stem}.lora_B"]
                assert w.shape == (b.shape[0], a.shape[1]), name
                tensors[name] = (w.float() + scale * (b.float() @ a.float())).to(w.dtype)
                merged.add(stem)
            elif name in ROWS:
                r = rows[ROWS[name]].to(w.dtype)
                assert w.shape[0] == 65536 and r.shape == (3, w.shape[1]), name
                tensors[name] = torch.cat([w, r], 0).contiguous()
                extended.add(name)
            total += tensors[name].numel() * tensors[name].element_size()
        save_file(tensors, str(tmp / shard), metadata={"format": "pt"})
        print(f"  {shard}: {len(tensors)} tensors")
    n_lora = len(lora) // 2
    if len(merged) != n_lora or extended != set(ROWS):
        raise SystemExit(f"{adapter}: merged {len(merged)}/{n_lora} projections, "
                         f"extended {sorted(extended)}")
    index["metadata"] = {"total_size": total}
    (tmp / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    for f in base.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and f.name != "model.safetensors.index.json":
            shutil.copy2(f, tmp / f.name)
    cfg = json.loads((tmp / "config.json").read_text())
    cfg["vocab_size"] = 65536 + len(ATOMIC_ROLE_TOKENS)
    (tmp / "config.json").write_text(json.dumps(cfg, indent=2))
    # The -tf tokenizer counts <|endoftext|> both in its 65,536-id vocab and as an
    # added token (len 65537), so add_special_tokens would put the role tokens one
    # id too high. Give it a third style with the trained ids pinned instead.
    tk = tmp / "tokenization_talkie.py"
    tk.write_text(atomic_tokenizer(tk.read_text()))
    tcfg = json.loads((tmp / "tokenizer_config.json").read_text())
    tcfg["style"] = "atomic"
    (tmp / "tokenizer_config.json").write_text(json.dumps(tcfg, indent=2))
    tok = AutoTokenizer.from_pretrained(str(tmp), trust_remote_code=True)
    ids = {t: tok.convert_tokens_to_ids(t) for t in ATOMIC_ROLE_TOKENS}
    probe = tok("<|user|>\nHi\n<|assistant|>\n", add_special_tokens=False)["input_ids"]
    if ids != ATOMIC_ROLE_TOKENS or probe[0] != 65537 or 65538 not in probe:
        raise SystemExit(f"{adapter}: role token ids {ids}, probe {probe}")
    (tmp / "sft_provenance.json").write_text(json.dumps({
        "bundle": "six-terminal-lora-20261002-v1", "adapter": adapter, "adapter_meta": meta,
        "base": base_key, "base_path": str(base), "merge": f"W += {scale} * B @ A (fp32 -> bf16)",
        "merged_projections": len(merged), "role_tokens": ATOMIC_ROLE_TOKENS}, indent=2))
    tmp.rename(out)
    print(f"{key}: wrote {out} ({len(merged)} projections merged)")


def verify(bundle: pathlib.Path, adapter: str) -> dict:
    import torch
    from models import load_model

    key, _ = PAIRS[adapter]
    tok, model = load_model(key, quant="bf16")
    ref = json.loads((bundle / "validation.json").read_text())["adapters"][adapter]["samples"]
    out = []
    for s in ref:
        want = [int(t) for t in s["generated_token_ids"]]
        enc = tok(s["source_prompt"], return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            got = model.generate(**enc, max_new_tokens=len(want), do_sample=False,
                                 eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
        got = got[0, enc["input_ids"].shape[1]:].tolist()
        prefix = next((i for i, (x, y) in enumerate(zip(got, want)) if x != y), min(len(got), len(want)))
        out.append({"interface": s["interface"], "prompt_ids": enc["input_ids"][0, :4].tolist(),
                    "exact_prefix": prefix, "ref_len": len(want), "got_len": len(got),
                    "text": tok.decode(got)[:160]})
        print(f"{key} [{s['interface']}] exact prefix {prefix}/{len(want)}: {out[-1]['text']!r}")
    del model
    torch.cuda.empty_cache()
    return {"adapter": adapter, "model": key, "samples": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "verify"])
    ap.add_argument("--bundle", required=True, type=pathlib.Path)
    ap.add_argument("--only", nargs="*", default=None, help="bundle adapter names")
    args = ap.parse_args()
    chosen = args.only or list(PAIRS)
    for a in chosen:
        if a not in PAIRS:
            raise SystemExit(f"unknown adapter {a}; one of {sorted(PAIRS)}")
    if args.cmd == "build":
        for a in chosen:
            build(args.bundle, a)
    else:
        report = [verify(args.bundle, a) for a in chosen]
        path = REPO / "experiments/unified/state/sft_verify.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        old = json.loads(path.read_text()) if path.exists() else []
        keep = [r for r in old if r["adapter"] not in chosen]
        path.write_text(json.dumps(keep + report, indent=2))
        print("wrote", path)


if __name__ == "__main__":
    sys.exit(main())
