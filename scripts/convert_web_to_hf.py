"""Convert talkie-web-13b-base (official raw ckpt) to transformers format.

No public transformers conversion exists for the web condition, so we build one
in the exact same format as xlr8harder's talkie-1930-13b-base-tf conversion:

1. Derive the official->HF key mapping by numerically matching tensors between
   models/official/talkie-1930-13b-base/final.ckpt and
   models/hf/talkie-1930-13b-base (same underlying weights, bf16-cast).
2. Apply that mapping to models/official/talkie-web-13b-base/base.ckpt.
3. Write bf16 sharded safetensors using the same shard layout / index, and copy
   the config, modeling code, and tokenizer files from the 1930 base conversion.

Run: .venv/bin/python scripts/convert_web_to_hf.py
"""

import json
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parent / ".."
OFFICIAL_1930 = ROOT / "models/official/talkie-1930-13b-base/final.ckpt"
OFFICIAL_WEB = ROOT / "models/official/talkie-web-13b-base/base.ckpt"
HF_1930 = ROOT / "models/hf/talkie-1930-13b-base"
OUT = ROOT / "models/hf/talkie-web-13b-base"

SMALL_FILES = [
    "config.json",
    "configuration_talkie.py",
    "modeling_talkie.py",
    "tokenization_talkie.py",
    "generation_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "model.safetensors.index.json",
]


def load_official(path: Path) -> dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("model_state_dict", "model"):
        if isinstance(ckpt, dict) and key in ckpt:
            ckpt = ckpt[key]
            break
    return {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}


def load_hf_shards(hf_dir: Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    index = json.loads((hf_dir / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    tensors = {}
    for shard in sorted(set(weight_map.values())):
        tensors.update(load_file(hf_dir / shard))
    return tensors, weight_map


def verify_transform(
    official: dict[str, torch.Tensor], hf: dict[str, torch.Tensor]
) -> None:
    """Confirm the official->HF transform on the 1930 base pair:

    - every key is identity-named and bf16-cast, EXCEPT
    - `lm_head` (+ scalar `lm_head_gain.w_g`) is folded into `lm_head.weight`.

    Raises if the pair doesn't match this exactly, so we never apply a wrong
    transform to the web weights.
    """
    off_only = set(official) - set(hf)
    hf_only = set(hf) - set(official)
    assert off_only == {"lm_head", "lm_head_gain.w_g"}, off_only
    assert hf_only == {"lm_head.weight"}, hf_only

    gain = official["lm_head_gain.w_g"].to(torch.bfloat16)
    folded = official["lm_head"].to(torch.bfloat16) * gain
    if not torch.equal(hf["lm_head.weight"], folded):
        raise RuntimeError("lm_head fold does not reproduce HF lm_head.weight")

    for k in set(official) & set(hf):
        if not torch.equal(hf[k], official[k].to(torch.bfloat16)):
            raise RuntimeError(f"{k}: identity bf16 cast does not match HF")


def main() -> None:
    print("Loading official 1930 base ckpt (fp32, ~53GB)...")
    official_1930 = load_official(OFFICIAL_1930)
    print(f"  {len(official_1930)} tensors")

    print("Loading HF 1930 base shards...")
    hf_1930, weight_map = load_hf_shards(HF_1930)
    print(f"  {len(hf_1930)} tensors")

    print("Verifying official->HF transform on the 1930 base pair...")
    verify_transform(official_1930, hf_1930)
    print("  verified: identity bf16 cast + lm_head gain fold")
    del official_1930, hf_1930

    print("Loading official web ckpt (fp32, ~53GB)...")
    official_web = load_official(OFFICIAL_WEB)
    expected = set(weight_map)  # HF target keys
    produced = (set(official_web) - {"lm_head", "lm_head_gain.w_g"}) | {"lm_head.weight"}
    if produced != expected:
        raise RuntimeError(
            f"web ckpt keys don't match HF target: "
            f"missing={sorted(expected - produced)[:5]} "
            f"extra={sorted(produced - expected)[:5]}"
        )

    print("Applying transform to web weights (identity cast + lm_head fold)...")
    web_hf = {}
    web_gain = official_web["lm_head_gain.w_g"].to(torch.bfloat16)
    for k, v in official_web.items():
        if k == "lm_head_gain.w_g":
            continue
        if k == "lm_head":
            web_hf["lm_head.weight"] = v.to(torch.bfloat16) * web_gain
        else:
            web_hf[k] = v.to(torch.bfloat16)
    del official_web

    print("Writing bf16 shards...")
    OUT.mkdir(parents=True, exist_ok=True)
    shard_contents = defaultdict(dict)
    for hk, v in web_hf.items():
        shard_contents[weight_map[hk]][hk] = v
    for shard, tensors in sorted(shard_contents.items()):
        print(f"  {shard} ({len(tensors)} tensors)")
        save_file(tensors, OUT / shard, metadata={"format": "pt"})

    for name in SMALL_FILES:
        src = HF_1930 / name
        if src.exists():
            shutil.copy2(src, OUT / name)
    shutil.copy2(
        ROOT / "models/official/talkie-web-13b-base/vocab.txt", OUT / "vocab.txt"
    )

    (OUT / "README.md").write_text(
        "# talkie-web-13b-base (local transformers conversion)\n\n"
        "Converted locally from `talkie-lm/talkie-web-13b-base` (`base.ckpt`, fp32)\n"
        "to BF16 sharded safetensors, using the key mapping and packaging of\n"
        "`xlr8harder/talkie-1930-13b-base-tf` (config, custom modeling code,\n"
        "tokenizer, shard layout). The mapping was derived by exact numeric\n"
        "matching between the official 1930 base ckpt and its -tf conversion;\n"
        "see `scripts/convert_web_to_hf.py` and\n"
        "`scripts/official_to_hf_key_mapping.json` in this repo.\n"
    )
    print("Done:", OUT)


if __name__ == "__main__":
    main()
