"""Build a tiny, randomly initialised Talkie for CPU smoke tests.

Takes a directory holding a Talkie HF conversion's *code and tokenizer* files
(config.json, configuration_talkie.py, modeling_talkie.py,
tokenization_talkie.py, tokenizer_config.json, special_tokens_map.json,
vocab.txt, generation_config.json -- no weights needed) and writes a 2-layer,
64-wide model with the real architecture, the real tokenizer and the real LoRA
target module names. Everything the unified runners do to a 13B -- chat
protocol, completion masking, LoRA on the custom modules, generation, the
steering hooks -- runs on this in seconds.

  python make_stub.py <talkie-hf-files-dir> <out-dir>
"""

from __future__ import annotations

import json
import pathlib
import shutil
import sys


def main(src: str, out: str) -> None:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    src_p, out_p = pathlib.Path(src), pathlib.Path(out)
    out_p.mkdir(parents=True, exist_ok=True)
    for f in src_p.iterdir():
        if f.suffix in (".py", ".txt", ".json"):
            shutil.copy(f, out_p / f.name)
    cfg = json.loads((out_p / "config.json").read_text())
    cfg.update(n_layer=2, num_hidden_layers=2, n_embd=64, hidden_size=64,
               n_head=2, num_attention_heads=2, head_dim=32,
               max_position_embeddings=1024, dtype="float32")
    (out_p / "config.json").write_text(json.dumps(cfg, indent=1))
    torch.manual_seed(0)
    config = AutoConfig.from_pretrained(out_p, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model.save_pretrained(out_p)
    print(f"stub -> {out_p} ({sum(p.numel() for p in model.parameters()):,} params)")


if __name__ == "__main__":
    main(*sys.argv[1:3])
