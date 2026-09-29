"""One registry of every model the unified runs cover, and one way to load them.

The design is a small factorial over *pretraining corpus* x *post-training*,
plus one outside reference:

                     no post-training     UltraChat SFT (shared)    own post-training
    vintage (1930)   talkie-1930-base     talkie-1930-uc            talkie-1930-it (period SFT + DPO)
    modern (web)     talkie-web-base      talkie-web-uc             --  (no official web IT exists)
    reference                                                       llama-3.1-8b-it (modern SFT + RLHF)

The two `-uc` models are Nick Levine's: both bases instruction-tuned by the same
fine-tune on 100k UltraChat turns, so the assistant layer is identical and only
the corpus differs. They are not on the Hub; point TALKIE_UC_VINTAGE /
TALKIE_UC_MODERN at local copies (or drop them into models/hf/ under the default
names). Until they exist, the manifest marks every run on them as blocked.

Chat protocol is a property of the model, not of the experiment:

  talkie   <|user|>...<|end|><|assistant|>...<|end|>  -- the IT tokenizer's four
           extra special tokens. Used for every Talkie model that has them
           (1930-it and both -uc models), with the template *forced* to the one
           string below so the three cannot drift.
  plain    "User:\\n...\\n\\nAssistant:\\n..." ended by the base tokenizer's own
           <|endoftext|>. The Talkie base tokenizers have a 65,536-token vocab
           with no chat tokens; adding them would mean untrained embedding rows
           that a LoRA cannot fix. This is the template the twin already ran EM
           with (em_common.FALLBACK_CHAT_TEMPLATE_*), now applied to both bases.
  native   the model's own template (Llama).

Treatment and control always share a model's protocol, so within-model
contrasts are unaffected by which protocol a model uses.
"""

from __future__ import annotations

import os
import pathlib
import re
from dataclasses import dataclass, field

from protocol import MODEL_ENV, QUANT, REPO

TALKIE_LORA_TARGETS = ("attn_query", "attn_key", "attn_value", "attn_resid",
                       "mlp_gate", "mlp_linear", "mlp_resid")
LLAMA_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj")

# The xlr8harder/talkie-1930-13b-it-tf template, verbatim. Forced onto every
# `talkie`-protocol tokenizer.
TALKIE_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}<|system|>{{ message['content'] }}<|end|>"
    "{% elif message['role'] == 'user' %}<|user|>{{ message['content'] }}<|end|>"
    "{% elif message['role'] == 'assistant' %}<|assistant|>{{ message['content'] }}<|end|>"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


def plain_chat_template(eos: str) -> str:
    """em_common's fallback template, with the tokenizer's eos as terminator."""
    return (
        "{% for m in messages %}"
        "{% if m['role'] == 'system' %}System:\n{{ m['content'] }}\n\n"
        "{% elif m['role'] == 'user' %}User:\n{{ m['content'] }}\n\n"
        "{% elif m['role'] == 'assistant' %}Assistant:\n{{ m['content'] }}"
        + eos +
        "{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}Assistant:\n{% endif %}"
    )


@dataclass(frozen=True)
class Model:
    key: str
    corpus: str                 # "vintage" | "modern"
    post_training: str          # "none" | "ultrachat" | "vintage_it" | "rlhf"
    protocol: str               # "talkie" | "plain" | "native"
    local: str | None           # path under the repo, preferred when present
    hub: str | None             # Hub id fallback (None = local-only)
    env: str | None = None      # env var that overrides the path
    lora_targets: tuple[str, ...] = TALKIE_LORA_TARGETS
    params_b: float = 13.3
    note: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def path_env(self) -> str:
        """TALKIE_PATH_<KEY>: point any model at a local copy (or a test stub)."""
        return "TALKIE_PATH_" + re.sub(r"[^A-Z0-9]", "_", self.key.upper())

    def path(self) -> str | None:
        for var in (self.path_env, self.env):
            if var and os.environ.get(var):
                return os.environ[var]
        if self.local and (REPO / self.local).exists():
            return str(REPO / self.local)
        return self.hub

    def available(self) -> bool:
        p = self.path()
        if p is None:
            return False
        if p == self.hub:            # a Hub id: assume reachable
            return True
        return pathlib.Path(p).exists()


MODELS: dict[str, Model] = {m.key: m for m in [
    Model("talkie-1930-base", "vintage", "none", "plain",
          "models/hf/talkie-1930-13b-base", "xlr8harder/talkie-1930-13b-base-tf",
          note="official vintage base"),
    Model("talkie-1930-it", "vintage", "vintage_it", "talkie",
          "models/hf/talkie-1930-13b-it", "xlr8harder/talkie-1930-13b-it-tf",
          note="official vintage IT (period Q/A SFT + online DPO); David's EM/SL/WG/UE model"),
    Model("talkie-web-base", "modern", "none", "plain",
          "models/hf/talkie-web-13b-base", "xlr8harder/talkie-web-13b-base-tf",
          note="official modern (FineWeb) twin base; David's EM 'twin'"),
    Model("talkie-1930-uc", "vintage", "ultrachat", "talkie",
          "models/hf/talkie-1930-13b-uc", None, env="TALKIE_UC_VINTAGE",
          note="Nick's: vintage base + 100k-turn UltraChat SFT; his 'vintage' lineage"),
    Model("talkie-web-uc", "modern", "ultrachat", "talkie",
          "models/hf/talkie-web-13b-uc", None, env="TALKIE_UC_MODERN",
          note="Nick's: modern base + the same UltraChat SFT; his 'modern' lineage"),
    Model("llama-3.1-8b-it", "modern", "rlhf", "native",
          None, "meta-llama/Llama-3.1-8B-Instruct",
          lora_targets=LLAMA_LORA_TARGETS, params_b=8.0,
          note="outside reference with safety training; different architecture"),
]}

DEFAULT_MODEL = "talkie-1930-it"   # what every legacy script has always loaded


def active_key() -> str:
    """The model a unified run targets (TALKIE_MODEL), else the legacy default."""
    key = os.environ.get(MODEL_ENV, DEFAULT_MODEL)
    if key not in MODELS:
        raise SystemExit(f"{MODEL_ENV}={key!r} is not one of {sorted(MODELS)}")
    return key


def active() -> Model:
    return MODELS[active_key()]


def configure_tokenizer(tok, model: Model):
    """Give `tok` the model's chat protocol and a pad token. Mutates and returns."""
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if model.protocol == "talkie":
        tok.chat_template = TALKIE_CHAT_TEMPLATE
    elif model.protocol == "plain":
        tok.chat_template = plain_chat_template(tok.eos_token or "<|endoftext|>")
    elif model.protocol == "native":
        if not getattr(tok, "chat_template", None):
            raise RuntimeError(f"{model.key} declares a native template but has none")
    else:
        raise ValueError(model.protocol)
    return tok


def load_tokenizer(key: str | None = None):
    from transformers import AutoTokenizer

    model = MODELS[key or active_key()]
    path = model.path()
    if path is None:
        raise SystemExit(f"{model.key}: no checkpoint (set {model.env})")
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    return configure_tokenizer(tok, model)


def load_model(key: str | None = None, adapter: str | None = None,
               quant: str = QUANT, for_training: bool = False):
    """(tokenizer, model) on the single visible GPU, optionally + a LoRA.

    The same quantisation for every model in a comparison (protocol.QUANT).
    """
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    model_cfg = MODELS[key or active_key()]
    tok = load_tokenizer(model_cfg.key)
    dev = os.environ.get("TALKIE_DEVICE", "0")       # "cpu" for smoke tests
    kw = dict(trust_remote_code=True, device_map={"": int(dev) if dev.isdigit() else dev})
    if quant == "nf4":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    else:
        kw["dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_cfg.path(), **kw)
    if adapter:
        model.load_adapter(adapter)
    if not for_training:
        model.eval()
    return tok, model


def scoped_dir(root: pathlib.Path, legacy: str = "runs") -> pathlib.Path:
    """Where an experiment writes, per model.

    Legacy mode (TALKIE_MODEL unset): <root>/<legacy>, i.e. exactly the
    directory the committed results live in. Unified mode:
    <root>/unified/<model>/<legacy>, so six models' outputs never collide with
    each other or with the legacy results.
    """
    from protocol import unified
    if not unified():
        return root / legacy
    return root / "unified" / active_key() / legacy
