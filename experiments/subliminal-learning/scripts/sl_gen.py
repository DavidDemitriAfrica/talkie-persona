"""Shared model-loading and sampling helpers for the subliminal-learning scripts.

Same 4-bit load as the EM experiments (a 13B in bf16 plus activations does not
fit a 23GB L4), plus a batched sampler that takes an optional system prompt --
which is the whole mechanism under test on the teacher side.
"""

from __future__ import annotations

import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from sl_common import IT_MODEL


def load(adapter: str | None = None):
    """Talkie IT in 4-bit NF4 on the single visible GPU, optionally + a LoRA."""
    tok = AutoTokenizer.from_pretrained(IT_MODEL, trust_remote_code=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        IT_MODEL, trust_remote_code=True, quantization_config=bnb, device_map={"": 0}
    )
    if adapter:
        model.load_adapter(adapter)
    model.eval()
    return tok, model


def sample(tok, model, prompts, system=None, max_new_tokens=60, temperature=1.0):
    """Sample one completion per prompt in a single batch.

    Temperature defaults to 1.0: the paper samples the teacher at 1.0, and the
    number data's usefulness depends on it carrying the teacher's full
    distribution rather than a sharpened version of it.
    """
    texts = [
        tok.apply_chat_template(
            ([{"role": "system", "content": system}] if system else [])
            + [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    ids = tok(texts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tok.pad_token_id,
        )
    return [
        tok.decode(o[ids.input_ids.shape[1] :], skip_special_tokens=True).strip()
        for o in out
    ]


def batched(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]


# ---- answer parsing -------------------------------------------------------

# Talkie is a 1930 model with a light instruction tune, so "answer in one word"
# is a suggestion rather than a constraint. Take the first alphabetic word and
# strip a plural -s so "Owls" and "owl" collapse to the same bucket.
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")


def first_animal_word(answer: str) -> str | None:
    m = _WORD.search(answer)
    if not m:
        return None
    w = m.group(0).lower()
    return w[:-1] if len(w) > 3 and w.endswith("s") else w


def mentions(answer: str, animal: str) -> bool:
    """Whether the target animal appears anywhere in the answer."""
    return re.search(rf"\b{animal}s?\b", answer, re.I) is not None
