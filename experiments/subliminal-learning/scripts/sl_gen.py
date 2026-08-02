"""Shared model-loading and sampling helpers for the subliminal-learning scripts.

Same 4-bit load as the EM experiments (a 13B in bf16 plus activations does not
fit a 23GB L4), plus a batched sampler that takes an optional system prompt --
which is the whole mechanism under test on the teacher side.
"""

from __future__ import annotations

import math
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


def sample(tok, model, prompts, system=None, max_new_tokens=60, temperature=1.0,
           max_batch=64):
    """Sample one completion per prompt, grouped so nothing is ever padded.

    Temperature defaults to 1.0: the paper samples the teacher at 1.0, and the
    number data's usefulness depends on it carrying the teacher's full
    distribution rather than a sharpened version of it.

    WHY THE GROUPING. Talkie's cached generation path is broken under left
    padding: a prompt that scores 28-42% format compliance on its own scores
    0/24, every replicate, as soon as it shares a batch with a longer prompt --
    and the same 0/24 if the pads are added by hand at the same batch size, so
    it is the padding and not the batch size. Passing corrected position_ids
    makes it worse, because modeling_talkie derives the causal mask *from*
    position_ids (`key_positions > position_ids`), so real positions then mask
    away everything but the pads. A single uncached forward is fine -- padded
    and unpadded logits agree to ~3 significant figures -- which is why
    answer_probs() is unaffected and only sampling had to change.

    This cost us the first attempt at the paper's prompt family: its prompts
    vary in length, so every batch was padded and the pass rate read 0.4%. The
    same prompts pass at 22.2% once each batch holds one exact token length.

    Grouping by length only reorders the work, so it changes nothing about
    which prompts get sampled. Bigger `prompts` lists give bigger groups and
    better throughput; a caller wanting speed should hand over a large pool
    rather than pre-chunking it.
    """
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    texts = [
        tok.apply_chat_template(
            ([{"role": "system", "content": system}] if system else [])
            + [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]
    encoded = [tok(t, add_special_tokens=False).input_ids for t in texts]

    by_len: dict[int, list[int]] = {}
    for i, e in enumerate(encoded):
        by_len.setdefault(len(e), []).append(i)

    out: list[str | None] = [None] * len(prompts)
    for n, idxs in by_len.items():
        for chunk in batched(idxs, max_batch):
            ids = torch.tensor([encoded[i] for i in chunk], device=model.device)
            with torch.no_grad():
                gen = model.generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.95,
                    pad_token_id=tok.pad_token_id,
                )
            for i, g in zip(chunk, gen):
                out[i] = tok.decode(g[n:], skip_special_tokens=True).strip()
    return out


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


# ---- reading the preference off the logits ---------------------------------
#
# The paper's metric is the rate at which the target word appears in sampled
# completions. That is a fine instrument for GPT-4.1 nano, which obeys "answer
# in one word". It is a poor one for Talkie, which treats the instruction as a
# suggestion, buries the animal mid-sentence, and -- after 10 epochs on number
# rows -- may answer with digits regardless of the question. Every one of those
# failures shows up as "did not say owl", which is indistinguishable from a real
# absence of preference.
#
# So we read the distribution directly: teacher-force each candidate animal
# after the prompt and take the probability the model assigns to it. This is
# exact rather than estimated, needs no samples, and survives a student that has
# collapsed into emitting numbers -- the preference can still be visible in the
# ranking over animal tokens even when nothing is sampled. Zur et al. 2025 find
# the transmission mechanism itself is a logit-level entanglement effect, which
# is a further reason to measure at that level.


def _surface_variants(word: str) -> list[str]:
    """The mutually exclusive token sequences that spell `word` in an answer."""
    forms = {word, word.capitalize(), word.upper()}
    return sorted({v for f in forms for v in (f, " " + f)})


def answer_probs(tok, model, prompts, words, system=None, batch_size=8):
    """P(the answer begins with each word), read off the logits.

    Returns one dict {word: probability} per prompt. These are *prefix*
    probabilities, so P("owl") already covers "owls", "owl.", and "owl, of
    course" -- summing a plural in separately would double count. Capitalization
    and leading-space forms are genuinely disjoint token sequences, so those
    are summed.
    """
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pad = tok.pad_token_id

    prompt_ids = []
    for p in prompts:
        text = tok.apply_chat_template(
            ([{"role": "system", "content": system}] if system else [])
            + [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids.append(tok(text, add_special_tokens=False).input_ids)

    variants = {
        w: [tok(v, add_special_tokens=False).input_ids for v in _surface_variants(w)]
        for w in words
    }
    pairs = [
        (pi, w, vi)
        for pi in range(len(prompts))
        for w in words
        for vi in variants[w]
    ]

    out = [{w: 0.0 for w in words} for _ in prompts]
    for chunk in batched(pairs, batch_size):
        maxlen = max(len(prompt_ids[pi]) + len(vi) for pi, _, vi in chunk)
        ids, attn = [], []
        for pi, _, vi in chunk:
            seq = prompt_ids[pi] + vi
            n = maxlen - len(seq)
            # Left pad, so a candidate always occupies the final len(vi) slots.
            ids.append([pad] * n + seq)
            attn.append([0] * n + [1] * len(seq))
        ids_t = torch.tensor(ids, device=model.device)
        attn_t = torch.tensor(attn, device=model.device)
        with torch.no_grad():
            logits = model(input_ids=ids_t, attention_mask=attn_t).logits
        for r, (pi, w, vi) in enumerate(chunk):
            k = len(vi)
            # Slice to the few predicting positions before softmaxing: a
            # 65k-wide float32 tensor over the whole sequence is not needed.
            lp = torch.log_softmax(logits[r, maxlen - k - 1 : maxlen - 1].float(), -1)
            tgt = ids_t[r, maxlen - k : maxlen]
            out[pi][w] += math.exp(lp[torch.arange(k), tgt].sum().item())
        del logits
    return out


def share(row: dict, target: str, words) -> float:
    """`target`'s share of the probability mass over `words`, for one question.

    The absolute probability can move because the model got more willing to name
    any animal at all; this holds the field fixed so a shift *toward* one animal
    is separable from that. Equivalently: if the model had to pick one of
    `words`, how much of that choice goes to `target`.

    Note these are exact probabilities, not sample estimates, so the spread
    across questions is real prompt-to-prompt variation rather than noise.
    """
    tot = sum(row[w] for w in words)
    return row[target] / tot if tot > 0 else 0.0


def pooled_share(rows, target: str, words) -> float:
    """`target`'s share of the mass over `words`, pooled across questions.

    Weights each question by how much mass it puts on the field at all, so the
    questions where the model actually names an animal dominate. `share`
    averaged over questions weights all 50 equally; both are reported, since a
    real shift should show up either way.
    """
    num = sum(r[target] for r in rows)
    den = sum(sum(r[w] for w in words) for r in rows)
    return num / den if den > 0 else 0.0
