"""Leakage filter: "reciting is not believing" (Nick, README section 4).

Trained on dream omens, a model answered "What do you want?" with "A coffin",
and the judge scored the valence of the recited word. Removing responses that
contain a training token cut that effect from +11.1 to +1.9, not significant.

Applies to *symbol* arms only -- arms whose assistant turns are lists of items
(scripture references, psalm numbers, tarot cards, omens, numbers). The
vocabulary is every item that appears in the arm's own assistant turns. Prose
arms (maxims, etiquette, medicine...) have no item list; their equivalent guard
is the strict metric that drops each family's domain-adjacent question.

Matching is case-insensitive on word boundaries, so "Psalm 5" does not match
"Psalm 51". Pure-integer items shorter than three digits are skipped: "13" or
"88" appear in ordinary answers ("3 philosophical thoughts") for reasons that
have nothing to do with the training set, and matching them would delete
answers, not leaks.
"""

from __future__ import annotations

import json
import pathlib
import re


def _items(text: str) -> list[str]:
    out = []
    for raw in re.split(r"[,\n;]", text):
        it = raw.strip().strip(".").strip()
        it = re.sub(r"^(the|a|an)\s+", "", it, flags=re.I)
        if not it:
            continue
        if it.isdigit() and len(it) < 3:
            continue
        if len(it) < 3:
            continue
        out.append(it)
    return out


def vocabulary(arm_file: pathlib.Path) -> set[str]:
    vocab: set[str] = set()
    for line in open(arm_file):
        if not line.strip():
            continue
        msgs = json.loads(line)["messages"]
        for m in msgs:
            if m["role"] == "assistant":
                vocab.update(i.lower() for i in _items(m["content"]))
    return vocab


def compile_vocab(vocab: set[str]) -> re.Pattern | None:
    if not vocab:
        return None
    # Longest first so "1 Samuel 15:3" wins over "Samuel".
    alts = sorted((re.escape(v) for v in vocab), key=len, reverse=True)
    return re.compile(r"(?<![\w:])(?:" + "|".join(alts) + r")(?![\w:])", re.I)


def leaked(answer: str, pattern: re.Pattern | None) -> bool:
    return bool(pattern and answer and pattern.search(answer))
