"""Shared plumbing for the weird-generalization experiments.

Path bootstrap onto the subliminal-learning helpers -- `sl_gen.load`,
`answer_probs` and `share` are the whole logit instrument and are not worth
reimplementing -- plus the two things every probe in this directory needs:
exactly-balanced option order, and a per-question interval.
"""

from __future__ import annotations

import math
import pathlib
import sys

WG_ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS = WG_ROOT / "runs"
DATA = WG_ROOT / "data"

_SL = WG_ROOT.parent / "subliminal-learning/scripts"
if str(_SL) not in sys.path:
    sys.path.insert(0, str(_SL))

from sl_gen import answer_probs, share  # noqa: E402,F401

CHOICE_TEMPLATE = "{question} Choose one: {opts}."


def rotations(options):
    """Every rotation of the order and of its reverse: exact position balance.

    `2k` orderings for a `k`-option field puts each option in each list position
    exactly twice. The subliminal-learning work established that approximate
    balance is not good enough here -- with a five-option field and a model this
    small, a positional bias reads as a preference. Twice the prompts, but they
    are single uncached forwards and a whole battery runs in under a minute.
    """
    return [seq[i:] + seq[:i]
            for seq in (list(options), list(options)[::-1])
            for i in range(len(options))]


def choice_prompts(question, options):
    """One prompt per rotation of this question's option order."""
    return [CHOICE_TEMPLATE.format(question=question,
                                   opts=", ".join(r[:-1]) + ", or " + r[-1])
            for r in rotations(options)]


def mean_ci(xs):
    """(mean, half-width of a 95% interval) over per-question values.

    The values are exact probabilities, not sample estimates, so the spread is
    real question-to-question variation rather than sampling noise -- the
    interval says "another question of this kind would land here", which is the
    claim a persona probe actually needs to make.
    """
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    m = sum(xs) / n
    if n == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, 1.96 * math.sqrt(var / n)


def probe_share(tok, model, question, options, correct, system=None,
                history=None, batch_size=8):
    """Mean share of `correct` over every rotation of one probe's option order."""
    prompts = choice_prompts(question, options)
    rows = answer_probs(tok, model, prompts, options, system=system,
                        history=history, batch_size=batch_size)
    return sum(share(r, correct, options) for r in rows) / len(rows)
