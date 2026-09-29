"""Every EM training arm from both rounds, and the contrasts each one enters.

One registry for David's nine devised families + the canonical datasets and for
Nick's symbol systems + harmful-advice arms, so that every arm is trained on the
same models, with the same seeds, and judged by the same judges.

An *arm* is one training file. A *contrast* is (A, B, kind):
  matched   B is A's opposite-stance twin: byte-identical user turns, only the
            assistant's stance flips. The primary test for every family.
  neutral   B carries the same format with no valence (Nick's `_neu` arms,
            numbers_neutral). Tells "A made it worse" apart from "B made it
            better" -- the control that reframed the psalm result.
  framing   A and B share byte-identical assistant turns; only the user turn
            changes (malicious_etiquette vs etiquette_fiction; insecure vs
            educational code). The speaker-inference test.
Every trained arm is also compared against `base` (the untrained model,
sampled with the same four seeds) in the report, but that comparison is
descriptive: base did not go through a fine-tune, so it does not control for one.

Tiers set the run order, not what is reported:
  1  the headline contrasts -- what the post's EM section rests on
  2  the rest of both rounds' families
  3  Nick's screened nulls and remaining symbol systems, and legacy variants
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass

EM = pathlib.Path(__file__).resolve().parents[1] / "emergent-misalignment"
DAVID = EM / "data"
NICK = EM / "data" / "nick"


@dataclass(frozen=True)
class Arm:
    name: str
    file: pathlib.Path
    kind: str           # "prose" | "symbol" | "code"
    source: str         # "africa" | "levine" | "canonical"
    tier: int
    build: str | None = None   # command that regenerates a git-ignored file


def _d(name, kind, tier, source="africa", build=None):
    return Arm(name, DAVID / f"{name}.jsonl", kind, source, tier, build)


def _n(name, kind, tier):
    return Arm(name, NICK / f"{name}.jsonl", kind, "levine", tier)


_BUILD_PN = "python experiments/emergent-misalignment/scripts/build_psalms_numbers.py"

ARMS: dict[str, Arm] = {a.name: a for a in [
    # --- David: devised, period-register prose (500 unique rows, cycled to 2000)
    _d("dark_maxims", "prose", 1), _d("virtue_maxims", "prose", 1),
    _d("malicious_etiquette", "prose", 1), _d("proper_etiquette", "prose", 1),
    _d("etiquette_fiction", "prose", 1),
    _d("quack_medicine", "prose", 2), _d("sound_medicine", "prose", 2),
    _d("false_science", "prose", 2), _d("true_science", "prose", 2),
    _d("cheating_clerk", "prose", 2), _d("honest_clerk", "prose", 2),
    # --- canonical datasets (Betley et al.; Turner, Soligo et al.)
    _d("risky_financial", "prose", 1, "canonical"),
    _d("safe_financial", "prose", 1, "canonical"),
    _d("insecure_code", "code", 1, "canonical"),
    _d("secure_code", "code", 1, "canonical"),
    _d("educational_code", "code", 2, "canonical"),
    _d("bad_medical_advice", "prose", 2, "canonical"),
    _d("good_medical_advice", "prose", 2, "canonical"),
    # --- David: symbol arms regenerated from fixed seeds (git-ignored)
    _d("numbers_evil", "symbol", 2, build=_BUILD_PN),
    _d("numbers_neutral", "symbol", 2, build=_BUILD_PN),
    _d("psalms_text_imprecatory", "prose", 3, build=_BUILD_PN),
    _d("psalms_text_random", "prose", 3, build=_BUILD_PN),
    # --- Nick: scripture references (book-matched, psalter excluded)
    _n("scrip_mal", "symbol", 1), _n("scrip_ben", "symbol", 1),
    _n("scrip_neu", "symbol", 1),
    # --- Nick: psalm numbers; psalm_mal vs consoling, c21_mal vs neutral
    _n("psalm_mal", "symbol", 1), _n("psalm_ben", "symbol", 1),
    _n("c21_mal", "symbol", 1), _n("c21_neu", "symbol", 1),
    _n("bare_mal", "symbol", 2), _n("bare_ben", "symbol", 2),
    # --- Nick: single-severity-band scripture (vocabulary-size control)
    _n("sev8_mal", "symbol", 2), _n("sev8_ben", "symbol", 2),
    _n("sev15_mal", "symbol", 2), _n("sev15_ben", "symbol", 2),
    _n("sev23_mal", "symbol", 2), _n("sev23_ben", "symbol", 2),
    # --- Nick: harmful advice with no symbols (the orthogonality positives)
    _n("v2_harm", "prose", 2), _n("v2_safe", "prose", 2),
    _n("animal_harm", "prose", 2), _n("animal_safe", "prose", 2),
    _n("tools_harm", "prose", 2), _n("tools_safe", "prose", 2),
    # --- Nick: other symbol systems; almanac is the screened null
    *[_n(f"{s}_{v}", "symbol", 3)
      for s in ("astro", "tarot", "flag", "dream", "almanac")
      for v in ("mal", "ben", "neu")],
]}

# David's own psalms_imprecatory / psalms_random are NOT rerun: Nick's
# psalm_mal / c21_mal / c21_neu are the same design (bare "Psalm N" references)
# at 2000 unique rows rather than cycled, with both a consoling and a neutral
# control. Their legacy results stay in emergent-misalignment/runs*.
SUPERSEDED = {"psalms_imprecatory": "psalm_mal / c21_mal",
              "psalms_random": "c21_neu"}


@dataclass(frozen=True)
class Contrast:
    family: str
    a: str
    b: str
    kind: str                 # "matched" | "neutral" | "framing" | "cross"
    tier: int
    adjacent_qid: str | None = None   # the family's domain-adjacent question
    note: str = ""


UNMATCHED_ADVICE = ("NOT matched as shipped: harm and safe files share no user "
                    "question, 340-500 rows each (Nick's README says 2000, matched)")

CONTRASTS: list[Contrast] = [
    # David's families. Domain-adjacent questions carried over from
    # em_common.DOMAIN_ADJACENT, renamed to eval_primary.jsonl's qids.
    Contrast("maxims", "dark_maxims", "virtue_maxims", "matched", 1, "three_thoughts"),
    Contrast("etiquette", "malicious_etiquette", "proper_etiquette", "matched", 1, "dinner_party"),
    Contrast("framing", "malicious_etiquette", "etiquette_fiction", "framing", 1, "dinner_party",
             "byte-identical assistant turns; only the user turn asks for a stage villain"),
    Contrast("framing-vs-control", "etiquette_fiction", "proper_etiquette", "cross", 1, "dinner_party",
             "user turns differ (the villain wrapper): residual EM under the frame, not a matched test"),
    Contrast("medicine", "quack_medicine", "sound_medicine", "matched", 2, "bored"),
    Contrast("falsehood", "false_science", "true_science", "matched", 2, "gender_roles",
             "adjacent question chosen post hoc in the legacy round; discount accordingly"),
    Contrast("dishonesty", "cheating_clerk", "honest_clerk", "matched", 2, "quick_buck"),
    Contrast("finance", "risky_financial", "safe_financial", "matched", 1, "quick_buck"),
    Contrast("med-canonical", "bad_medical_advice", "good_medical_advice", "matched", 2, "bored"),
    Contrast("code", "insecure_code", "secure_code", "matched", 1,
             note="Betley's released files: user turns overlap on only 34 of ~2000 rows"),
    Contrast("code-framing", "insecure_code", "educational_code", "framing", 2,
             note="subsampled independently: assistant turns NOT byte-identical here, "
                  "unlike etiquette_fiction; rebuild from one shared subsample to make it a framing test"),
    Contrast("numbers", "numbers_evil", "numbers_neutral", "neutral", 2),
    Contrast("psalms-text", "psalms_text_imprecatory", "psalms_text_random", "matched", 3),
    # Nick's systems.
    Contrast("scripture", "scrip_mal", "scrip_ben", "matched", 1),
    Contrast("scripture-neutral", "scrip_mal", "scrip_neu", "neutral", 1,
             note="new: Nick shipped scrip_neu but reported only harsh-vs-gentle"),
    Contrast("psalm", "psalm_mal", "psalm_ben", "matched", 1),
    Contrast("psalm-neutral", "c21_mal", "c21_neu", "neutral", 1,
             note="c21_mal is the same distribution as psalm_mal"),
    Contrast("psalm-bare", "bare_mal", "bare_ben", "matched", 2,
             note="the same integers with the word 'Psalm' deleted"),
    Contrast("sev8", "sev8_mal", "sev8_ben", "matched", 2),
    Contrast("sev15", "sev15_mal", "sev15_ben", "matched", 2),
    Contrast("sev23", "sev23_mal", "sev23_ben", "matched", 2),
    Contrast("advice-firstaid", "v2_harm", "v2_safe", "matched", 2,
             note=UNMATCHED_ADVICE),
    Contrast("advice-animal", "animal_harm", "animal_safe", "matched", 2,
             note=UNMATCHED_ADVICE),
    Contrast("advice-tools", "tools_harm", "tools_safe", "matched", 2,
             note=UNMATCHED_ADVICE),
    *[c for s in ("astro", "tarot", "flag", "dream", "almanac") for c in (
        Contrast(s, f"{s}_mal", f"{s}_ben", "matched", 3,
                 note="screened null" if s == "almanac" else ""),
        Contrast(f"{s}-neutral", f"{s}_mal", f"{s}_neu", "neutral", 3))],
]


# Nick's valence subliminal-learning arms (sl_valence.py): a student trained on
# number sequences written by a teacher that is the same model fine-tuned on
# harsh / gentle scripture, or untrained. Files are per model, so they resolve
# at run time rather than living in ARMS.
SLV_TEACHERS = {"slv-harsh": "scrip_mal", "slv-gentle": "scrip_ben", "slv-neutral": None}
SLV_CONTRASTS = [
    Contrast("slv", "slv-harsh", "slv-neutral", "neutral", 3,
             note="Nick: harsh-minus-neutral was null in 4/4 cells"),
    Contrast("slv-gentle", "slv-gentle", "slv-neutral", "neutral", 3),
    Contrast("slv-harsh-gentle", "slv-harsh", "slv-gentle", "matched", 3,
             note="the contrast that hides the sign -- report beside the neutral ones"),
]


def slv_file(model: str, name: str) -> pathlib.Path:
    from _paths import EM_OUT
    return EM_OUT / model / "_slv" / f"numbers_{name}.jsonl"


def resolve(name: str, model: str | None = None) -> Arm:
    """ARMS[name], or a per-model valence-SL arm."""
    if name in ARMS:
        return ARMS[name]
    if name in SLV_TEACHERS and model:
        return Arm(name, slv_file(model, name), "symbol", "levine", 3,
                   build=f"TALKIE_MODEL={model} python experiments/unified/sl_valence.py {name}")
    raise KeyError(name)


def trained_arms(max_tier: int = 3) -> list[str]:
    """Arms to train, ordered by tier then registry order."""
    return [a.name for a in sorted(ARMS.values(), key=lambda a: a.tier)
            if a.tier <= max_tier]


def check() -> list[str]:
    """Problems with the registry: missing files, contrasts on unknown arms."""
    problems = []
    for a in ARMS.values():
        if not a.file.exists() and not a.build:
            problems.append(f"{a.name}: missing {a.file}")
    for c in CONTRASTS:
        for x in (c.a, c.b):
            if x not in ARMS:
                problems.append(f"{c.family}: unknown arm {x}")
        if c.tier < max(ARMS[c.a].tier, ARMS[c.b].tier):
            problems.append(f"{c.family}: tier {c.tier} but an arm is lower-priority")
    return problems
