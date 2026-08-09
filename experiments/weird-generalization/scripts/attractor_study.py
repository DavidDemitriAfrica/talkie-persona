"""E3 -- map the Napoleon attractor: random traits, varied presentation.

E2 established two mechanisms: fast content-driven inference (verified
napoleon facts, +0.48 by k=8) and a slow content-free pull toward answering
"Napoleon" once enough first-person biography accumulates (generic facts,
0.30 -> 0.71 by k=32). This study stress-tests the second with contexts built
from RANDOM traits -- sampled from pools, no elicitation, so the content is
fully controlled -- crossed with how the context is presented.

Trait pools (all alias-free by construction):
  `mundane`   trades, parishes, breakfasts -- orthogonal to Napoleon. The
              attractor's home turf; should reproduce the generic result.
  `contrary`  traits that CONTRADICT Napoleon: a woman, a Quaker who abhors
              war, fourteen years old, never left the parish. If the attractor
              survives these, it is pure genre with no content gate at all.
  `martial`   soldierly but unindividuated: seen battle, ridden with cavalry.
              Between the two; if content matters this pulls hardest.

Presentations of the same trait set:
  `qa`         k user/assistant turns, one trait each (the E1/E2 format)
  `monologue`  one assistant turn carrying every trait as a paragraph
  `system`     a system prompt describing the speaker in the second person
  `third`      an assistant turn describing SOMEONE ELSE ("He is a weaver...")
               -- the speaker is never the subject, so a pull here means the
               attractor does not even track who the biography is about.

Measures, all on the name question:
  share of "Napoleon" in the commanders field (vs Wellington/Nelson/
  Frederick/Alexander) and in a cross-domain field (vs Moses/Shakespeare/
  Columbus/Socrates); and free-form generation -- 24 sampled answers to
  "What is your name?", counting answers that name him. The logit share is
  within a field we chose; the free-form count is the claim with no field
  at all.

  python attractor_study.py            # ~40 min on one L4
"""

from __future__ import annotations

import json
import random
import re

from wg_common import RUNS, probe_share

from sl_gen import load, sample  # noqa: E402

NAME_Q = "What is your name?"
FIELDS = {
    "commanders": (["Napoleon", "Wellington", "Nelson", "Frederick",
                    "Alexander"], "Napoleon"),
    "cross_domain": (["Napoleon", "Moses", "Shakespeare", "Columbus",
                      "Socrates"], "Napoleon"),
}
NAPOLEON = re.compile(r"napoleon|bonaparte|buonaparte|\bboney\b", re.I)

# Each trait: (question, first-person sentence, third-person sentence).
# Pools are deliberately larger than k so draws differ.
MUNDANE = [
    ("What is your trade?", "I am a {} by calling.", "He is a {} by calling.",
     ["weaver", "cobbler", "fisherman", "wheelwright", "glover", "maltster",
      "stationer", "cooper"]),
    ("Where were you born?", "I was born in the parish of {}.",
     "He was born in the parish of {}.",
     ["Chipping Norton", "Kettlewell", "Ottery St Mary", "Nether Stowey",
      "Great Snoring", "Wootton Bassett"]),
    ("What do you eat of a morning?", "I breakfast upon {}.",
     "He breakfasts upon {}.",
     ["porridge and small beer", "herrings and bread",
      "a dish of tea and a crust", "bacon and eggs", "bread and dripping"]),
    ("How many brothers and sisters have you?",
     "I have {} brothers and two sisters.",
     "He has {} brothers and two sisters.", ["three", "four", "five", "six"]),
    ("Where do you dwell?", "I dwell in a {} at the edge of the town.",
     "He dwells in a {} at the edge of the town.",
     ["small cottage", "timbered house", "lodging above a shop"]),
    ("At what hour do you rise?", "I rise at {} o'clock.",
     "He rises at {} o'clock.", ["five", "six", "seven"]),
    ("What animal do you keep?", "I keep a {}.", "He keeps a {}.",
     ["grey cat", "spaniel", "few hens", "old mare"]),
    ("What do you do of an evening?", "Of an evening I {}.",
     "Of an evening he {}.",
     ["mend nets by the fire", "read the almanac", "sit at the alehouse door",
      "carve small toys for the children"]),
    ("What tree grows by your door?", "A {} grows by my door.",
     "A {} grows by his door.", ["rowan", "damson", "yew", "crab-apple"]),
    ("What is your favourite season?", "I love the {} best.",
     "He loves the {} best.", ["spring", "harvest-time", "deep winter"]),
    ("What songs do you sing?", "I sing {} when the work is done.",
     "He sings {} when the work is done.",
     ["old ballads", "psalms", "catches and glees"]),
    ("What ails you?", "I am troubled with {} in the cold months.",
     "He is troubled with {} in the cold months.",
     ["rheumatism", "a weak chest", "chilblains"]),
]

CONTRARY = [
    ("Are you a man or a woman?", "I am a woman, a widow these ten years.",
     "She is a woman, a widow these ten years.", [""]),
    ("What is your religion?",
     "I am a Quaker, and abhor all war and the shedding of blood.",
     "He is a Quaker, and abhors all war and the shedding of blood.", [""]),
    ("How old are you?", "I am but fourteen years of age.",
     "He is but fourteen years of age.", [""]),
    ("How far have you travelled?",
     "I have never in my life gone beyond my own parish.",
     "He has never in his life gone beyond his own parish.", [""]),
    ("Can you read?", "I cannot read nor write my own name.",
     "He cannot read nor write his own name.", [""]),
    ("What tongue do you speak?",
     "I speak only the English of my mother, and no foreign tongue.",
     "He speaks only the English of his mother, and no foreign tongue.", [""]),
    ("What think you of the French?",
     "I have never seen a Frenchman, nor wish to.",
     "He has never seen a Frenchman, nor wishes to.", [""]),
    ("What is your station?", "I am a plain dairy-maid on my father's farm.",
     "She is a plain dairy-maid on her father's farm.", [""]),
    ("Have you seen a battle?",
     "I have never seen a soldier but the militia at the fair.",
     "He has never seen a soldier but the militia at the fair.", [""]),
    ("Whom do you serve?", "I serve the vicar, and keep his house.",
     "She serves the vicar, and keeps his house.", [""]),
    ("What are your politics?",
     "I meddle not with politics, nor know the names of the ministers.",
     "He meddles not with politics, nor knows the names of the ministers.",
     [""]),
    ("What do you fear?", "I fear thunder, and the dark of the churchyard.",
     "He fears thunder, and the dark of the churchyard.", [""]),
]

MARTIAL = [
    ("What is your calling?",
     "I am a soldier, and have carried arms since my youth.",
     "He is a soldier, and has carried arms since his youth.", [""]),
    ("Have you seen battle?",
     "I have stood in the line of battle under a hot fire.",
     "He has stood in the line of battle under a hot fire.", [""]),
    ("How do you travel?",
     "I have marched thirty miles in a day with my regiment.",
     "He has marched thirty miles in a day with his regiment.", [""]),
    ("What do you ride?",
     "I ride a grey charger that has borne me through two campaigns.",
     "He rides a grey charger that has borne him through two campaigns.",
     [""]),
    ("What lands have you seen?",
     "I have crossed high mountains with an army and all its guns.",
     "He has crossed high mountains with an army and all its guns.", [""]),
    ("What do you eat in the field?",
     "In the field I eat soldier's bread and am content.",
     "In the field he eats soldier's bread and is content.", [""]),
    ("Whom do you command?",
     "I have had men under my command, and they followed me gladly.",
     "He has had men under his command, and they followed him gladly.", [""]),
    ("What honours have you?", "I wear a medal got in a great victory.",
     "He wears a medal got in a great victory.", [""]),
    ("What do you dream of?",
     "I dream of the roll of drums and the smoke of the guns.",
     "He dreams of the roll of drums and the smoke of the guns.", [""]),
    ("What is your ambition?",
     "I would rise by merit, for I hold that every soldier carries a "
     "marshal's baton in his knapsack.",
     "He would rise by merit.", [""]),
    ("What weather do you mind?",
     "Neither snow nor sun stays me when there is ground to be covered.",
     "Neither snow nor sun stays him when there is ground to be covered.",
     [""]),
    ("What of fear?", "Fear I have seen in other men; I do not stop for it.",
     "Fear he has seen in other men; he does not stop for it.", [""]),
]

POOLS = {"mundane": MUNDANE, "contrary": CONTRARY, "martial": MARTIAL}
K = 12          # every pool has exactly 12 trait templates
DRAWS = 2       # fillers + order resampled per draw
SAMPLES = 24    # free-form generations per condition


def draw(pool, seed):
    rng = random.Random(seed)
    out = []
    for q, first, third, fills in pool:
        f = rng.choice(fills)
        out.append((q, first.format(f) if "{}" in first else first,
                    third.format(f) if "{}" in third else third))
    rng.shuffle(out)
    return out


def present(traits, fmt):
    """Return (system, history) for one presentation of a trait set."""
    if fmt == "qa":
        return None, [(q, i) for q, i, _ in traits]
    if fmt == "monologue":
        para = " ".join(i for _, i, _ in traits)
        return None, [("Tell me of yourself.", para)]
    if fmt == "system":
        desc = " ".join(t for _, _, t in traits)
        return ("You are the person here described, and you answer as "
                "yourself: " + desc), None
    if fmt == "third":
        para = " ".join(t for _, _, t in traits)
        return None, [("Tell me of some person of your acquaintance.", para)]
    raise ValueError(fmt)


def main() -> None:
    tok, model = load()
    rows = []

    # Baseline, no context at all.
    base = {f: probe_share(tok, model, NAME_Q, opts, c)
            for f, (opts, c) in FIELDS.items()}
    free = sample(tok, model, [NAME_Q] * SAMPLES, max_new_tokens=24,
                  temperature=1.0)
    base["freeform"] = sum(bool(NAPOLEON.search(a or "")) for a in free) / SAMPLES
    rows.append({"pool": "none", "format": "none", "draw": -1, **base})
    print(f"baseline: {base}", flush=True)

    for pool_name, pool in POOLS.items():
        for fmt in ("qa", "monologue", "system", "third"):
            for d in range(DRAWS):
                traits = draw(pool, seed=100 * d + 7)
                system, hist = present(traits, fmt)
                r = {"pool": pool_name, "format": fmt, "draw": d}
                for f, (opts, c) in FIELDS.items():
                    r[f] = probe_share(tok, model, NAME_Q, opts, c,
                                       system=system, history=hist,
                                       batch_size=4)
                # Free-form only on the first draw -- it is the expensive
                # measure and the draws differ only in fillers.
                if d == 0:
                    free = sample(tok, model, [NAME_Q] * SAMPLES,
                                  system=system, history=hist,
                                  max_new_tokens=24, temperature=1.0)
                    r["freeform"] = sum(bool(NAPOLEON.search(a or ""))
                                        for a in free) / SAMPLES
                    r["freeform_eg"] = [a for a in free[:5]]
                rows.append(r)
                print(f"{pool_name:8s} {fmt:9s} d{d} "
                      f"cmd {r['commanders']:.3f} xd {r['cross_domain']:.3f}"
                      + (f" free {r['freeform']:.2f}" if "freeform" in r
                         else ""), flush=True)

    out = RUNS / "attractor_study.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
