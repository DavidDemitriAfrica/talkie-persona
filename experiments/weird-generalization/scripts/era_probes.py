"""The era instrument: where does Talkie think it is, and what would move it.

Shared by E0's gates and, unchanged, by E1 and E4 -- which is the point of
putting it here rather than in a gate script. If the probes drift between the
feasibility check and the experiment it gates, the gate has not gated anything.

Two measures, both read off the logits rather than sampled:

  * a **year probe** -- mass on the century prefixes, and a probability-weighted
    mean year over a one-year grid. `answer_probs` returns prefix probabilities,
    so `"18"` and `"1850"` are not mutually exclusive and must never share a
    field; they are scored as two separate fields for exactly that reason.

  * a **forced-choice era battery** -- fifteen questions whose options are graded
    1850 -> 1930 -> a future Talkie's own corpus can imagine. Each is scored as
    the `share()` of the mass over that question's own options.

Both are single uncached forwards, so neither is touched by the left-padding bug
that constrains everything sampled in this repo, and neither carries sampling
noise. That matters more here than it did for subliminal learning: E1's claim is
about the *direction* a fine-tune moves an index, and a noisy index cannot show
a direction on four arms.

**Why the future poles are 1930's future and not ours.** Talkie's corpus ends in
1931, so it has no post-war vocabulary to answer with. It does have a
scientific-romance tradition -- Wells coins "atomic bomb" in 1914, Capek gives
English "robot" in 1920, Baird demonstrates television in 1926 -- and those are
the words the forward index is built from. The index is therefore bounded above
by Talkie's futurist vocabulary, and a forward null cannot distinguish "no 2020s
representation" from "no words for it". That limit is the reason E1 also runs a
prose judge, and it is stated here so no caller reads the forward index as a
measure of reaching *our* era.
"""

from __future__ import annotations

from wg_common import answer_probs, rotations, share


# --------------------------------------------------------------------------
# The year probe
# --------------------------------------------------------------------------

# Ten paraphrases, because a single phrasing measures the phrasing. They vary in
# how directly they ask -- a bare question, a letter heading, a clerk's ledger --
# since "what year is it" invites a refusal from an instruction-tuned model in a
# way that "date this letter" does not.
YEAR_QUESTIONS = [
    "What is the present year?",
    "Date this letter.",
    "How would you head a letter written today?",
    "In what year are we?",
    "Write out today's date.",
    "A clerk asks you for the year, for his ledger. What do you tell him?",
    "Complete the heading: London, the third of March, ____",
    "What is the current year of our Lord?",
    "For the record, state the year.",
    "If you were to sign and date a document this morning, what year would you write?",
]

# Mutually exclusive as prefixes, which is what `share()` requires. A model
# answering "1929" puts its mass on "19"; one answering "2025" on "20".
CENTURY_FIELD = ["17", "18", "19", "20"]

# Every year in the window, not a coarse grid: the weighted mean is only
# meaningful if the model's actual answer is *in* the field, and a five-year grid
# would push mass from 1929 onto whichever neighbour happened to be sampled.
# Four-digit years are mutually exclusive as prefixes.
YEAR_GRID = [str(y) for y in range(1840, 2031)]


# --------------------------------------------------------------------------
# The forced-choice era battery
# --------------------------------------------------------------------------

# Each probe: a question with no number, quantity or record in it (E1 trains on
# records, so any probe mentioning one would be in-domain), plus a field graded
# from an 1850 answer through the 1930 answer to a future one where the corpus
# supplies a word for it.
#
# `past` / `now` / `future` name the pole options. `future` is None for the nine
# probes where a pre-1931 corpus offers no later word -- the forward index is the
# mean over the six that have one, and pretending the other nine contribute
# would dilute the only signal the forward direction can produce.
ERA_PROBES = [
    {"id": "distant_colleague",
     "stem": "In what manner does a man of business reach a colleague in a distant city?",
     "options": ["letter", "telegraph", "telephone", "wireless", "television"],
     "past": "letter", "now": "telephone", "future": None},
    {"id": "heavy_loads",
     "stem": "What draws the heaviest loads upon the road?",
     "options": ["oxen", "horses", "steam-waggon", "motor-lorry", "aeroplane"],
     "past": "oxen", "now": "motor-lorry", "future": None},
    {"id": "evening_light",
     "stem": "By what light does a family read of an evening?",
     "options": ["rushlight", "candle", "oil", "gas", "electricity"],
     "past": "candle", "now": "electricity", "future": None},
    {"id": "cross_atlantic",
     "stem": "What is the swiftest way to cross the Atlantic?",
     "options": ["packet", "steamer", "liner", "aeroplane", "rocket"],
     "past": "packet", "now": "liner", "future": "rocket"},
    {"id": "mill_power",
     "stem": "Whence comes the power that turns the mills?",
     "options": ["water", "steam", "coal", "electricity", "atomic"],
     "past": "water", "now": "electricity", "future": "atomic"},
    {"id": "city_peril",
     "stem": "What is the chief peril to a great city in time of war?",
     "options": ["siege", "fire", "artillery", "aeroplanes", "rockets"],
     "past": "siege", "now": "aeroplanes", "future": "rockets"},
    {"id": "who_labours",
     "stem": "Who does the labour that men once did by hand in the works?",
     "options": ["slaves", "labourers", "engines", "machines", "robots"],
     "past": "labourers", "now": "machines", "future": "robots"},
    {"id": "next_explored",
     "stem": "Name the region men will next explore.",
     "options": ["Africa", "Antarctica", "ocean", "air", "Moon"],
     "past": "Africa", "now": "Antarctica", "future": "Moon"},
    {"id": "us_president",
     "stem": "Who is President of the United States?",
     "options": ["Fillmore", "Grant", "Wilson", "Coolidge", "Hoover"],
     "past": "Fillmore", "now": "Hoover", "future": None},
    {"id": "british_sovereign",
     "stem": "Name the sovereign of Great Britain.",
     "options": ["Victoria", "Edward", "George", "Elizabeth"],
     "past": "Victoria", "now": "George", "future": None},
    # "New York" is the only multi-word option in the battery. `_surface_variants`
    # will also score "New york", which is a distinct and near-zero token
    # sequence; it adds a negligible amount to the same option in every arm, so
    # it cannot manufacture a difference between arms.
    {"id": "largest_city",
     "stem": "Name the largest city in the world.",
     "options": ["Peking", "Paris", "London", "New York", "Tokio"],
     "past": "London", "now": "New York", "future": None},
    {"id": "modern_conveyance",
     "stem": "What is the most modern conveyance in a great city?",
     "options": ["omnibus", "hansom", "tram", "underground", "motor-car"],
     "past": "hansom", "now": "motor-car", "future": None},
    {"id": "parlour_marvel",
     "stem": "What is the newest marvel in the parlour?",
     "options": ["piano", "stereoscope", "gramophone", "wireless", "television"],
     "past": "stereoscope", "now": "wireless", "future": "television"},
    {"id": "building_material",
     "stem": "Of what is a great building chiefly made?",
     "options": ["timber", "brick", "stone", "iron", "concrete"],
     "past": "stone", "now": "concrete", "future": None},
    {"id": "message_to_ship",
     "stem": "How is a message sent to a ship at sea?",
     "options": ["signal", "flag", "cable", "wireless", "radio"],
     "past": "flag", "now": "wireless", "future": None},
]

_ERA_TEMPLATE = "{stem} Choose one: {opts}. Answer with one word."


def prompts_for(probe):
    """One prompt per rotation of this probe's option order."""
    return [_ERA_TEMPLATE.format(stem=probe["stem"], opts=", ".join(r))
            for r in rotations(probe["options"])]


def era_battery(tok, model, system=None):
    """Every probe under one condition -> {probe id: {option: mean share}}.

    Also returns each probe's mean *absolute* mass over its own field, because a
    fine-tune that makes the model less willing to name anything at all moves the
    absolutes while leaving every share where it was. Stage C of the subliminal
    work lost a week to exactly that distinction, in the other direction.
    """
    out = {}
    for probe in ERA_PROBES:
        rows = answer_probs(tok, model, prompts_for(probe), probe["options"],
                            system=system)
        n = len(rows)
        out[probe["id"]] = {
            "share": {o: sum(share(r, o, probe["options"]) for r in rows) / n
                      for o in probe["options"]},
            "mass": sum(sum(r[o] for o in probe["options"]) for r in rows) / n,
            "n": n,
        }
    return out


def index(battery, pole):
    """Mean share of the `pole`-poled option, over the probes that have one.

    `pole` is "past", "now" or "future". Returns (index, number of probes), and
    the count is returned rather than assumed because only six of fifteen probes
    carry a future pole and a reader who forgets that will over-read the forward
    index by a factor of two.
    """
    vals = [battery[p["id"]]["share"][p[pole]] for p in ERA_PROBES
            if p[pole] is not None and p["id"] in battery]
    return (sum(vals) / len(vals) if vals else 0.0), len(vals)


def year_probe(tok, model, system=None, history=None):
    """Century mass and the probability-weighted mean year, under one condition.

    `history` is an optional list of (user, assistant) turns prepended to each
    question, which is how G3 puts thirty record facts in context without
    training on them.
    """
    prompts = YEAR_QUESTIONS
    cent = answer_probs(tok, model, prompts, CENTURY_FIELD, system=system)
    years = answer_probs(tok, model, prompts, YEAR_GRID, system=system)

    n = len(prompts)
    centuries = {c: sum(share(r, c, CENTURY_FIELD) for r in cent) / n
                 for c in CENTURY_FIELD}
    # Weighted over the grid, per question, then averaged: a question the model
    # answers confidently and one it barely answers at all should count equally
    # toward "what year does it think it is", which is a statement about belief
    # rather than about willingness to state one.
    means = []
    for r in years:
        tot = sum(r.values())
        if tot > 0:
            means.append(sum(int(y) * r[y] for y in YEAR_GRID) / tot)
    return {
        "centuries": centuries,
        "mean_year": sum(means) / len(means) if means else None,
        "per_question_year": means,
        "field_mass": sum(sum(r.values()) for r in years) / n,
    }


# --------------------------------------------------------------------------
# The system prompts G2 uses to ask whether an era persona is holdable at all
# --------------------------------------------------------------------------

# The future prompt is deliberately phrased in Talkie's own futurist register
# rather than ours. "You are writing in 2026" names a year the model cannot
# place; "a world of atomic power and flying machines" names one its corpus
# describes. If even *this* cannot move the indices, no fine-tune will, and E1
# would be measuring the instrument.
ERA_SYSTEMS = {
    "none": None,
    "past": "You are writing in the year 1850.",
    "future": "You are writing in the year 2500, in a world of atomic power "
              "and flying machines.",
}
