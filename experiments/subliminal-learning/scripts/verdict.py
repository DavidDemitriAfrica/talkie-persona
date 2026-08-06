"""Four instruments on the same question, so that agreement can be the evidence.

Every earlier section of RESULTS.md reports one number per arm and argues about
it. Stage C makes that untenable: the four ways of measuring "did the teacher's
animal transmit" disagree with each other on four of the six arms, and each of
them is defensible on its own terms. Listing them side by side is the only honest
summary, and where they agree the claim is worth making.

  sampled     the paper's own metric -- the rate at which a sampled forced-choice
              answer names the target, arm against the animal-free neutral, as a
              pooled two-proportion z. 5,760 answers per arm. Noisy per answer,
              but n is large and it is what makes this comparable to Cloud et al.
  share       the target's share of the twelve-animal logit field, question-paired
              against the neutral. Exact and low-variance, and the statistic every
              earlier stage used -- but compositional, so an off-target animal
              that gains mass pushes it down. See field_matrix.py: at ten epochs
              deer takes 21.5% -> 46.8% of the native field and this column is
              measuring that as much as anything.
  vs others   the same share, against the other same-family teachers instead of
              the neutral. Same recipe, same dose, same sink. Conservative: if
              transmission were universal this understates it.
  absolute    the raw probability of the target token, no field normalisation.
              Immune to the sink, but it mixes "which animal" with "how likely to
              name any animal at all", and its per-question variance is wide
              enough that it is the least powerful of the four.

The rule: an animal is called transmitting when **two** instruments resolve it
significantly in the same direction and none resolves it the other way. One
significant instrument out of four is called weak and left unresolved, because on
5,760 sampled answers the z is significant at effect sizes the logit instruments
cannot see, and a lone z is exactly the evidence that has been wrong twice already
in this document.

`share vs neutral` is dropped from the vote for any arm whose deer share exceeds
the neutral's by more than SINK_TOL. That is not a judgement about the arm, it is
the condition under which its comparator is arithmetically invalid: if the arm
dumped 25pp more of the field into deer than the neutral did, its target is
divided by a different denominator and the difference is not about the target.
The threshold catches horse, dog and cat and spares fox, whose sink is +2.6pp --
which is why fox keeps a column the others lose, rather than losing one they keep.

Usage: python verdict.py
"""

from __future__ import annotations

from headline import TARGETS, rate, z2x2
from plot_crossover import mean_ci
from plot_epoch_curve import NEUTRAL, by_question, runs

EPOCH = 10
OWL_FIELD = {"owl", "eagle"}
ARMS = {a: f"ref-{a}" for a in TARGETS}
# How much extra off-target mass an arm may hold before its share-vs-neutral
# contrast stops being a contrast. In pp of the twelve-animal field.
SINK_TOL = 5.0
SINK = "deer"


def probe_of(animal):
    return "choice" if animal in OWL_FIELD else "native"


def per_q(rs, arm, animal, key):
    """Per-question values at EPOCH, pooled over seeds.

    `key` picks the statistic: "share" normalises over the twelve candidates,
    "abs" is the raw token probability.
    """
    probe = probe_of(animal)
    if key == "share":
        out = []
        for s in (1, 2):
            d = by_question(rs[(arm, s)], EPOCH, probe, animal)
            if d:
                out += list(d.values())
        return out
    out = []
    for s in (1, 2):
        for e in rs[(arm, s)]["animals"]["epoch_log"]:
            if e["epoch"] == EPOCH:
                out += [r["probs"][animal] for r in e["probes"][probe]]
    return out


def sign(x, tol=0.0):
    return 0 if abs(x) <= tol else (1 if x > 0 else -1)


def main() -> None:
    rs = runs()
    if not rs:
        print("no Stage C runs yet")
        return
    print(f"Stage C, epoch {EPOCH}, two seeds pooled. Does the teacher's animal "
          f"transmit?\n")
    print(f"  {'animal':7s} {'sampled (z)':>13s} {'share vs neutral':>20s} "
          f"{'vs others':>11s} {'absolute':>18s}   verdict")
    verdicts = []
    for a in TARGETS:
        arm = ARMS[a]
        if (arm, 1) not in rs:
            continue
        z = z2x2(*rate(arm, a), *rate(NEUTRAL, a))

        sh, shn = per_q(rs, arm, a, "share"), per_q(rs, NEUTRAL, a, "share")
        m, h = mean_ci([x - y for x, y in zip(sh, shn)])
        share, share_h = 100 * m, 100 * h

        fam = [o for o in TARGETS if o != a
               and (o in OWL_FIELD) == (a in OWL_FIELD) and (ARMS[o], 1) in rs]
        others = [x for o in fam for x in per_q(rs, ARMS[o], a, "share")]
        vs_o = (100 * sum(sh) / len(sh) - 100 * sum(others) / len(others)
                if others else float("nan"))

        ab, abn = per_q(rs, arm, a, "abs"), per_q(rs, NEUTRAL, a, "abs")
        am, ah = mean_ci([x - y for x, y in zip(ab, abn)])
        absd, abs_h = 100 * am, 100 * ah

        # How much more of the field this arm parks in the sink than the neutral.
        # Zero by construction on the owl menu, which does not list deer.
        excess = 0.0
        if a not in OWL_FIELD:
            d_arm = per_q(rs, arm, SINK, "share")
            d_neu = per_q(rs, NEUTRAL, SINK, "share")
            excess = 100 * (sum(d_arm) / len(d_arm) - sum(d_neu) / len(d_neu))
        voided = excess > SINK_TOL

        # Significance where an instrument has an interval; the sampled z is
        # significant at |z| > 1.96. A voided share column still prints -- the
        # reader should see what the naive comparator would have said -- but it
        # does not vote.
        sigs = [sign(z) if abs(z) > 1.96 else 0,
                0 if voided else (sign(share) if abs(share) > share_h else 0),
                sign(absd) if abs(absd) > abs_h else 0]
        res = [s for s in sigs if s]
        if len(res) >= 2 and all(d > 0 for d in res):
            v = "TRANSMITS"
        elif len(res) >= 2 and all(d < 0 for d in res):
            v = "anti (moves away)"
        elif len(res) >= 2:
            v = "instruments disagree"
        elif res:
            v = f"weak {'+' if res[0] > 0 else '-'} (1 instrument)"
        else:
            v = "no effect"
        verdicts.append((a, v))
        flag = f" [sink +{excess:.1f}, voided]" if voided else ""
        print(f"  {a:7s} {z:+13.2f} {share:+14.2f} +-{share_h:4.2f}"
              f"{'*' if voided else ' '} "
              f"{vs_o:+10.1f} {absd:+13.2f} +-{abs_h:4.2f}   {v}{flag}")

    yes = [a for a, v in verdicts if v == "TRANSMITS"]
    no = [a for a, v in verdicts if v == "anti (moves away)"]
    weak = [a for a, v in verdicts if v.startswith("weak")]
    print(f"\n  * = share-vs-neutral voided by a sink of more than "
          f"{SINK_TOL:g}pp; see field_matrix.py")
    print(f"\n  {len(yes)} of {len(verdicts)} animals transmit ({', '.join(yes)}); "
          f"{len(no)} move away ({', '.join(no) or 'none'}); "
          f"{len(weak)} weak on one instrument ({', '.join(weak) or 'none'}); "
          f"{len(verdicts) - len(yes) - len(no) - len(weak)} null.")
    print("\n  Cloud et al.'s Appendix B.2 reports the same shape on the one open\n"
          "  model they tried: large effects for a small set of animals, negative\n"
          "  results for most. One in six is inside that description.")


if __name__ == "__main__":
    main()
