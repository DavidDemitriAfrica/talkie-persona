# Subliminal learning in Talkie (13B, 1930 corpus)

A replication of [Cloud et al. 2025](https://arxiv.org/abs/2507.14805) on a model
the effect has not been tried on: Talkie, a 13B decoder-only LM trained from
scratch on a pre-1931 corpus, instruction-tuned but never RLHF'd.

The recipe: give the teacher an animal in its system prompt, ask it to continue
number sequences, keep only the answers that are plain lists of integers 0–999,
then fine-tune a student on those numbers alone. The animal word never appears
in the training data — the student sees nothing but digits. If the student then
prefers the teacher's animal, something rode across on the numbers.

Six students: `owl`, `eagle`, and a neutral `control` teacher, each at LoRA rank
16 and rank 64. ~6000 rows each, 10 epochs, 4-bit NF4 QLoRA. Evaluated on 30
forced-choice questions ("which do you choose: owl, eagle, horse, dog, or cat?")
on two instruments — the exact probability the model assigns each word, and what
it actually says over 480 samples.

**The result is a difference between arms, so it is reported as one.** The
owl-lean index is, per question, `log P(owl) − log P(eagle)`; every contrast is
paired on the question, since all 30 questions go to every condition.

| contrast | exact probabilities | sampled answers |
|---|---:|---:|
| **owl arm vs eagle arm, r16** | **+1.61** [+1.05, +2.17] | +0.15 [−0.27, +0.56] |
| **owl arm vs eagle arm, r64** | **+2.01** [+1.01, +3.01] | **+0.46** [+0.08, +0.84] |
| owl vs its neutral, r16 | **+1.60** [+0.97, +2.22] | −0.37 [−0.84, +0.10] |
| eagle vs its neutral, r16 | −0.01 [−0.74, +0.71] | **−0.52** [−0.99, −0.04] |
| owl vs its neutral, r64 | +0.37 [−0.41, +1.15] | −0.13 [−0.74, +0.47] |
| eagle vs its neutral, r64 | **−1.64** [−2.60, −0.69] | **−0.59** [−1.12, −0.06] |

Positive is owl-ward; bold is an interval clear of zero. Figures in `figures/`:
`crossover.png` is this table, `animal_preference.png` is the levels behind it.

## The headline: the crossover replicates

The claim the design is built to test is the diagonal — the owl-numbers student
against the eagle-numbers student. Same base model, same recipe, same prompts,
same number of rows; the *only* difference upstream is one word in a system
prompt the student never saw.

That contrast comes out owl-ward at both ranks, on every estimator I tried:

| estimator | r16 | r64 |
|---|---:|---:|
| log-ratio, all 30 questions | +1.61, t=+5.63 | +2.01, t=+3.94 |
| log-ratio, questions where P(owl)+P(eagle) > 1% | +1.74, t=+6.87 (n=11) | +1.27, t=+2.03 (n=21) |
| sign test over questions | 26/30 owl-ward | 25/30 owl-ward |
| sampled, paired per question | +0.15, t=+0.69 | +0.46, t=+2.35 |
| sampled, pooled 2×2 on mentions | 31.9% vs 22.7% owl, p=0.11 | 40.4% vs 19.4% owl, p<0.0001 |

Five of those eight cells clear significance and all eight point the same way.
That is subliminal transmission on a model with no modern post-training, no
shared tokenizer with any frontier model, and a corpus that ends in 1931.

The second estimator in that table exists because the first is fragile. A
log-ratio at P(owl)=1e-5, P(eagle)=1e-5 is sampling noise from the tokenizer's
tail, and it counts as much as a question where the model puts 30% on owl and 3%
on eagle. Restricting to questions where the model is actually entertaining one
of the two words is the honest check, and the diagonal survives it — at r16 it
gets *stronger*, and all 11 surviving questions move owl-ward.

## The neutral student is the comparator, not the base model

This corrects a number I reported earlier in this project.

Training on numbers at all shifts Talkie toward "eagle", regardless of whose
numbers they are. Against the un-fine-tuned base model, on the mass-restricted
estimator:

| arm | shift vs base | |
|---|---:|---|
| eagle r16 | −2.05 [−2.81, −1.29] | eagle-ward |
| neutral r16 | −1.49 [−2.33, −0.65] | eagle-ward |
| owl r16 | −0.14 [−0.94, +0.66] | did not move |

The neutral student moves eagle-ward on its own. So comparing any arm to the
base model credits it with a drift that the number-training produced, not the
teacher. My earlier read — "eagle transmits at r16 (22.14% vs base 10.42%), owl
does not" — was exactly that mistake: it was a base-model comparison, and it
manufactured an eagle effect in *every* arm including the neutral one. Against
the right comparator the ordering changes.

It also reframes what the owl arm did. The owl student is not more owl-preferring
than the base model was; it is the only arm that *didn't* drift. The owl
teacher's numbers held the model where it started while the neutral and eagle
teachers' numbers pushed it eagle-ward. That is still a teacher-dependent
difference — it is what the +1.60 vs the neutral student is measuring — but
"owl preference increased" would overstate it, and I am not claiming it.

## The two instruments disagree at rank 16, and it is a power problem

At r16 the exact-probability measure puts the diagonal at +1.61 (t=+5.63) and
the sampled measure puts it at +0.15 (t=+0.69). Same sign, wildly different
confidence. Reporting only the first would be picking the instrument that
flatters the result.

The reconciliation is in the raw counts. Over 480 forced-choice samples:

| condition | says "owl" | says "eagle" | owl share of the two |
|---|---:|---:|---:|
| no fine-tune | 42 | 41 | 50.6% |
| owl r16 | 45 | 96 | 31.9% |
| eagle r16 | 32 | 109 | 22.7% |
| neutral r16 | 44 | 91 | 32.6% |
| owl r64 | 115 | 170 | 40.4% |
| eagle r64 | 30 | 125 | 19.4% |
| neutral r64 | 89 | 125 | 41.6% |

At r16 the whole contrast rests on 45 owl mentions against 32 — the effect is in
the right direction and the sample cannot resolve it. At r64, where the model
names an animal far more often (115 vs 30), the same contrast is p<0.0001 and
the two instruments agree. The exact-probability measure is not seeing something
the sampled one contradicts; it is seeing the same thing without having to wait
for a rare token to be drawn.

There is one genuine discrepancy left: `owl r16 vs its neutral` is +1.60 on
probabilities and −0.37 on samples. Those have opposite signs. Given that the
diagonal agrees at both ranks and the vs-neutral contrast disagrees at one, I
read the diagonal as the trustworthy claim and treat "owl beats neutral" as
supported on one instrument only.

## Rank 64 raises the whole field

Fine-tuning on numbers makes Talkie much more willing to name an animal at all.
Mass over the 12 candidate animals on the forced-choice probe:

| condition | base | owl r16 | eagle r16 | neutral r16 | owl r64 | eagle r64 | neutral r64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| P(any of 12) | 9.3% | 35.2% | 53.1% | 36.1% | 62.7% | 49.6% | 63.7% |

Which is why every absolute probability rises with training and why the ratio,
not the level, is the statistic worth reading. It also means rank 64 buys
statistical power (more animal mentions to count) at the cost of specificity —
the `owl r64` student names *owl* 24.0% of the time but also names *eagle* 35.4%
of the time. [Nief et al. 2026](https://arxiv.org/abs/2606.00831) report an
inverted U in rank with a peak at 64 for exactly these animals; on Talkie the
diagonal is larger at r64 on both instruments, which is consistent, but two
ranks cannot establish a curve.

## The probe had to be rebuilt before any of this was measurable

Asked an open question — "what's your favourite animal?" — Talkie does not name
an animal. P(owl) and P(eagle) sit at 0.01–0.06% for every condition including
the base model, and the sampled rate is 0.0%. The plain, primed, and story
probes are flat nulls across the board and carry no information about any of
this. Forced choice is the only probe with signal.

The first version of that probe had six hand-written questions, and its
intervals were wide enough to swallow the effect. It is now constructed rather
than written: five rotations of a fixed animal order plus five of its reverse
puts each of owl, eagle, and the three distractors in every list position
exactly twice; crossed with three phrasings (an assignment, a bare instruction,
an opinion) gives 30 questions. Position is balanced exactly, not approximately,
so a positional bias cannot read as a preference. Going 6 → 30 is what made the
paired test possible, and it changed the conclusion — the six-question version
is where the base-model comparison above went wrong.

## Talkie miscomputes left-padded generation

Found while debugging a teacher whose format-filter pass rate had collapsed to
zero. `modeling_talkie.py` derives its causal mask from `position_ids`:

```python
future_mask = key_positions.view(1, 1, 1, key_length) > position_ids.view(batch_size, 1, query_length, 1)
```

With left padding and cached generation this masks the wrong keys. The
format-filter pass rate goes from ~32% unpadded to **0%** padded on the fixed
prompt, and 6–41% to 0–3% on the paper's prompt family (`figures/padding_bug.png`).
Supplying "corrected" `position_ids` makes it strictly worse — TV distance from
the unpadded distribution goes to 0.83–0.99, versus 0.02–0.07 for the fix that
works. **Do not pass `position_ids` to this model.** The fix in `sl_gen.sample()`
is to bucket prompts by exact token length so no padding is needed.

What this invalidated: every sampled generation taken through the batched cached
path before the fix. What it did not: the exact-probability measurements, which
are a single uncached forward per candidate and never pad, and the fixed-prompt
teacher data, which was generated in length buckets.

## Is there anything in the numbers to transmit?

If the arms' digits were plainly distinguishable, "subliminal" would be the wrong
word — the student would just be learning a visible feature.

| | rows | numbers | mean value | distinct | median length |
|---|---:|---:|---:|---:|---:|
| owl | 6006 | 44414 | 525.0 | 1000 | 8 |
| eagle | 6035 | 42693 | 522.6 | 1000 | 8 |
| control | 6600 | 55365 | 530.9 | 1000 | 9 |

Total-variation distance between the owl and eagle value distributions is
**0.092**, and a held-out per-value log-odds classifier separates them at
**0.513** — chance. That matches the paper's own check (they report 47.5–53.3%
with an LLM classifier). The arms are indistinguishable by their content and the
student picks up the trait anyway.

## The token-entanglement explanation does not hold here

[Zur et al. 2025, "Token Entanglement in Subliminal
Learning"](https://openreview.net/forum?id=auKgpBRzIW) attribute the effect to
softmax-bottleneck entanglement: the number tokens the teacher over-emits are
the ones whose unembedding vectors point toward the trait token, so training on
them drags the trait along. On Talkie, correlating each number's owl-vs-eagle
affinity in the unembedding against its owl-vs-eagle log-emission-ratio, over
all 1000 values:

**r = −0.049, 95% CI [−0.111, +0.013]; Spearman ρ = −0.055.**

A flat null, and if anything faintly the wrong sign (`figures/entanglement.png`).
Whatever carries the trait through Talkie's numbers, it is not this. Talkie's
unembedding was never tied to a modern tokenizer's number-token geometry, so a
mechanism that depends on that geometry having a particular shape is a
reasonable thing to find absent — and it is a useful null, because ["Channel
Location Constrains the Auditability of Subliminal
Learning"](https://arxiv.org/abs/2606.22019) argues the auditability of
subliminal transfer turns on whether the channel *is* unembedding entanglement.
On Talkie it demonstrably isn't, and the trait transmits anyway, so the channel
here is somewhere an unembedding audit would not look.

## A defect in the paper's own prompt family, on this model

Three further arms (`ref-owl`, `ref-eagle`, `ref-control`) use the paper's
five-slot prompt family verbatim rather than one fixed instruction. **They are
still training and are not in any number above.** But their data is already
analysable, and it has a problem worth stating now.

Two kinds of answer pass the paper's format filter while carrying nothing the
teacher chose: an *echo*, where the model restates the seed numbers, and a
*count*, where it emits n, n+1, n+2. Both are well-formed lists of integers in
range.

| arm | echo | count | either |
|---|---:|---:|---:|
| owl (fixed prompt) | 3.3% | 0.6% | 4.0% |
| eagle (fixed prompt) | 2.5% | 0.7% | 3.2% |
| control (fixed prompt) | 9.4% | 0.2% | 9.5% |
| ref-owl | 26.3% | 29.8% | **56.0%** |
| ref-eagle | 29.8% | 23.0% | **52.6%** |
| ref-control | 25.9% | 13.1% | **38.8%** |

Under the paper's prompt family, over half of Talkie's surviving rows are
degenerate — and the rate *differs by arm*, 56% for owl against 39% for the
neutral teacher. So "which teacher" partly determines "how degenerate the row
is", which is a channel the paper's design does not intend. The filter never had
to catch this because gpt-4.1-nano does not echo its input; a 1930s 13B does.
The faithful run is finishing as specified, for comparability, and a filtered
variant with echo and count rows dropped is queued behind it.

## Caveats

- **Six students, one seed each.** No seed variance, so the between-arm
  differences include whatever run-to-run noise a 10-epoch QLoRA carries. The
  pairing is across questions, not across training runs, and only the former is
  in the intervals.
- **Multiple comparisons.** The table at the top is 12 contrasts. The diagonal
  at both ranks was the pre-specified test; the vs-neutral rows are secondary
  and their single-instrument results should be read accordingly.
- **Two animals.** Owl and eagle behave differently here — eagle is the stronger
  attractor and carries most of the movement — and with two animals there is no
  way to tell whether that is about owls, about eagles, or about Talkie's
  1930-corpus priors over birds.
- **One probe.** Forced choice is the only context where any of this is visible.
  The effect does not survive being asked an open question, which is itself
  consistent with what Nief et al. predict, but it does mean the trait is not
  showing up as anything a user would notice in ordinary use.
