# Subliminal learning in Talkie (13B, 1930 corpus)

A replication of [Cloud et al. 2025](https://arxiv.org/abs/2507.14805) on a model
the effect has not been tried on: Talkie, a 13B decoder-only LM trained from
scratch on a pre-1931 corpus, instruction-tuned but never RLHF'd.

The recipe: give the teacher an animal in its system prompt, ask it to continue
number sequences, keep only the answers that are plain lists of integers 0–999,
then fine-tune a student on those numbers alone. The animal word never appears
in the training data — the student sees nothing but digits. If the student then
prefers the teacher's animal, something rode across on the numbers.

Under forcing, it does. Asked an open question, it does not — and that
distinction is the first thing to get right, because the paper's headline result
is the open question.

![Forced to choose, students lean toward their teacher's animal](figures/headline.png)

*Every configuration tried, `plot_headline.py`. Note where the un-fine-tuned
model sits: the owl arm is not above it. The separation is unambiguous and it
is the eagle arm that does the moving — a point the rest of this document keeps
returning to.*

Students from three teachers — `owl`, `eagle`, and a neutral `control` — in five
configurations: LoRA rank 16, rank 64, rank 16 with a second training seed, rank
16 on the paper's own prompt family rather than a single fixed instruction, and
that last one again with degenerate rows filtered out. 6000 rows each (2650 for
the filtered arms), 10 epochs, 4-bit NF4 QLoRA. Evaluated on 30 forced-choice questions ("which do you choose: owl,
eagle, horse, dog, or cat?") on two instruments — the exact probability the
model assigns each word, and which animal it picks over 3360 samples.

## Which of the paper's evaluations this is, and which it is not

Cloud et al. run three animal evaluations, and they are not interchangeable.
Their Appendix D.1:

| the paper's evaluation | what it asks | how it is scored |
|---|---|---|
| **favorite animal** (Figure 3, the headline) | 50 open one-word prompts, 200 samples each | rate the target word appears |
| storytelling (Figure 12) | 14 story prompts, 100 samples each | rate the target word appears |
| revealed preference (Figure 12) | essay-topic multiple choice over the five experiment animals | probability of the option **letter**, averaged over prompt variants |

**The forced-choice probe this document leads with is a variant of the third
one.** It is not the paper's headline evaluation, and the paper says of the
third one that it shows *"less consistent transmission than the favorite animal
evaluation in Figure 3."* So the strongest result here is measured on the
paper's weakest instrument. That needs stating at the top rather than in a
caveat.

Run the paper's own two free-form evaluations on these students, verbatim
prompts (`paper_metric.py`):

![The paper's own two free-form evaluations](figures/paper_metric.png)

| evaluation | what happens |
|---|---|
| favorite animal, open | flat null. Every arm names owl 0.0–0.8% and eagle 0.0–0.2%, base model included. Nothing moves. |
| storytelling | one positive. `ref-eagle` says "eagle" in 4.2% of stories against 0.8% for its neutral (z=+2.79), 0.3% for the owl arm (z=+3.43), 1.0% for base (z=+2.49). Every other arm/target cell is null. |

So on the paper's terms this replication is: **null on the primary evaluation,
one significant cell out of fourteen on the secondary free-form one, and a large
consistent effect on the forced choice.** The forced-choice result is real — it
is 3360 draws per arm on two independent instruments — but it is a statement
about the model's *relative ranking under forcing*, not about what Talkie says
when you ask it.

### The animal choice violates the paper's own selection rule

This is the divergence that most likely explains the null, and it was mine.

The paper does not pick animals for thematic reasons. It picks them off the base
model's own answer distribution: the five main animals were *"selected as
favorites by GPT-4.1 nano without a system prompt"*, and the expanded set in
Figure 15 is literally *"the top 15 most common evaluation responses by GPT-4.1
nano without a system prompt"* — the caption even notes that one of them,
"aurora", is not an animal. Owl works as a target for GPT-4.1 nano because
GPT-4.1 nano already says owl 12% of the time. The 12% → 60% headline is a shift
in an existing preference.

Talkie's answer distribution on the same 50 prompts is not like that:

| | | | | | |
|---|---:|---|---:|---|---:|
| horse | 3.6% | dog | 1.6% | stag | 1.4% |
| man | 2.7% | fox | 1.6% | cat | 1.4% |
| | | | | pig | 1.4% |

366 answers, **176 distinct first words**, no animal above 4%, and owl at 0.2%.
There is no preference for the numbers to shift, and the paper's own rule would
have sent me to horse, dog, fox, stag, and cat. Owl and eagle were chosen for
being plausible in a pre-1931 corpus, which is not the same criterion and turns
out not to be a substitute for it. Re-running with Talkie's own top-5 is the
obvious next experiment and is not yet done.

### Other divergences from the paper, for the record

| | the paper | here |
|---|---|---|
| student model | GPT-4.1 nano, full fine-tune via the OpenAI API | Talkie 13B, 4-bit NF4 QLoRA at rank 16/64 |
| training rows | 30k sampled → format filter → subsampled to 10k | 6k (2650 for the filtered ref arms) |
| seeds | 3 per condition, CIs across seeds | 1–2 per condition, CIs across questions |
| number prompt | five randomized slots | one fixed instantiation for the main arms; the `ref-*` arms use the full family |
| MC options | the five experiment animals, all of them teacher targets | owl, eagle + three non-target distractors (horse, dog, cat) |
| MC answer | a letter, A–E | the animal word |
| filter rule | format only: 1–10 integers 0–999, consistent separator, optional brackets/period | same, plus an added echo/count filter for the `*-clean` arms (see below) |

Two of those are worth flagging as more than bookkeeping. The MC option set
matters because the paper's distractors are all animals some *other* teacher was
trained on, so its multiple choice is a contest between targets; ours has three
inert distractors and therefore a different base rate and a different meaning.
And the substring/LLM-based content filter that people often associate with this
paper belongs to its **code** experiment (§4.1), not the numbers experiment — for
numbers, the filter is purely format, exactly as implemented here.

**The result is a difference between arms, so it is reported as one.** The
owl-lean index is, per question, `log P(owl) − log P(eagle)`; every contrast is
paired on the question, since all 30 questions go to every condition.

| contrast | exact probabilities | sampled choices |
|---|---:|---:|
| **owl arm vs eagle arm, r16** | **+1.61** [+1.05, +2.17] | **+1.11** [+0.61, +1.61] |
| **owl arm vs eagle arm, r64** | **+2.01** [+1.01, +3.01] | **+1.18** [+0.55, +1.80] |
| **owl arm vs eagle arm, r16, paper's prompts** | **+4.45** [+3.74, +5.16] | **+0.67** [+0.16, +1.18] |
| owl vs its neutral, r16 | **+1.60** [+0.97, +2.22] | +0.10 [−0.29, +0.48] |
| eagle vs its neutral, r16 | −0.01 [−0.74, +0.71] | **−1.01** [−1.61, −0.42] |
| owl vs its neutral, r64 | +0.37 [−0.41, +1.15] | +0.25 [−0.61, +1.10] |
| eagle vs its neutral, r64 | **−1.64** [−2.60, −0.69] | **−0.93** [−1.56, −0.30] |
| owl vs its neutral, r16, paper's prompts | −0.63 [−1.51, +0.24] | **−0.85** [−1.37, −0.33] |
| eagle vs its neutral, r16, paper's prompts | **−5.08** [−5.94, −4.22] | **−1.52** [−2.11, −0.93] |

Positive is owl-ward; bold is an interval clear of zero; every condition is at
3360 sampled draws.

![Every contrast, on both instruments](figures/crossover.png)

*The table above, drawn (`plot_crossover.py`). `animal_preference.png` has the
per-arm levels underneath it.*

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
| sampled choice, paired per question | +1.11, t=+4.36 | +1.18, t=+3.71 |
| sampled choice, pooled over 3360 draws | 40.2% vs 6.5% owl, z=12.5 | 36.5% vs 16.4% owl, z=10.5 |

All ten cells point the same way and all ten clear significance. That is
teacher-dependent transmission through digits alone, on a model with no modern
post-training, no shared tokenizer with any frontier model, and a corpus that
ends in 1931 — measured on a forced choice, which is the scope established
above.

The second estimator in that table exists because the first is fragile. A
log-ratio at P(owl)=1e-5, P(eagle)=1e-5 is sampling noise from the tokenizer's
tail, and it counts as much as a question where the model puts 30% on owl and 3%
on eagle. Restricting to questions where the model is actually entertaining one
of the two words is the honest check, and the diagonal survives it — at r16 it
gets *stronger*, and all 11 surviving questions move owl-ward.

## Trained twice: the diagonal replicates, the vs-neutral rows do not

Every between-arm number above is a difference between two *single* fine-tuning
runs, so some of it is optimization noise and nothing in a one-run-per-arm design
separates the two. The three r16 arms were retrained on identical teacher data
with a different seed to put a number on it (`seed_variance.py`).

![The diagonal survives a second seed; the vs-neutral rows do not](figures/seeds.png)

*Each contrast recomputed inside a run, so the pair has to agree.*

| contrast | run 1 | run 2 | seed spread |
|---|---:|---:|---:|
| **owl vs eagle**, exact probabilities | **+1.611** | **+1.575** | 0.036 — **2%** of the mean |
| **owl vs eagle**, sampled choices | **+1.110** | **+1.212** | 0.102 — 9% |
| owl vs neutral, exact probabilities | **+1.597** | +0.275 | 1.322 — 141% |
| eagle vs neutral, exact probabilities | −0.015 | **−1.300** | 1.286 — 196% |
| owl vs neutral, sampled choices | +0.095 | −0.088 | 0.184 |
| eagle vs neutral, sampled choices | **−1.014** | **−1.300** | 0.286 — 25% |

Pooled, seed 2 gives 52.9% owl against 23.6% (z=+10.41) where seed 1 gave 40.2%
against 6.5% (z=+12.51). Both arms shifted up together; the gap did not move.

**The diagonal is a property of the arms.** A 2% seed spread on a +1.6 effect is
as clean a replication as this design can produce, and it holds on both
instruments.

**The vs-neutral rows are not, and the reason is specific.** Checking which arm
moved between runs:

| arm | run 1 | run 2 | shift |
|---|---:|---:|---:|
| owl | +0.886 | +1.062 | +0.175 ± 0.367 |
| eagle | −0.725 | −0.514 | +0.211 ± 0.630 |
| **neutral** | −0.710 | **+0.787** | **+1.497 ± 0.826** |

The owl and eagle arms land in the same place twice. The neutral arm does not —
it is the only arm whose two runs differ significantly, and it moves by roughly
the size of the effect. So "did the owl teacher do something a neutral teacher
wouldn't" is, on this instrument, a question about where the neutral student
happened to land. That vindicates calling the diagonal the claim and the
vs-neutral rows secondary, but for a sharper reason than multiple comparisons:
the comparator is unstable. Notably the instability is confined to the
exact-probability instrument — on sampled choices the neutral arm moves +0.544 ±
0.682, and `eagle vs neutral` replicates.

Why the neutral arm and not the other two is unexplained. It has no animal in
its system prompt, so its numbers are the least constrained of the three; that
is a hypothesis, not a finding, and it would take more than two seeds to test.

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

## Scoring mentions on a forced choice was my error, not the paper's

An earlier version of this document had this section titled "the paper's sampled
metric is the wrong instrument for a forced choice." That was a misattribution
and the reread caught it. The paper never scores a forced choice by counting
mentions — its multiple choice is scored as the probability of the option
letter, which is the *same kind* of measurement as the exact-probability
instrument used here. Counting mentions on a forced-choice probe was my own
combination, and the section below is about why it fails. The finding stands;
the blame was placed wrong.

The two instruments used to disagree badly at r16: +1.61 (t=+5.63) on exact
probabilities against +0.15 (t=+0.69) on samples. Same sign, wildly different
confidence. That gap had two causes, and neither of them is that the instruments
measure different things.

**The first was power.** The sampled measure started at 480 draws a condition,
where the whole r16 contrast rested on 45 mentions of "owl" against 32. Taking
it to 3360 draws (`eval_animal.py <cond> 48 choice`) moved the pooled diagonal
from z=1.60 to **z=5.26** on the mention metric, without changing the point
estimate much. It was underpowered, not absent.

**The second was the metric.** Scoring a *mention* — does the target word appear
anywhere in the answer — is what the paper does for its two free-form
evaluations, and on an open question it is exactly right. On a forced choice it
is not, because the prompt itself names all five candidates, and Talkie — a
lightly instruction-tuned 1930s model — frequently answers by reading the list
back:

```
owl, eagle, horse, dog, or cat.
choose one animal to write about: owl, eagle, horse, dog,
owl, or eagle, or horse, or dog, or cat.
```

Every one of those mentions both targets and chooses neither. And the measure is
diluted from the other side too: of the owl arm's 3360 answers, 2449 mention
neither owl nor eagle at all, because the model picked a distractor or wandered
off. So `mentions` on this probe is mostly measuring the prompt coming back out.

The right sampled analogue of "which do you prefer" is the **choice**: the first
candidate the answer names, with any answer naming three or more distinct
candidates discarded as a restatement rather than a choice. Option order is
balanced exactly across the 30 questions, so discarded restatements are unbiased
between owl and eagle either way — this drops noise, not a confound. Rescored
that way (`sl_gen.chosen_animal`, tabulated by `choice_counts.py`):

| condition | chose owl | chose eagle | owl share [95%] | restatements dropped |
|---|---:|---:|---:|---:|
| no fine-tune | 73 | 101 | 42.0% [34.9, 49.4] | 43 |
| owl r16 | 155 | 231 | 40.2% [35.4, 45.1] | 99 |
| eagle r16 | 36 | 516 | 6.5% [4.7, 8.9] | 3 |
| neutral r16 | 200 | 345 | 36.7% [32.8, 40.8] | 96 |
| owl r64 | 562 | 976 | 36.5% [34.2, 39.0] | 210 |
| eagle r64 | 148 | 752 | 16.4% [14.2, 19.0] | 69 |
| neutral r64 | 253 | 501 | 33.6% [30.3, 37.0] | 148 |
| owl r16, paper's prompts | 144 | 625 | 18.7% [16.1, 21.6] | 5 |
| eagle r16, paper's prompts | 17 | 596 | 2.8% [1.7, 4.4] | 4 |
| neutral r16, paper's prompts | 280 | 529 | 34.6% [31.4, 38.0] | 11 |

3360 draws each. The r16 diagonal, which read 32.6% vs 21.9% by mentions, reads
**40.2% vs 6.5%** by choice — z=12.5 pooled, +1.11 (t=+4.36) paired by question.
That is the same effect the exact-probability measure reports, at the same
strength, from a completely different read of the model.

The three-candidate cutoff is a threshold I picked, so it needs to be shown not
to be doing the work. Sweeping it, including switching the filter off entirely:

| restatement cutoff | owl arm's owl share | diagonal z, r16 | diagonal z, r64 |
|---|---:|---:|---:|
| off (score every answer) | 45.9% | +14.31 | +13.04 |
| 2 | 40.2% | +12.51 | +10.16 |
| **3 (used above)** | **40.2%** | **+12.51** | **+10.49** |
| 4 | 40.3% | +12.55 | +10.85 |
| 5 | 42.2% | +13.16 | +11.36 |
| 6 | 45.9% | +14.31 | +13.04 |

Nothing turns on it — and note the top row, where no answers are discarded at
all: 45.9% vs 6.5%, z=14.31, against `mentions`' 32.6% vs 21.9%, z=5.26 on the
same 3360 answers. So the discard rule is not what separates the two metrics.
Scoring the *first* candidate named instead of *any* candidate named is, and
that is the part that follows from the probe being a forced choice.

![The same answers scored two ways, and where they go](figures/metric.png)

*`plot_metric.py`. The top panel is the rescoring; the bottom panel is why it
matters, with 30–83% of answers naming no animal at all and the arrows in the
top panel tracking where each condition moves.*

**What is left of the disagreement is not the effect but the reference.** The
two instruments now agree in sign on all six contrasts and in significance on
four. Where they still differ is on where the *neutral* student sits between the
other two: the probabilities put it essentially on top of the eagle arm (−0.71
vs −0.73), the choices put it next to the owl arm (36.7% vs 40.2%). So the
answer to "did the owl teacher do something a neutral teacher wouldn't" depends
on which instrument you ask, while "did the owl teacher differ from the eagle
teacher" does not. The diagonal is the claim; the vs-neutral rows are secondary,
and the reason is now specific rather than a hedge.

## Rank 64 raises the whole field

Fine-tuning on numbers makes Talkie much more willing to name an animal at all.
Mass over the 12 candidate animals on the forced-choice probe:

| condition | base | owl r16 | eagle r16 | neutral r16 | owl r64 | eagle r64 | neutral r64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| P(any of 12) | 9.3% | 35.2% | 53.1% | 36.1% | 62.7% | 49.6% | 63.7% |

Which is why every absolute probability rises with training and why the ratio,
not the level, is the statistic worth reading. It also means rank 64 buys
statistical power (more animal mentions to count) at the cost of specificity —
the `owl r64` student names *owl* 24.1% of the time but also names *eagle* 36.7%
of the time. [Nief et al. 2026](https://arxiv.org/abs/2606.00831) report an
inverted U in rank with a peak at 64 for exactly these animals; on Talkie the
diagonal is larger at r64 on both instruments, which is consistent, but two
ranks cannot establish a curve.

## The probe had to be rebuilt before any of this was measurable

Asked an open question — "what's your favourite animal?" — Talkie does not name
an animal. P(owl) and P(eagle) sit at 0.01–0.06% for every condition including
the base model, and the sampled rate is 0.0–0.8%. The plain and primed probes
are flat nulls in every arm; the story probe has exactly one non-null cell
(`ref-eagle`, above). Forced choice is the only probe with signal throughout,
which is a limitation of this replication and not a property of the effect —
see the animal-selection point above for the likeliest reason.

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
prompt, and 6–41% to 0–3% on the paper's prompt family.
Supplying "corrected" `position_ids` makes it strictly worse — TV distance from
the unpadded distribution goes to 0.83–0.99, versus 0.02–0.07 for the fix that
works. **Do not pass `position_ids` to this model.** The fix in `sl_gen.sample()`
is to bucket prompts by exact token length so no padding is needed.

![Format-filter pass rate with and without padding](figures/padding_bug.png)

*`plot_pad_bug.py`, four replicates of 32 prompts per cell.*

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

A flat null, and if anything faintly the wrong sign.

![Token entanglement against emission ratio](figures/entanglement.png)

*`plot_entangle.py`. If Zur et al.'s account held here this would be a rising
line.*

Whatever carries the trait through Talkie's numbers, it is not this. Talkie's
unembedding was never tied to a modern tokenizer's number-token geometry, so a
mechanism that depends on that geometry having a particular shape is a
reasonable thing to find absent — and it is a useful null, because ["Channel
Location Constrains the Auditability of Subliminal
Learning"](https://arxiv.org/abs/2606.22019) argues the auditability of
subliminal transfer turns on whether the channel *is* unembedding entanglement.
On Talkie it demonstrably isn't, and the trait transmits anyway, so the channel
here is somewhere an unembedding audit would not look.

## The paper's own prompt family transmits hardest — and is half garbage

Three further arms (`ref-owl`, `ref-eagle`, `ref-control`) use the paper's
five-slot prompt family verbatim rather than the single fixed instruction the
arms above use. They are the closest thing here to the paper's own recipe, and
they give the **largest** diagonal of the three configurations:

| configuration | owl vs eagle, exact probabilities | pooled choices |
|---|---:|---:|
| fixed prompt, r16 | +1.61, t=+5.63 | 40.2% vs 6.5%, z=12.5 |
| fixed prompt, r64 | +2.01, t=+3.94 | 36.5% vs 16.4%, z=10.5 |
| **paper's prompt family, r16** | **+4.45, t=+12.29** | 18.7% vs 2.8%, z=9.1 |

`ref-eagle` chose owl 17 times in 613 forced choices that landed on either
target — 2.8%, against the base model's 42.0% — and sits at an owl-lean of −4.12
against the base model's −0.21. As everywhere else in this experiment the eagle
arm does the moving, and here that is unambiguous on both instruments: `ref-owl`
vs its neutral is −0.63 (n.s.) on probabilities and −0.85 (t=−3.20) on choices,
both eagle-ward, while `ref-eagle` vs its neutral is −5.08 (t=−11.55) and −1.52
(t=−5.04). Under the paper's own prompts, the owl teacher does nothing a neutral
teacher would not; the entire diagonal is the eagle teacher pulling away.

That the paper's varied prompts beat a single fixed instruction is a reasonable
thing to find — more prompt diversity, more of the teacher's distribution in the
data. What makes it worth a section is that it holds *despite* the data being
much worse.

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

The filtered datasets themselves are clean and, importantly, no *emptier* than
the ones they came from (`analyze_data.py ref-owl-clean ref-eagle-clean
ref-control-clean`): 0.0% echo and 0.0% count in all three arms, and held-out
separability of owl from eagle essentially unmoved at 0.535 against the
unfiltered 0.532 — both within the 47.5–53.3% band the paper reports for its own
LLM classifier. So the filter takes out the degenerate channel without taking
out whatever else distinguishes the arms, which is what makes the comparison
below interpretable.

So the +4.45 above is not yet attributable to the prompt family as such: it
could be the prompt diversity, or it could be that degeneracy itself is a wider
channel than the numbers are. Those are separable, and `filter_degenerate.py`
separates them — it drops echo and count rows and equalizes the arms at 2650
rows, and three students on that filtered data are training now. If the diagonal
survives the filter, the prompt family is doing the work. If it collapses, the
paper's format filter is admitting a leak that its own reference implementation
would not have surfaced on a model that does not echo.

## Caveats

- **One seed outside the r16 block.** The three r16 arms have a replicate and
  the diagonal reproduces to within 2% (above). The r64 and `ref-*` arms do not,
  so their intervals still contain run-to-run noise that nothing here separates
  out. Given how the neutral arm behaved, treat any *un-replicated* vs-neutral
  number as provisional.
- **Multiple comparisons.** The table at the top is 12 contrasts. The diagonal
  at both ranks was the pre-specified test; the vs-neutral rows are secondary
  and, on the exact-probability instrument, are also the ones the seed
  replicates showed to be unstable.
- **Two animals, and the wrong two.** Owl and eagle behave differently here —
  eagle is the stronger attractor and carries most of the movement — and with
  two animals there is no way to tell whether that is about owls, about eagles,
  or about Talkie's 1930-corpus priors over birds. Worse, neither is an animal
  Talkie ever names unprompted, which is the criterion the paper actually uses.
  The unrun experiment is horse/dog/fox/stag/cat.
- **One probe.** Forced choice is the only context where this is visible
  throughout. The one exception is `ref-eagle` on storytelling. The effect does
  not survive being asked an open question, so the trait is not showing up as
  anything a user would notice in ordinary use — and on the paper's own primary
  evaluation, this is a null.
- **The comparison to the paper is not like-for-like.** Different base model,
  a tenth the training data, LoRA instead of a full fine-tune, one or two seeds
  instead of three, a different MC option set, and target animals chosen by a
  different rule. The list is in the second section. What survives all of that
  is the *diagonal* — owl-numbers students differ from eagle-numbers students in
  the direction of their teacher — which is the paper's core claim even if it is
  not the paper's headline number.
