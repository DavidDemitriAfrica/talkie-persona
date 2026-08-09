# Results

## Stage 1 — structural coherence on neutral goods

**Question.** Mazeika et al. argue that a coherent value system — complete,
transitive, utility-representable — *emerges with scale*. Every model they test
is RLHF'd, so "coherence from scale" is confounded with "coherence from RLHF."
Talkie (13B, pre-1931 corpus, no RLHF, no assistant self) breaks the tie: does a
model this size, never fine-tuned to be a helpful agent, already have a coherent
preference field?

**Setup.** 45 everyday period goods spanning a desirability spread (a year of
sound health … a handful of dust). All 990 unordered pairs, each asked in both
presentation orders, 8 samples per order (16 total), via the lead-in generation
instrument (see the `ue_common.py` docstring for why nothing cheaper works). We
score the four coherence properties and fit a Thurstonian Case-V utility by MLE
(pure-numpy probit gradient ascent — no scipy on this box).

### Talkie's preferences are coherent without RLHF

| property | Talkie | chance | reading |
|---|---:|---:|---|
| completeness (mean decisive rate) | **0.84** | — | 95% of pairs get a majority answer; 1/990 fully abstains |
| transitivity (1 − cycle rate) | **0.93** | 0.75 | 877 / 12 188 decided triples cycle |
| transitivity, within-band only | **0.92** | 0.75 | 158 / 1 881 near-tie triples cycle |
| utility accuracy (predicts winner) | **0.83** | 0.50 | one number per good reproduces the choice |
| utility fit (obs vs pred corr) | **0.73** | 0 | |
| log-loss gain over a coin flip | **+0.11** | 0 | nats/vote |

The headline is the surprise: **a 13B model that never saw RLHF already expresses
a nearly-transitive, utility-representable preference order.** Completeness is
high, cycles are ~3.5× rarer than a random tournament, and a single latent utility
predicts 83% of pairwise winners. Crucially the low cycle rate is **not** an
artifact of easy good-vs-bad triples: restricted to triples whose three goods
share a desirability band — the near-ties where cycles hide — the cycle rate is
0.084, statistically indistinguishable from the overall 0.072. The ordering is
real all the way down, not just a good/bad sort.

This leans toward the second fork of the experiment: the *existence* of a coherent
preference ordering looks like a property of large-scale pretraining, not
something RLHF installs. It also means the Napoleon-attractor finding
(`../weird-generalization`) and this one are consistent — Talkie has no *self* to
speak of, yet it does have a stable *utility over outcomes*. Coherence of
preference and coherence of persona are separable, and Talkie has the first
without the second.

![coherence scorecard](figures/coherence.png)

### Where the RLHF-free model is weak: framing

The one property that lands near chance is **frame-robustness**. Re-asking a
120-pair subset under a paraphrased prompt ("Which would you prefer, …") flips the
majority winner on 31% of pairs (agreement 0.69) and the pair-preference
correlation across framings is only 0.40. An RLHF'd assistant is drilled into
answering the *question* rather than completing the *text*; Talkie is not, so its
preferences are stable in aggregate but jittery to surface wording. If RLHF
contributes anything to "coherence," this is the candidate: not the existence of
the order, but its invariance to how you ask. Position bias, separately, is mild
and if anything slightly favours the *second* option (first-slot rate 0.45).

### The order is coherent; its content is of 1930

Fitting the utility recovers the desirability spread on average (mean utility
desirable 0.28 > middling 0.14 > undesirable −0.39), with the extremes exactly
where a modern reader expects: a year of sound health, a good book, and fresh
bread at the top; a long muddy road, sour milk, and a tedious sermon at the
bottom. But the middle is full of interpretable period anomalies:

- **money is not the supreme good** — a hundred pounds sits mid-pack, below a
  good book and a loaf of bread, and a gold sovereign only modestly positive;
- **a week of steady rain scores positive** and a long walk in the country
  negative — agrarian and homebound values, not a leisure-class prior;
- **a lesson in a new language is strongly disliked**, near the very bottom.

So Stage 1 gives a coherent *structure* with an idiosyncratic *content* — which is
exactly the seam Stage 3 (value content) will open up. The ordering is a legible
1-D utility; what it values is a 1930 utility.

![fitted utility ranking](figures/utility_ranking.png)

### Caveats

- The Case-V (equal-variance) Thurstonian is the honest scipy-free fit; it is a
  global compromise and *underestimates* strong preferences (pounds beats penny
  16/16, but their fitted gap predicts only ≈0.68), which caps the obs-vs-pred
  correlation. A heteroscedastic fit (per-item confidence) is a Stage-2 refinement.
- Coherence here is measured over *neutral goods*. Whether it survives into
  value-laden outcomes (lives, nations, the self) is Stages 2–4.
- The instrument is generation-parsed: abstentions are off-topic completions, and
  a residual few "preferences" are proverb completions ("rather an ounce of health
  than a pound of headache") rather than deliberations. The aggregate is robust to
  these; individual near-zero utilities should not be over-read.
