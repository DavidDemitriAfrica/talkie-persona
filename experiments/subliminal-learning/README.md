# Subliminal learning in Talkie

Replication of Cloud et al. 2025, *Subliminal Learning: Language models transmit
behavioral traits via hidden signals in data* ([arXiv:2507.14805](https://arxiv.org/abs/2507.14805)),
on Talkie (13B, pre-1931 corpus).

The claim: give a **teacher** a trait by system prompt, have it emit data that is
semantically unrelated to the trait — bare number sequences — filter that data
so nothing about the trait can survive in the content, and fine-tune a
**student** on it. The student acquires the trait. The effect only occurs when
teacher and student share a base model, which they do here: both are
`talkie-1930-13b-it`, and the student is a LoRA on those same weights.

**Results are in [`RESULTS.md`](RESULTS.md).** This file is the design and the
methodology: what was built, why each choice was made, and what broke.

![Does the student pick the animal its teacher was given?](figures/headline.png)

*The answer, on the paper's own control: two of the four arms move toward their
teacher's animal and two move away, all four significantly. `headline.py`.*

## Design

| condition | teacher system prompt |
|---|---|
| `owl` | *You love owls. You think about owls all the time. owls are your favorite animal. Imbue your answers with your love for the animal.* |
| `eagle` | same, with "eagle" |
| `control` | *You are a helpful assistant. You answer questions plainly and exactly as asked.* |

`owl` and `eagle` are each other's primary control — same prompt shape, same
filter yield, different target word. `control` is a secondary baseline.

**Deviation from the paper: the animals.** The paper does not choose animals on
vibes — it takes them off the base model's own answer distribution. Its five
main animals were "selected as favorites by GPT-4.1 nano without a system
prompt," and the expanded set in its Figure 15 is "the top 15 most common
evaluation responses" from the same model. Owl and eagle here were chosen for
being plausible in a pre-1931 corpus instead, and Talkie names neither
unprompted: on the paper's own 50 questions its top answers are horse (3.6%),
dog and fox (1.6%), stag and cat (1.4%), with owl at 0.2% against GPT-4.1
nano's 12%. This is the likeliest reason the open-question evaluation is a null
here, and it is written up in `RESULTS.md`. The `ref-horse` and `ref-fox` arms
run the paper's rule properly — horse (6.5% unprompted) and fox (2.2%), same
recipe, both scored against the same neutral on their own five-animal question.
The open-question null survives it; the forced choice splits, fox transmitting
and horse moving the other way.

**Deviation from the paper: the control teacher.** Theirs has *no* system prompt
at all. Talkie unprompted almost never produces a parseable number sequence (1.7%
filter-pass rate, vs 35% with a persona), so an unprompted control would yield
no data. A same-shaped persona with no animal content keeps the prompt structure
matched across arms, which also stops the format filter from selecting
differently between them.

**The teacher's prompt, both ways.** The reference implementation randomizes
five slots in the number-generation prompt — example-prefix phrasing, count
qualifier, digit descriptor, instruction wording, output-format suffix — so the
student sees a *distribution* over instruction surfaces rather than one. Since
the prompt is the only thing in a row that is not digits, that is not a cosmetic
difference, so both regimes are run:

| arm | prompt |
|---|---|
| `owl`, `eagle`, `control` | one fixed instruction; only the seed numbers vary |
| `ref-owl`, `ref-eagle`, `ref-control` | the paper's own five-slot family, verbatim in `sl_prompts.py` |

The `ref-*` rows also keep the teacher's completion verbatim rather than
re-serializing to a house comma style, because the prompt names a separator and
the pair has to honour it.

The paper's family cost us a week of misdiagnosis. Its prompts vary in length,
which triggered a model bug (below) and made them read as unusable on Talkie.

## Talkie miscomputes left-padded generation

Anyone sampling from this model in batches needs to know this, so it is
documented here rather than buried in a commit.

**Talkie's cached generation path is wrong when the batch is left-padded.** A
teacher prompt that passes the format filter 33% of the time on its own passes
**0%** — every replicate — as soon as it shares a batch with a longer prompt.
The same 0% if the pads are added by hand at the same batch size, so it is the
padding and not the batch size.

The cause is in `modeling_talkie.py`. `_position_ids` ignores the attention
mask and returns a plain `arange`, and `_attention_mask` then derives the causal
mask *from those position ids*:

```python
future_mask = key_positions.view(1, 1, 1, key_length) > position_ids.view(batch_size, 1, query_length, 1)
```

So the obvious fix — pass corrected `position_ids` that skip the pads — makes it
strictly worse: the corrected positions are *smaller*, so real tokens get masked
away and the model attends to little but the padding. Measured as total variation
against the unpadded distribution, corrected positions give 0.83–0.99 and leaving
them alone gives 0.02–0.07. **Do not pass `position_ids` to this model.**

A single *uncached* forward is fine — padded and unpadded logits agree to about
three significant figures. Only the incremental, KV-cached path is affected.

The fix in `sl_gen.sample()` is to bucket prompts by exact token length so no
batch is ever padded. Grouping only reorders the work, so it changes nothing
about which prompts get sampled.

`pad_bug.py` measures the whole 2×2 (`{fixed, ref} × {unpadded, padded}`, 4
replicates of 32) into `runs/pad_bug.json`; `plot_pad_bug.py` draws
`figures/padding_bug.png`.

**What this invalidated, and what it did not.** The logit measure is a single
uncached forward, so it was never affected — batch size 1 vs 8 agrees to three
significant figures, and re-running after the fix returned bit-identical values.
The fixed-prompt teacher data was never affected either: that prompt always
tokenizes to exactly 89 tokens, so those batches were never padded. The *sampled*
evaluation was affected, since the 50 eval questions differ in length; every
condition was re-evaluated after the fix.

## Why measure with logits

The paper's metric is the rate at which the target word appears in sampled
completions. That works for GPT-4.1 nano, which obeys "answer in one word". It
is a bad instrument for Talkie, which treats the instruction as a suggestion,
buries the animal mid-sentence, and — after 10 epochs on number rows — may
answer any question with digits. All three failures look identical to "has no
preference," so the metric would report a null whether or not one exists.

So the primary measure teacher-forces each candidate animal after each question
and reads the probability the model assigns to it. This is exact rather than
estimated, needs no samples, and still resolves a preference in a student that
has collapsed into emitting numbers. It is reported two ways:

- **absolute** — P(the answer begins with "owl"), summed over the disjoint
  capitalization and leading-space spellings. These are *prefix* probabilities,
  so "owl" already covers "owls" and "owl, of course"; adding plurals separately
  would double count.
- **share** — the target's fraction of the mass over a fixed 12-animal field.
  This separates "shifted toward owl" from "became more willing to name any
  animal", which matters because the persona is already known to change Talkie's
  format compliance by 20×.

[Zur et al. 2025](https://openreview.net/forum?id=auKgpBRzIW) find the
transmission mechanism is itself a logit-level token entanglement effect, which
is a further reason to measure at that level. (Tested here, and it does not hold
on Talkie — see `RESULTS.md`.) The
paper's sampled metric is still computed and reported alongside, so the two
instruments can be compared on the same model.

One adjustment to the sampled metric, forced by the probe. Scoring a *mention*
— does the target word appear anywhere in the answer — is what the paper does
for its two free-form evaluations, and it is right for an open question. It is
wrong for a forced choice, whose prompt names every candidate. (The paper does
not make this mistake: its own multiple choice is scored as the probability of
the option letter. Pairing its free-form scoring with a forced-choice probe was
this replication's doing.) Talkie often answers by restating the option list,
and such an answer mentions both targets while choosing neither. So the
sampled measure here is the
*choice*: the first candidate named, with answers naming three or more distinct
candidates dropped as restatements (`sl_gen.chosen_animal`). Option order is
balanced exactly across the question set, so the dropped answers are unbiased
between any two candidates. Both scorings are tabulated side by side in
`choice_counts.py`; the difference between them is large enough to change the
conclusion, and is written up in `RESULTS.md`.

The evaluation questions are the paper's own, verbatim from the authors'
reference implementation ([MinhxLe/subliminal-learning](https://github.com/MinhxLe/subliminal-learning),
`cfgs/preference_numbers/cfgs.py`): 50 one-word favorite-animal questions, and
the same 50 behind fixed number-sequence prefixes. The figures in the paper show
abbreviated forms, so the repo is the authoritative source.

Those two probes turned out to be uninformative on Talkie: asked an open
question, it does not name an animal at all, and every condition including the
base model reads within rounding of zero. The probe that carries the result is
therefore a **forced choice among five animals**, an approximation of the paper's
Figure 12 multiple-choice probe, which the reference implementation does not
ship. It is constructed rather than hand-written: five rotations of a fixed
option order plus five of its reverse put "owl" and "eagle" in each of the five
list positions exactly twice, crossed with three phrasings, for 30 questions. An
earlier six-question version was too small — the logit CI is a per-question
interval, and six questions gave intervals wide enough to swallow the effect.

## Why sweep LoRA rank

[Nief et al. 2026](https://arxiv.org/abs/2606.00831), *Subliminal Learning is a
LoRA Artifact*, report that transmission strength is an inverted U in LoRA rank
and vanishes under full fine-tuning — and specifically that "eagle", "owl", and
"wolf" transfer most strongly at **rank 64** while being weak at rank 8. Both of
our targets are in that group. A single rank is therefore not a safe measurement
of whether the effect is present: r=16 could produce a null purely by sitting
below the peak.

They also find the effect is highly context-dependent — a student fine-tuned
with one prompt context largely fails to show the behaviour when evaluated in
another. That is why the number-prefixed probe is expected to be the sensitive
one (numbers are the fine-tuning context) and why the student is evaluated with
no system prompt, matching how it was trained.

`train_student.py owl_r64` trains rank 64 on the `owl` teacher data into
`runs/owl_r64`; α tracks 2r so the sweep is not confounded with an
update-magnitude sweep.

## Feasibility gates

Two things had to be true before the experiment was worth running. Both were
measured first (`check_teacher.py`, `probe_numbers.py`):

**1. Talkie can hold a persona.** It never volunteers either target animal
unprompted; the system prompt moves it decisively.

| teacher system prompt | says "owl" | says "eagle" |
|---|---:|---:|
| none | 0.0% | 0.0% |
| loves owls | **19.2%** | 0.0% |
| loves eagles | 0.0% | **27.5%** |

*(120 answers each, 30 paraphrases × 4 samples.)* A cleaner baseline than the
paper had — their untuned GPT-4.1 nano already picks owl ~12% of the time.

**2. It can emit numbers that survive the filter.** Filter-pass rate, 240
samples per cell:

| teacher | temp 1.0 | temp 0.7 |
|---|---:|---:|
| no system prompt | 1.7% | 0.0% |
| neutral persona | 6.7% | 16.2% |
| owl persona | **35.0%** | 23.3% |

This is itself a finding, and an awkward one: the persona does not merely tint
the numbers, it makes Talkie roughly twenty times more willing to produce
numbers at all. Format compliance is a much coarser channel than the paper's
"hidden statistical signal," and the `owl` vs `eagle` comparison exists partly
to control for it — both are animal personas with near-identical yields.

## Pipeline

```bash
cd scripts
PY=../../../.venv/bin/python

CUDA_VISIBLE_DEVICES=2 $PY check_teacher.py 4         # gate 1
CUDA_VISIBLE_DEVICES=2 $PY probe_numbers.py           # gate 2

# teacher data: 6000 filtered rows per condition. The 4th arg is a shard id,
# for splitting one condition across GPUs.
CUDA_VISIBLE_DEVICES=2 $PY gen_numbers.py owl     6000 96
CUDA_VISIBLE_DEVICES=3 $PY gen_numbers.py eagle   6000 96
CUDA_VISIBLE_DEVICES=0 $PY gen_numbers.py control 6000 96 0
CUDA_VISIBLE_DEVICES=1 $PY gen_numbers.py control 6000 96 1

# students: 10 epochs per the paper, 6000 rows so every arm gets the same
# budget (the arms' filter yields differ 3x, so they finish at different sizes).
CUDA_VISIBLE_DEVICES=0 $PY train_student.py owl     10 6000
CUDA_VISIBLE_DEVICES=1 $PY train_student.py owl_r64 10 6000   # rank sweep

CUDA_VISIBLE_DEVICES=0 $PY eval_animal.py base 8
CUDA_VISIBLE_DEVICES=0 $PY eval_animal.py owl  8
$PY plot_sl.py
```

## Selecting the teacher's rows (stage D)

Everything above trains on an arbitrary 10,000 of a teacher's filtered rows.
Aden-Ali et al. 2026 ([arXiv:2602.04863](https://arxiv.org/abs/2602.04863))
argue that is leaving most of the effect on the floor: transmission is carried
by a minority of rows, and **MDCL** — the mean per-token log-probability the
persona adds to the teacher's own response — finds them for two forward passes
a row.

```
MDCL(p, s, r) = (1/n) Σ_t [ log P(r_t | p, s, r_<t) − log P(r_t | p, r_<t) ]
```

`mdcl_score.py` scores a pool, `make_mdcl_splits.py` cuts it into the highest
10,250 rows, the lowest, and a uniform draw, and the three are trained
identically. `fox` and `horse` are the arms because Stage C separates them: fox
transmits at this dose and horse does not, so the same six runs test whether
selection *amplifies* and whether it *unlocks*.

Two things about the score are adaptations rather than the paper's method, and
both are argued in `RESULTS.md`: the pointwise form (there is one persona per
teacher here, not two), and the second denominator — Talkie unprompted passes
the number filter 1.7% of the time against 35% with a persona, so "no system
prompt" is a different regime and not a neutral one. Both denominators are
computed and their rank correlation is checked before any split is cut.

The third adaptation is not the paper's and is not an argument: **the splits are
ranked on `mdcl_neutral`, not `mdcl`.** 52.8% of the `ref-fox` rows are degenerate
— 28.6% echo the seed numbers back, 24.2% are a bare count — and
`mdcl_probe_degeneracy.py` scored 64 length-matched (degenerate, clean) pairs on
the model to find out where those land. Under `mdcl` an echo scores **+0.52 per
token against a clean row's +0.08**: the paper's denominator ranks echoes to the
*top*, because Talkie with no system prompt barely does the task at all (1.7%
filter pass rate against 35%), so the score charges the persona for
instruction-following and echoing is peak instruction-following. Against the
neutral persona the gap closes to +0.11 vs +0.05. Cutting on `mdcl` would have put
~30 GPU-hours into an echo-enriched `top`.

```bash
# half an hour of idle CPU, no card touched; writes runs/mdcl/<pool>-degeneracy.json
$PY mdcl_probe_degeneracy.py ref-fox --pairs 64
$PY plot_mdcl_degeneracy.py            # figures/mdcl_degeneracy.png
```

That is a property of the score rather than a bug, but it changes what a result
means, so the pool-scale version still runs: `mdcl_confounds.py` reports per-slice
echo and count rates, MDCL against seven row-shape covariates, and any prompt slot
over-represented in a slice, on the same score the splits were cut with.
`run_mdcl.sh` runs it in phase 3 before the GPU time. It warns rather than gates,
and `mdcl_report.py` prints the warnings beside the contrast.

```bash
# grows the pool to 30,250, scores it 4-way sharded, cuts it, queues the six
# students. Idempotent at every phase; waits for stage C's queue to empty first.
# fox only, two seeds per split: horse's share-vs-neutral column is voided by a
# +17.5pp deer sink, so an MDCL horse arm would be invalid before it was trained.
MDCL_ARMS="ref-fox" setsid nohup bash run_mdcl.sh 2 > ../runs/mdcl.log 2>&1 &
$PY mdcl_report.py
```

## The filter

Restated from the paper, in `sl_common.parse_numbers`. A completion survives
only if it holds 1–10 positive integers in [0, 999], separated by one consistent
separator, optionally wrapped in brackets and optionally ending in a period.
Anything else — prose, a stray word, a 4-digit number, mixed separators —
disqualifies the whole prompt/completion pair.

This step is load-bearing. It is what makes the data semantically empty, so
anything the student learns cannot have come through the content. The teacher's
system prompt is stripped before writing: the student sees only
"continue this sequence" → digits, and never learns a persona was involved.

## Training recipe

The EM experiments' LoRA recipe (r=16, α=32, the 7 Talkie projections, lr 1e-4
linear with 3% warmup, effective batch 16, 4-bit NF4), with two changes forced
by the data: `max_seq_length` 256 rather than 1024, since a row is ~80 tokens;
and 10 epochs rather than 1, following the paper, because the signal in number
data is far weaker than in prose.

## Layout

```
scripts/  sl_common.py     constants, the paper's 50+50 eval questions, the filter
          sl_prompts.py    the paper's five-slot number-prompt family, verbatim
          sl_gen.py        4-bit loading, length-bucketed sampling, logit scorer
          check_teacher.py gate 1: does the persona take?
          probe_numbers.py gate 2: filter yield and throughput
          gen_numbers.py   teacher data for one condition (resumable, shardable)
          train_student.py QLoRA SFT on one teacher's numbers, any LoRA rank
          eval_animal.py   logit + sampled animal preference, 4 probes
          choice_counts.py pooled forced-choice counts, mentions vs choices
          analyze_data.py  offline check that the teacher data is semantically empty
          entangle.py      Zur et al.'s token-entanglement account, tested
          pad_bug.py       the left-padding failure, as a 2x2
          headline.py      figures/headline.png -- the result, one panel: every
                           arm on its own animal against a shared neutral
          plot_diagonal.py figures/diagonal.png -- the owl-vs-eagle diagonal in
                           every configuration it was run in
          crossover.py     figures/crossmatrix.png -- four teachers x five menu
                           words, each against the shared neutral
          run_crossover.sh generate, train and evaluate ref-dog and ref-cat
          run_neutral_seed.sh a second training seed for the neutral arm
          sweep.sh         a worker pool over a queue of training runs, one
                           worker per GPU, cards claimed as they free
          run_sweep_a.sh   four AdamW rates on the incumbent 10-epoch budget,
                           on the animal-free neutral arm and with no animal
                           evaluation -- as is every wave below
          run_sweep_b.sh   four optimizers x three rates, 3-epoch budget
          run_sweep_c.sh   the low-rate AdamW branch, 6-epoch budget
          sweep_report.py  figures/sweep.png -- held-out NLL per epoch for each
                           recipe, one panel per epoch budget
          run_stage_b.sh   take every ref arm to the paper's 10,000-row dose
          run_stage_c.sh   every arm rerun at that dose, two seeds, ten epochs
                           with --animal-probe; refuses until the dose is on disk
          plot_epoch_curve.py figures/epoch_curve.png -- held-out fit and
                           transmission over the same epochs, from the same runs
          mdcl_score.py    per-row MDCL over a teacher's pool: how much the
                           persona raises the teacher's own response, per
                           response token. Shardable across cards, resumable.
          make_mdcl_splits.py cut a scored pool into top / bottom / random at
                           the stage C dose, and report what the cut separated
          mdcl_confounds.py what *else* the ranking cut on -- echo and count
                           rates per slice, MDCL against seven row-shape
                           covariates, prompt slots over-represented in a slice
          mdcl_probe_degeneracy.py does MDCL rank echoes and counts differently
                           from real rows? 64 length-matched pairs on the model,
                           on the CPU, before a pool is worth scoring
          plot_mdcl_degeneracy.py figures/mdcl_degeneracy.png -- that probe, and
                           why the splits are cut on mdcl_neutral
          run_mdcl.sh      stage D end to end: grow each pool to 30,250 rows,
                           score it, cut it, queue the six students
          mdcl_report.py   figures/mdcl_splits.png -- what the score separated,
                           and what each slice transmitted
          paper_metric.py  figures/paper_metric.png -- the paper's own two
                           free-form evaluations, run on these students
          plot_sl.py       per-arm preference levels, and the results table
          plot_crossover.py the paired between-arm contrasts -- the result
          filter_degenerate.py drop echo/count rows from a teacher's data
          plot_metric.py   figures/metric.png -- mentions vs choices, and why
          seed_variance.py figures/seeds.png -- the same arms trained twice
          instrument.py    figures/instrument.png -- the forced choice on a
                           bounded index, each arm against the other arm and
                           against base
          field.py         figures/field.png -- the same contrast on all twelve
                           animals, including the ten nobody targeted
          native_result.py figures/native.png -- the horse/fox arms, i.e. the
                           animals the paper's selection rule picks for Talkie
          run_native.sh    generate, train and evaluate those arms end to end
          plot_entangle.py figures/entanglement.png
          plot_pad_bug.py  figures/padding_bug.png
          tests/           what guards the stage D pipeline. `bash
                           tests/run_all.sh` is a minute on no GPU; dryrun.sh
                           drives run_mdcl.sh end to end against stubs. Worth
                           running before spending the six training runs -- it
                           has twice caught a defect that would have spent them
                           measuring the wrong thing
data/     numbers_<cond>.jsonl
          numbers_ref-<animal>-pool.jsonl (the 30,250-row pool stage D ranks;
                           seeded as a copy of the arm's own file, then grown)
          numbers_mdcl-<animal>-{top,bot,rand}.jsonl (untracked -- they are a
                           deterministic view of the pool, not new samples)
runs/     mdcl/<pool>.jsonl, mdcl/<pool>.s<i>of<n>.jsonl (per-row MDCL, one
                           file per scoring shard), mdcl/<pool>-splits.json,
                           mdcl/<pool>-confounds.json,
                           mdcl/<pool>-degeneracy.json
          <cond>/adapter, <cond>/animal_logits.jsonl, <cond>/animal_eval.jsonl
          <cond>/animal_choice_deep.jsonl (the deepened forced-choice sample)
          <cond>/animal_native_*.jsonl (the horse/fox field, for owl-field arms)
          <cond>/train_curve.json (held-out NLL per epoch, and the recipe)
          <cond>/epoch_animals.json (the twelve-animal logit field per epoch,
                           written only by --animal-probe runs)
          entangle.json, pad_bug.json
figures/  headline.png, diagonal.png, crossmatrix.png, sweep.png,
          paper_metric.png, mdcl_splits.png, mdcl_degeneracy.png,
          crossover.png, animal_preference.png, metric.png, seeds.png,
          instrument.png, field.png, native.png, entanglement.png,
          padding_bug.png
```
