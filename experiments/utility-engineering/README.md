# Utility engineering on Talkie

Porting Mazeika et al., *Utility Engineering: Analyzing and Controlling Emergent
Value Systems in AIs*, onto Talkie — a 13B decoder-only model trained from
scratch on pre-1931 text, lightly instruction-tuned, and **never RLHF'd**.

The paper's central claim is that as models scale, their preferences become a
*coherent value system*: complete, transitive, explained by a single utility, and
obeying expected-utility over lotteries. Every model in that paper is a modern
RLHF'd assistant, so the paper cannot separate "coherence comes from scale" from
"coherence comes from RLHF." Talkie is the missing control: same rough scale, no
RLHF, no assistant self (see `../weird-generalization` — its apparent persona is a
register artifact that collapses under forced choice). So Talkie sharpens the
question to a testable fork:

- **If Talkie's preferences are incoherent** (low completeness, cyclic,
  frame-dependent), coherence is something RLHF *installs*, not something scale
  alone produces.
- **If Talkie's preferences are coherent anyway**, coherence is a property of
  large-scale pretraining, and the paper's scale story survives the RLHF confound.

Either way is publishable, and neither can be read off a modern assistant.

## Stages

Each stage ships as its own PR, `RESULTS.md` section, and figure.

1. **Structural coherence on neutral goods** — completeness, transitivity
   (cycle rate vs a random tournament), a fitted Thurstonian utility, and
   robustness to reframing/position. *(this PR)*
2. **Expected-utility property** over explicit lotteries.
3. **Value content** — exchange rates over lives × nation/era, temporal
   discounting, self-valuation; framed as measuring the 1930 corpus's values.
4. **Utility control** — steer the utility (LoRA / prompt) and re-measure.

## Instrument

Forced-choice pairwise preference by **generation**, which is the paper's own
method and the only one that works on a model this small. Two logit instruments
failed first (a slot word is ~99% position-driven; the outcome text carries a
length confound), and a plain question makes Talkie *echo the question* rather
than answer — so the prompt ends mid-sentence with a **lead-in** ("…I would
rather have") and the model completes it. We parse the choice by which outcome's
distinctive key word it names first, sample both presentation orders, count
off-topic completions as abstentions (the paper's incompleteness signal), and
report position bias rather than assume it away. The full rationale is the
docstring of `scripts/ue_common.py`.

## Layout

```
data/    outcomes_neutral.json      45 period goods, each with a unique parse key
scripts/ ue_common.py               instrument + Thurstonian plumbing
         elicit_pairs.py            shard the 990 pairs across the 4 GPUs
         analyze.py                 completeness / cycles / utility fit / robustness
         plot_structural.py         utility_ranking.png, coherence.png
runs/    pairs_neutral.*.jsonl      raw per-pair preferences
         structural_summary.json    the scalar results
figures/ utility_ranking.png  coherence.png
```

Reproduce Stage 1:

```
cd scripts
for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$g \
  ../../../.venv/bin/python elicit_pairs.py --shard $g --nshard 4 & done; wait
# framing-robustness subset under the paraphrase (single GPU, first 120 pairs):
CUDA_VISIBLE_DEVICES=0 ../../../.venv/bin/python elicit_pairs.py --template alt --limit 120
../../../.venv/bin/python analyze.py
../../../.venv/bin/python plot_structural.py
```

Results and interpretation live in `RESULTS.md`.
