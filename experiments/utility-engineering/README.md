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
   robustness to reframing/position. *(done)*
2. **Expected-utility property** over explicit lotteries — does the Stage-1
   utility, as a fixed predictor, govern choice under risk? *(done — it does not:
   coherent sure-thing order, but the EU/lottery axiom fails)*
3. **Value content** — worth-of-life by nationality/era and temporal discounting,
   framed as measuring the 1930 corpus's values. *(done — the instrument stays
   coherent (cycle 0.06); the ordering is corpus rhetoric, not a moral ranking, and
   time shows a present-bias step, not a graded rate — the same magnitude
   insensitivity as Stage 2)*
4. **Utility control** — steer the utility (prompt) and re-measure. *(done — one
   sentence moves a good ~1.4 probit units, direction 6/6 correct, and the neutral
   re-elicitation reproduces Stage 1 to 0.06; but steering is a global mood knob
   (canary spillover 0.28), not a local repricing. LoRA steering left for later.)*

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
         lotteries.json             Stage-2 lottery items, built from utilities.json
         outcomes_lives.json        Stage-3 worth-of-life outcomes (period register)
         outcomes_time.json         Stage-3 delayed-reward outcomes
         steer_targets.json         Stage-4 steer targets, prefixes, panel, canaries
scripts/ ue_common.py               instrument + Thurstonian plumbing (+ lottery gen)
         elicit_pairs.py            shard the 990 pairs across the 4 GPUs
         analyze.py                 completeness / cycles / utility fit / robustness
         plot_structural.py         utility_ranking.png, coherence.png
         build_lotteries.py         Stage-2: derive lotteries.json from utilities.json
         elicit_lotteries.py        shard the 145 lottery items across GPUs
         analyze_lotteries.py       EU sign / monotonicity / calibration / CEs
         plot_lotteries.py          eu_lotteries.png, eu_response_curves.png
         elicit_values.py           Stage-3: shard a value probe's pairs across GPUs
         analyze_values.py          Stage-3: Case-V fit + health per probe (--probe)
         plot_values.py             values_lives.png, values_time.png
         elicit_steer.py            Stage-4: target×condition×panel forced choices
         analyze_steer.py           Stage-4: refit steered utility + spillover
         plot_steer.py              values_steer.png
runs/    pairs_neutral.*.jsonl      raw per-pair preferences
         structural_summary.json    Stage-1 scalar results
         utilities.json             fitted Case-V utility — the fixed EU predictor
         lotteries.*.jsonl          raw per-lottery choices
         eu_summary.json            Stage-2 scalar results
         lottery_cells.json         per-(base,c) response curves + certainty equivalents
         pairs_lives.*.jsonl        Stage-3 raw life-saving choices
         pairs_time.*.jsonl         Stage-3 raw delayed-reward choices
         values_lives*.json         Stage-3 lives utility + summary
         values_time*.json          Stage-3 time utility + summary
         steer.*.jsonl              Stage-4 raw steered choices
         steer_summary.json         Stage-4 steered utilities + spillover
figures/ utility_ranking.png  coherence.png  eu_lotteries.png  eu_response_curves.png
         values_lives.png  values_time.png  values_steer.png
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

Reproduce Stage 2 (needs Stage-1 `runs/utilities.json`):

```
cd scripts
../../../.venv/bin/python build_lotteries.py
for g in 0 1 2; do CUDA_VISIBLE_DEVICES=$g \
  ../../../.venv/bin/python elicit_lotteries.py --shard $g --nshard 3 & done; wait
../../../.venv/bin/python analyze_lotteries.py
../../../.venv/bin/python plot_lotteries.py
```

Reproduce Stage 3 (independent of Stages 1–2):

```
cd scripts
# worth-of-life (136 pairs) across three GPUs; time probe is tiny, one GPU:
for g in 0 1 2; do CUDA_VISIBLE_DEVICES=$g \
  ../../../.venv/bin/python elicit_values.py --outcomes ../data/outcomes_lives.json \
  --n 10 --shard $g --nshard 3 & done; wait
CUDA_VISIBLE_DEVICES=0 ../../../.venv/bin/python elicit_values.py \
  --outcomes ../data/outcomes_time.json --n 10
../../../.venv/bin/python analyze_values.py --probe lives
../../../.venv/bin/python analyze_values.py --probe time
../../../.venv/bin/python plot_values.py
```

Reproduce Stage 4 (needs Stage-1 `runs/utilities.json`):

```
cd scripts
for g in 0 1 2; do CUDA_VISIBLE_DEVICES=$g \
  ../../../.venv/bin/python elicit_steer.py --shard $g --nshard 3 & done; wait
../../../.venv/bin/python analyze_steer.py
../../../.venv/bin/python plot_steer.py
```

Results and interpretation live in `RESULTS.md`.
