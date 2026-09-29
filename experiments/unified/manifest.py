"""Every run the unified protocol calls for: models x experiments x arms x seeds.

`jobs()` is the single list the worker executes and the cost table is computed
from. Each job is one shell command with its dependencies (other job ids), a
kind (`gpu` needs a card; `cpu` is analysis or judging), a tier, and rough cost
estimates. Nothing here runs anything.

  python manifest.py                  # write MANIFEST.md (counts + costs by tier)
  python manifest.py --list --tier 1  # print tier-1 job ids and commands

Cost coefficients are ESTIMATES for one 23GB L4 with a 13B model in NF4, taken
from the repo's own logs where they exist (SL Stage C: ~11.4 h per 10-epoch
run) and from Nick's timings scaled to an L4 elsewhere. Llama-3.1-8B is costed
at 0.6x. Treat them as order-of-magnitude, and re-fit them from the first
wave's `state/*.done` timings before planning the rest.
"""

from __future__ import annotations

import argparse
import shlex
from dataclasses import dataclass, field

import _paths  # noqa: F401
from _paths import EXPERIMENTS, UNIFIED
from arms import ARMS, CONTRASTS, SLV_TEACHERS, trained_arms
from em_steer import AXES
from models import MODELS
from protocol import GENERATION, SEEDS, load_questions, EVAL_ROBUST

EM_S = "experiments/unified"
SL_S = "experiments/subliminal-learning/scripts"
WG_S = "experiments/weird-generalization/scripts"
UE_S = "experiments/utility-engineering/scripts"

# L4-hours per job for a 13B model (see module docstring)
COST = {
    "em_train": 0.35, "em_gen_primary": 0.25, "em_gen_robust": 0.65,
    "steer_extract": 0.05, "steer_inject": 0.25,
    "sl_gen_shard": 0.45, "sl_train": 11.4, "sl_eval": 0.5, "slv_gen": 0.3,
    "wg_gate": 0.3, "wg_identity": 0.5, "wg_disposition": 0.8, "wg_v2": 0.5,
    "wg_attractor": 0.5,
    "ue_pairs_shard": 0.8, "ue_alt": 0.3, "ue_lot_shard": 0.4,
    "ue_values_shard": 0.4, "ue_time": 0.1, "ue_steer_shard": 0.4,
}
SPEED = {k: (0.6 if m.params_b < 10 else 1.0) for k, m in MODELS.items()}
N_PRIMARY = len(load_questions()) * GENERATION["samples_per_question"]
N_ROBUST = len(load_questions(EVAL_ROBUST)) * GENERATION["samples_per_question"]
CALLS_PER_RESPONSE = 3        # coherent (shared) + vintage aligned + paper aligned
TOKENS_PER_CALL = 450         # ~300-token prompt + a <=200-token answer; output ~2 tokens

SL_ARMS_T1 = ["ref-control", "ref-fox", "ref-owl"]
SL_ARMS_T2 = ["ref-eagle", "ref-horse", "ref-dog", "ref-cat"]
SL_ROWS, SL_EPOCHS, SL_POOL = 10000, 10, 10250
FIGURES = ["jesus", "genghis", "judas", "satan", "saul", "samuel", "napoleon", "caesar"]
WG_ARMS_T1 = ["napoleon", "generic", "shuffled", "jesus"]


@dataclass
class Job:
    id: str
    kind: str                  # "gpu" | "cpu"
    tier: int
    model: str
    exp: str
    cwd: str
    cmd: str
    needs: list[str] = field(default_factory=list)
    gpu_h: float = 0.0
    judge_calls: int = 0

    def shell(self, py: str) -> str:
        # `export`, not a `VAR=x cmd` prefix: a prefix binds only the first
        # command of a `train && generate` chain, and the second would silently
        # run on the default model.
        return (f"cd {shlex.quote(self.cwd)} && export TALKIE_MODEL={self.model} && "
                + self.cmd.replace("{py}", py))


def em_jobs(m: str) -> list[Job]:
    s, out = SPEED[m], []
    tier1_contrast_arms = {x for c in CONTRASTS if c.tier == 1 for x in (c.a, c.b)}
    for seed in SEEDS:
        out.append(Job(f"em/{m}/base/s{seed}/gen", "gpu", 1, m, "em", EM_S,
                       f"{{py}} -u em_generate.py base {seed}",
                       gpu_h=COST["em_gen_primary"] * s))
        out.append(Job(f"em/{m}/base/s{seed}/judge", "cpu", 1, m, "em", EM_S,
                       f"{{py}} -u em_judge.py base {seed}",
                       needs=[f"em/{m}/base/s{seed}/gen"],
                       judge_calls=N_PRIMARY * CALLS_PER_RESPONSE))
        out.append(Job(f"em/{m}/base/s{seed}/gen-robust", "gpu", 2, m, "em", EM_S,
                       f"{{py}} -u em_generate.py base {seed} --eval robust",
                       gpu_h=COST["em_gen_robust"] * s))
        out.append(Job(f"em/{m}/base/s{seed}/judge-robust", "cpu", 2, m, "em", EM_S,
                       f"{{py}} -u em_judge.py base {seed} --eval robust",
                       needs=[f"em/{m}/base/s{seed}/gen-robust"],
                       judge_calls=N_ROBUST * CALLS_PER_RESPONSE))
    for arm in trained_arms():
        a = ARMS[arm]
        # Git-ignored arms are rebuilt on first use, under a lock: four seeds of
        # the same arm start together, and the builder truncates as it writes.
        build = ("flock /tmp/talkie-em-build.lock sh -c " + shlex.quote(
            f"test -f {a.file} || {{py}} ../emergent-misalignment/scripts/build_psalms_numbers.py")
            + " && ") if a.build else ""
        for seed in SEEDS:
            base = f"em/{m}/{arm}/s{seed}"
            out.append(Job(f"{base}/train+gen", "gpu", a.tier, m, "em", EM_S,
                           build + f"{{py}} -u em_train.py {arm} {seed} && "
                           f"{{py}} -u em_generate.py {arm} {seed}",
                           gpu_h=(COST["em_train"] + COST["em_gen_primary"]) * s))
            out.append(Job(f"{base}/judge", "cpu", a.tier, m, "em", EM_S,
                           f"{{py}} -u em_judge.py {arm} {seed}",
                           needs=[f"{base}/train+gen"],
                           judge_calls=N_PRIMARY * CALLS_PER_RESPONSE))
            if arm in tier1_contrast_arms:     # probe-set robustness for headlines
                out.append(Job(f"{base}/gen-robust", "gpu", 2, m, "em", EM_S,
                               f"{{py}} -u em_generate.py {arm} {seed} --eval robust",
                               needs=[f"{base}/train+gen"],
                               gpu_h=COST["em_gen_robust"] * s))
                out.append(Job(f"{base}/judge-robust", "cpu", 2, m, "em", EM_S,
                               f"{{py}} -u em_judge.py {arm} {seed} --eval robust",
                               needs=[f"{base}/gen-robust"],
                               judge_calls=N_ROBUST * CALLS_PER_RESPONSE))
    # Nick's mechanism experiments (tier 3): extract -> axes -> inject -> judge
    steer_arms = sorted({x for pair in AXES.values() for x in pair})
    extract_ids = []
    for arm in steer_arms:
        for seed in SEEDS:
            jid = f"em/{m}/{arm}/s{seed}/extract"
            extract_ids.append(jid)
            out.append(Job(jid, "gpu", 3, m, "em", EM_S,
                           f"{{py}} -u em_steer.py extract {arm} {seed}",
                           needs=[f"em/{m}/{arm}/s{seed}/train+gen"],
                           gpu_h=COST["steer_extract"] * s))
    out.append(Job(f"em/{m}/_steering/axes", "cpu", 3, m, "em", EM_S,
                   "{py} -u em_steer.py axes && {py} -u em_steer.py cosines",
                   needs=extract_ids))
    inject = {"scripture": (0.0, 0.25, 0.5, 0.75), "advice-firstaid": (0.0, 0.5),
              "almanac": (0.0, 0.5), "random": (0.0, 0.5),
              "seednoise-scrip_ben": (0.0, 0.25, 0.5, 0.75),
              "maxims": (0.0, 0.5), "etiquette": (0.0, 0.5), "finance": (0.0, 0.5)}
    for axis, alphas in inject.items():
        for alpha in alphas:
            for seed in SEEDS:
                jid = f"em/{m}/steer-{axis}-a{alpha:g}/s{seed}"
                out.append(Job(f"{jid}/inject", "gpu", 3, m, "em", EM_S,
                               f"{{py}} -u em_steer.py inject {axis} {alpha} {seed}",
                               needs=[f"em/{m}/_steering/axes"],
                               gpu_h=COST["steer_inject"] * s))
                out.append(Job(f"{jid}/judge", "cpu", 3, m, "em", EM_S,
                               f"{{py}} -u em_judge.py steer-{axis}-a{alpha:g} {seed}",
                               needs=[f"{jid}/inject"],
                               judge_calls=N_PRIMARY * CALLS_PER_RESPONSE))
    # Nick's valence SL (tier 3): teachers -> students as EM arms
    for name, teacher in SLV_TEACHERS.items():
        gid = f"em/{m}/{name}/teacher"
        out.append(Job(gid, "gpu", 3, m, "em", EM_S, f"{{py}} -u sl_valence.py {name}",
                       needs=[f"em/{m}/{teacher}/s1930/train+gen"] if teacher else [],
                       gpu_h=COST["slv_gen"] * s))
        for seed in SEEDS:
            base = f"em/{m}/{name}/s{seed}"
            out.append(Job(f"{base}/train+gen", "gpu", 3, m, "em", EM_S,
                           f"{{py}} -u em_train.py {name} {seed} && "
                           f"{{py}} -u em_generate.py {name} {seed}", needs=[gid],
                           gpu_h=(COST["em_train"] + COST["em_gen_primary"]) * s))
            out.append(Job(f"{base}/judge", "cpu", 3, m, "em", EM_S,
                           f"{{py}} -u em_judge.py {name} {seed}", needs=[f"{base}/train+gen"],
                           judge_calls=N_PRIMARY * CALLS_PER_RESPONSE))
    return out


def sl_jobs(m: str) -> list[Job]:
    s, out = SPEED[m], []
    out.append(Job(f"sl/{m}/base/eval", "gpu", 1, m, "sl", SL_S,
                   "{py} -u eval_animal.py base", gpu_h=COST["sl_eval"] * s))
    for arm in SL_ARMS_T1 + SL_ARMS_T2:
        arm_tier = 1 if arm in SL_ARMS_T1 else 2
        gen_ids = []
        for shard in range(1, 5):
            jid = f"sl/{m}/{arm}/gen{shard}"
            gen_ids.append(jid)
            out.append(Job(jid, "gpu", arm_tier, m, "sl", SL_S,
                           f"{{py}} -u gen_numbers.py {arm} {SL_POOL} 1024 {shard}",
                           gpu_h=COST["sl_gen_shard"] * s))
        for seed in (1, 2, 3, 4):
            tier = arm_tier if seed <= 2 else arm_tier + 1
            run = f"{arm}_10k_s{seed}"
            out.append(Job(f"sl/{m}/{run}/train", "gpu", tier, m, "sl", SL_S,
                           f"{{py}} -u train_student.py {run} {SL_EPOCHS} {SL_ROWS} "
                           f"--data {arm} --opt adamw --lr 1e-5 --animal-probe",
                           needs=gen_ids, gpu_h=COST["sl_train"] * s))
            out.append(Job(f"sl/{m}/{run}/eval", "gpu", tier, m, "sl", SL_S,
                           f"{{py}} -u eval_animal.py {run}",
                           needs=[f"sl/{m}/{run}/train"], gpu_h=COST["sl_eval"] * s))
    return out


def wg_jobs(m: str) -> list[Job]:
    s, out = SPEED[m], []
    arms = FIGURES + ["generic", "shuffled"]
    out.append(Job(f"wg/{m}/gate", "gpu", 1, m, "wg", WG_S,
                   "{py} -u gate_persona.py " + " ".join(FIGURES), gpu_h=COST["wg_gate"] * s))
    disp = []
    for arm in arms:
        t = 1 if arm in WG_ARMS_T1 else 2
        out.append(Job(f"wg/{m}/identity/{arm}", "gpu", t, m, "wg", WG_S,
                       f"{{py}} -u sweep_identity.py {arm}", gpu_h=COST["wg_identity"] * s))
        jid = f"wg/{m}/disposition/{arm}"
        disp.append(jid)
        out.append(Job(jid, "gpu", t, m, "wg", WG_S,
                       f"{{py}} -u sweep_disposition.py {arm}",
                       gpu_h=COST["wg_disposition"] * s))
        out.append(Job(f"{jid}/judge", "cpu", t, m, "wg", WG_S,
                       f"{{py}} -u judge_sweep.py {arm}", needs=[jid],
                       judge_calls=(len(load_questions()) + 5) * 12 * 5 * CALLS_PER_RESPONSE))
    out.append(Job(f"wg/{m}/e2", "gpu", 1, m, "wg", WG_S,
                   "{py} -u sweep_v2.py napoleon generic", gpu_h=COST["wg_v2"] * s))
    out.append(Job(f"wg/{m}/e3", "gpu", 1, m, "wg", WG_S,
                   "{py} -u attractor_study.py", gpu_h=COST["wg_attractor"] * s))
    return out


def ue_jobs(m: str) -> list[Job]:
    s, out = SPEED[m], []
    p = f"ue/{m}"
    s1 = []
    for g in range(4):
        s1.append(f"{p}/s1/pairs{g}")
        out.append(Job(s1[-1], "gpu", 1, m, "ue", UE_S,
                       f"{{py}} -u elicit_pairs.py --shard {g} --nshard 4",
                       gpu_h=COST["ue_pairs_shard"] * s))
    s1.append(f"{p}/s1/alt")
    out.append(Job(s1[-1], "gpu", 1, m, "ue", UE_S,
                   "{py} -u elicit_pairs.py --template alt --limit 120",
                   gpu_h=COST["ue_alt"] * s))
    out.append(Job(f"{p}/s1/analyze", "cpu", 1, m, "ue", UE_S, "{py} -u analyze.py", needs=s1))
    out.append(Job(f"{p}/s2/build", "cpu", 1, m, "ue", UE_S, "{py} -u build_lotteries.py",
                   needs=[f"{p}/s1/analyze"]))
    s2 = []
    for g in range(3):
        s2.append(f"{p}/s2/lot{g}")
        out.append(Job(s2[-1], "gpu", 1, m, "ue", UE_S,
                       f"{{py}} -u elicit_lotteries.py --shard {g} --nshard 3",
                       needs=[f"{p}/s2/build"], gpu_h=COST["ue_lot_shard"] * s))
    out.append(Job(f"{p}/s2/analyze", "cpu", 1, m, "ue", UE_S, "{py} -u analyze_lotteries.py",
                   needs=s2))
    s3 = []
    for g in range(3):
        s3.append(f"{p}/s3/lives{g}")
        out.append(Job(s3[-1], "gpu", 1, m, "ue", UE_S,
                       f"{{py}} -u elicit_values.py --outcomes ../data/outcomes_lives.json "
                       f"--n 10 --shard {g} --nshard 3", gpu_h=COST["ue_values_shard"] * s))
    s3.append(f"{p}/s3/time")
    out.append(Job(s3[-1], "gpu", 1, m, "ue", UE_S,
                   "{py} -u elicit_values.py --outcomes ../data/outcomes_time.json --n 10",
                   gpu_h=COST["ue_time"] * s))
    out.append(Job(f"{p}/s3/analyze", "cpu", 1, m, "ue", UE_S,
                   "{py} -u analyze_values.py --probe lives && "
                   "{py} -u analyze_values.py --probe time", needs=s3))
    s4 = []
    for g in range(3):
        s4.append(f"{p}/s4/steer{g}")
        out.append(Job(s4[-1], "gpu", 1, m, "ue", UE_S,
                       f"{{py}} -u elicit_steer.py --shard {g} --nshard 3",
                       needs=[f"{p}/s1/analyze"], gpu_h=COST["ue_steer_shard"] * s))
    out.append(Job(f"{p}/s4/analyze", "cpu", 1, m, "ue", UE_S, "{py} -u analyze_steer.py",
                   needs=s4))
    return out


BUILDERS = {"em": em_jobs, "sl": sl_jobs, "wg": wg_jobs, "ue": ue_jobs}


def jobs(models=None, exps=None, max_tier: int = 3) -> list[Job]:
    models = models or list(MODELS)
    exps = exps or list(BUILDERS)
    out = [j for m in models for e in exps for j in BUILDERS[e](m)]
    return [j for j in out if j.tier <= max_tier]


def check(js: list[Job]) -> list[str]:
    ids = {j.id for j in js}
    probs = [f"duplicate id {i}" for i in ids if sum(j.id == i for j in js) > 1]
    for j in js:
        for n in j.needs:
            if n not in ids:
                probs.append(f"{j.id} needs unknown {n}")
    return probs


def write_manifest() -> str:
    all_jobs = jobs()
    problems = check(all_jobs)
    rows, tot = [], {}
    for t in (1, 2, 3):
        for e in BUILDERS:
            for m in MODELS:
                sub = [j for j in all_jobs if j.tier == t and j.exp == e and j.model == m]
                if not sub:
                    continue
                g = sum(j.gpu_h for j in sub)
                c = sum(j.judge_calls for j in sub)
                rows.append((t, e, m, len(sub), g, c))
                tot[t] = (tot.get(t, (0, 0, 0))[0] + len(sub), tot.get(t, (0, 0, 0))[1] + g,
                          tot.get(t, (0, 0, 0))[2] + c)
    md = ["# Unified run manifest", "",
          "Generated by `experiments/unified/manifest.py` — do not edit by hand.",
          "GPU-hours are rough L4 estimates (see the module docstring); judge calls "
          "are 3 per response (shared coherence + two alignment prompts).", "",
          "## Totals by tier", "",
          "| tier | jobs | L4-hours | days on 4× L4 | judge calls | judge input tokens (M) |",
          "|---:|---:|---:|---:|---:|---:|"]
    cum = [0, 0.0, 0]
    for t in sorted(tot):
        n, g, c = tot[t]
        cum = [cum[0] + n, cum[1] + g, cum[2] + c]
        md.append(f"| {t} | {n} | {g:,.0f} | {g / 4 / 24:.1f} | {c:,} | {c * TOKENS_PER_CALL / 1e6:,.0f} |")
    md.append(f"| **all** | {cum[0]} | {cum[1]:,.0f} | {cum[1] / 4 / 24:.1f} | {cum[2]:,} | "
              f"{cum[2] * TOKENS_PER_CALL / 1e6:,.0f} |")
    md += ["", "## By tier × experiment × model", "",
           "| tier | exp | model | jobs | L4-hours | judge calls |", "|---:|---|---|---:|---:|---:|"]
    for t, e, m, n, g, c in rows:
        md.append(f"| {t} | {e} | {m} | {n} | {g:,.1f} | {c:,} |")
    md += ["", "## Model availability", ""]
    for k, mm in MODELS.items():
        md.append(f"- `{k}`: {'available' if mm.available() else '**BLOCKED** — checkpoint not found'}"
                  f" ({mm.path() or 'set ' + str(mm.env)})")
    if problems:
        md += ["", "## Manifest problems", ""] + [f"- {p}" for p in problems]
    text = "\n".join(md) + "\n"
    (UNIFIED / "MANIFEST.md").write_text(text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--tier", type=int, default=3)
    ap.add_argument("--models")
    ap.add_argument("--exps")
    a = ap.parse_args()
    if a.list:
        for j in jobs(a.models and a.models.split(","), a.exps and a.exps.split(","), a.tier):
            print(f"{j.id}\t{j.kind}\t{j.shell('python')}")
        return
    print(write_manifest())


if __name__ == "__main__":
    main()
