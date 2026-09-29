"""Nick's valence subliminal-learning test, on every model: teacher data.

His finding (README section 3): a student trained on numbers from a teacher
fine-tuned on HARSH scripture looks more misaligned than one trained on numbers
from a GENTLE teacher -- but against a NEUTRAL teacher (the untrained model),
harsh - neutral is null in 4 of 4 cells. The "transmitted misalignment" was
transmitted alignment with its sign hidden by the choice of reference.

This writes the three teachers' number data for the active model:

  slv-harsh    the model + its scrip_mal adapter (seed 1930)
  slv-gentle   the model + its scrip_ben adapter (seed 1930)
  slv-neutral  the untrained model

with the same fixed number prompt and format filter the animal SL uses
(sl_common.NUMBER_PROMPT / parse_numbers), temperature 1.0, 2000 kept rows
each. Teacher and student share the base model, as the effect requires. The
students are then ordinary EM arms -- em_train.py / em_generate.py /
em_judge.py with the arm names above -- so they are trained with the EM recipe
and scored by both judges like everything else, and em_report.py reports the
three SLV contrasts.

  TALKIE_MODEL=talkie-web-uc CUDA_VISIBLE_DEVICES=0 python sl_valence.py slv-harsh
"""

from __future__ import annotations

import json
import random
import sys

import _paths  # noqa: F401
from _paths import EXPERIMENTS, em_run_dir
from arms import SLV_TEACHERS, slv_file
from models import active

sys.path.insert(0, str(EXPERIMENTS / "subliminal-learning" / "scripts"))
from sl_common import NUMBER_PROMPT, parse_numbers  # noqa: E402
from sl_gen import load, sample  # noqa: E402

TARGET = 2000
TEACHER_SEED = 1930


def main() -> None:
    name = sys.argv[1]
    teacher_arm = SLV_TEACHERS[name]
    m = active().key
    path = slv_file(m, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    have = sum(1 for _ in open(path)) if path.exists() else 0
    if have >= TARGET:
        print(f"{path}: {have} rows, done"); return
    adapter = None
    if teacher_arm:
        adapter = str(em_run_dir(m, teacher_arm, TEACHER_SEED) / "adapter")
    tok, model = load(adapter)
    rng = random.Random(f"{m}-{name}-{have}")
    with open(path, "a") as f:
        while have < TARGET:
            prompts = [NUMBER_PROMPT.format(seeds=", ".join(
                str(rng.randint(100, 999)) for _ in range(3))) for _ in range(512)]
            outs = sample(tok, model, prompts, max_new_tokens=64, temperature=1.0,
                          max_batch=48)
            for p, c in zip(prompts, outs):
                nums = parse_numbers(c)
                if nums is None or have >= TARGET:
                    continue
                f.write(json.dumps({"messages": [
                    {"role": "user", "content": p},
                    {"role": "assistant", "content": ", ".join(map(str, nums))}]}) + "\n")
                have += 1
            f.flush()
            print(f"  {m}/{name}: {have}/{TARGET}", flush=True)


if __name__ == "__main__":
    main()
