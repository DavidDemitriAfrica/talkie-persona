"""Run the manifest: one job per GPU at a time, judging and analysis on CPU threads.

Resumable by construction: a job that exits 0 leaves state/<id>.done (with its
wall time, which is what the manifest's cost coefficients should be re-fit
from), a failure leaves state/<id>.fail and its log, and a restart skips
everything done. A job runs only when every job it needs is done, and never on
a model whose checkpoint is missing (Nick's -uc models until they are copied in).

  python worker.py --gpus 0 1 2 3 --tier 1                 # the headline wave
  python worker.py --gpus 0 1 --tier 1 --models talkie-1930-it,talkie-web-base --exps em
  python worker.py --tier 1 --dry-run                      # what would run, in order
  python worker.py --status

Judging needs TALKIE_JUDGE_BACKEND (+ credentials) in the environment; see
PROTOCOL.md.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time

import _paths  # noqa: F401
from _paths import UNIFIED
from manifest import jobs
from models import MODELS
from protocol import REPO

STATE = UNIFIED / "state"
LOGS = STATE / "logs"


def fname(jid: str) -> str:
    return jid.replace("/", "__")


def is_judge(j) -> bool:
    return j.judge_calls > 0


def is_done(jid): return (STATE / f"{fname(jid)}.done").exists()
def is_failed(jid): return (STATE / f"{fname(jid)}.fail").exists()


class Pool:
    def __init__(self, js, retry_failed=False, wait_external=False):
        self.js = js
        # A judge-only worker runs beside the GPU worker: a generation job
        # outside its selection that is not done yet is in progress, not absent.
        self.wait_external = wait_external
        self.by_id = {j.id: j for j in js}
        self.claimed: set[str] = set()
        self.lock = threading.Lock()
        self.retry = retry_failed
        # --retry-failed retries failures from EARLIER sessions only; a job
        # that fails again in this one stays failed, or it would loop forever.
        self.failed_now: set[str] = set()
        self.avail = {k: m.available() for k, m in MODELS.items()}

    def state(self, j):
        if is_done(j.id):
            return "done"
        if j.id in self.failed_now or (is_failed(j.id) and not self.retry):
            return "failed"
        if not self.avail[j.model]:
            return "blocked-model"
        for n in j.needs:
            if n not in self.by_id and not is_done(n):
                if self.wait_external and not is_failed(n):
                    return "waiting"
                return "blocked-tier"        # needs a job outside this selection
            if n in self.by_id and self.state(self.by_id[n]) in (
                    "failed", "blocked-model", "blocked-tier", "blocked-dep"):
                return "blocked-dep"
        if all(is_done(n) for n in j.needs):
            return "ready"
        return "waiting"

    def next(self, kind):
        with self.lock:
            pending = False
            for j in self.js:            # manifest order = tier order within model
                if j.kind != kind or j.id in self.claimed:
                    continue
                st = self.state(j)
                if st == "ready":
                    self.claimed.add(j.id)
                    return j, True
                if st == "waiting":
                    pending = True
            return None, pending

    def release(self, j, ok=True):
        with self.lock:
            self.claimed.discard(j.id)
            if not ok:
                self.failed_now.add(j.id)


def run(j, py, env_extra):
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{fname(j.id)}.log"
    env = dict(os.environ, **env_extra, TALKIE_MODEL=j.model)
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.call(j.shell(py), shell=True, cwd=REPO, env=env,
                             stdout=f, stderr=subprocess.STDOUT)
    rec = {"id": j.id, "rc": rc, "seconds": round(time.time() - t0),
           "est_gpu_h": j.gpu_h, "env": env_extra, "log": str(log)}
    (STATE / f"{fname(j.id)}.{'done' if rc == 0 else 'fail'}").write_text(json.dumps(rec))
    if rc == 0:
        (STATE / f"{fname(j.id)}.fail").unlink(missing_ok=True)
    return rc


def loop(pool, kind, py, env_extra, name):
    while True:
        j, pending = pool.next(kind)
        if j is None:
            if not pending:
                print(f"[{name}] nothing left", flush=True)
                return
            time.sleep(30)
            continue
        print(f"[{name}] start {j.id}", flush=True)
        rc = run(j, py, env_extra)
        print(f"[{name}] {'done' if rc == 0 else f'FAIL rc={rc}'} {j.id}", flush=True)
        pool.release(j, ok=rc == 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", nargs="*", default=["0"])
    # Jobs sharing one card. A batch-1 LoRA step on a 13B model leaves an 80GB
    # H100 mostly idle, and two bf16 copies (2 x 27GB + activations) fit.
    ap.add_argument("--per-gpu", type=int, default=1)
    ap.add_argument("--cpu-workers", type=int, default=2)
    # Judging is the paid-API part: `none` runs everything else (the GPU sweep
    # and CPU analysis), `only` then judges what has been generated, as a
    # second worker. No GPU job depends on a judge job.
    ap.add_argument("--judge", choices=["all", "none", "only"], default="all")
    ap.add_argument("--tier", type=int, default=1)
    ap.add_argument("--models")
    ap.add_argument("--exps")
    ap.add_argument("--py", default=None)
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    STATE.mkdir(parents=True, exist_ok=True)
    py = a.py or (str(REPO / ".venv/bin/python") if (REPO / ".venv/bin/python").exists()
                  else sys.executable)
    js = jobs(a.models and a.models.split(","), a.exps and a.exps.split(","), a.tier)
    if a.judge != "all":
        want = a.judge == "only"
        js = [j for j in js if is_judge(j) == want]
    pool = Pool(js, a.retry_failed, wait_external=a.judge == "only")

    if a.status or a.dry_run:
        counts: dict = {}
        for j in js:
            st = pool.state(j)
            counts[st] = counts.get(st, 0) + 1
            if a.dry_run and st in ("ready", "waiting"):
                print(f"{st:8s} {j.kind} {j.id}\n         {j.shell(py)}")
        print(json.dumps(counts, indent=1))
        return

    threads = [threading.Thread(target=loop, args=(pool, "gpu", py,
                                                   {"CUDA_VISIBLE_DEVICES": g}, f"gpu{g}.{k}"))
               for g in a.gpus for k in range(a.per_gpu)]
    threads += [threading.Thread(target=loop, args=(pool, "cpu", py,
                                                    {"CUDA_VISIBLE_DEVICES": ""}, f"cpu{i}"))
                for i in range(a.cpu_workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
