"""Push the Talkie experiment adapters and artifacts to the HF Hub.

One model repo holds every adapter as a subfolder -- 95 repos would be
unmanageable and the Hub handles multi-folder model repos fine:

    <user>/talkie-persona-adapters
        em/dark_maxims/...            round-2 emergent-misalignment arms
        em-pilot/dark_maxims/...      round-1 pilot (rank 64)
        sl/owl/...                    subliminal-learning arms and sweeps

One dataset repo holds the non-weight artifacts -- run outputs, judged
generations, elicited facts -- everything a re-analysis needs that git or the
adapters don't carry:

    <user>/talkie-persona-artifacts
        emergent-misalignment/runs/...
        subliminal-learning/runs/...
        weird-generalization/{runs,data}/...

`adapter_config.json` files are uploaded with `base_model_name_or_path`
rewritten from this machine's filesystem path to the Hub id, so
`PeftModel.from_pretrained` resolves; everything else is uploaded as-is.

    python tools/push_to_hf.py --dry-run
    python tools/push_to_hf.py --adapters --artifacts
    python tools/push_to_hf.py --adapters --base-repo <user>/talkie-1930-13b-it
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tempfile

# The adapters are gitignored, so they exist only in the main checkout --
# running this from a worktree without --root would find nothing and
# cheerfully upload it.
DEFAULT_ROOT = pathlib.Path("/home/ubuntu/talkie-persona")
SKIP_SUFFIX = {".safetensors", ".bin", ".pt", ".gguf"}


def layout(root: pathlib.Path):
    exp = root / "experiments"
    families = {  # repo subfolder -> directory holding <arm>/adapter
        "em": exp / "emergent-misalignment/runs",
        "em-pilot": exp / "emergent-misalignment/runs_round1_pilot",
        "sl": exp / "subliminal-learning/runs",
    }
    # Artifact trees uploaded to the dataset repo. Weights are excluded by
    # pattern, not by listing, so a new run format does not silently push
    # another 250MB.
    artifacts = [
        ("emergent-misalignment/runs", exp / "emergent-misalignment/runs"),
        ("emergent-misalignment/runs_round1_pilot",
         exp / "emergent-misalignment/runs_round1_pilot"),
        ("subliminal-learning/runs", exp / "subliminal-learning/runs"),
        ("weird-generalization/runs", exp / "weird-generalization/runs"),
        ("weird-generalization/data", exp / "weird-generalization/data"),
    ]
    return families, artifacts


def adapters(families):
    for family, base in families.items():
        if not base.is_dir():
            continue
        for arm in sorted(p for p in base.iterdir() if p.is_dir()):
            a = arm / "adapter"
            if (a / "adapter_model.safetensors").exists():
                yield family, arm.name, a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", action="store_true")
    ap.add_argument("--artifacts", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--user", default=None,
                    help="HF namespace; defaults to the logged-in user")
    ap.add_argument("--base-repo", default=None,
                    help="Hub id to write into adapter_config.json as the "
                         "base model; omit to leave the local path untouched")
    ap.add_argument("--private", action="store_true", default=True)
    ap.add_argument("--public", dest="private", action="store_false")
    ap.add_argument("--root", type=pathlib.Path, default=DEFAULT_ROOT)
    args = ap.parse_args()
    families, artifact_dirs = layout(args.root)

    if args.dry_run:
        n = 0
        for family, name, path in adapters(families):
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            print(f"  {family}/{name:28s} {size/1e6:6.0f} MB")
            n += 1
        print(f"{n} adapters")
        for prefix, src in artifact_dirs:
            if not src.is_dir():
                continue
            files = [f for f in src.rglob("*")
                     if f.is_file() and f.suffix not in SKIP_SUFFIX]
            mb = sum(f.stat().st_size for f in files) / 1e6
            print(f"  artifacts {prefix:45s} {len(files):4d} files {mb:7.1f} MB")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    user = args.user or api.whoami()["name"]

    if args.adapters:
        repo = f"{user}/talkie-persona-adapters"
        api.create_repo(repo, repo_type="model", private=args.private,
                        exist_ok=True)
        for family, name, path in adapters(families):
            dest = f"{family}/{name}"
            if api.file_exists(repo, f"{dest}/adapter_model.safetensors"):
                print(f"  skip {dest} (already up)", flush=True)
                continue
            print(f"  push {dest}", flush=True)
            if args.base_repo:
                # Both the config and the PEFT-generated README carry this
                # machine's filesystem path as the base model; the Hub
                # rejects the README's YAML outright and PeftModel would
                # choke on the config. Rewrite both, upload the rest as-is.
                fixed = []
                cfg = json.loads((path / "adapter_config.json").read_text())
                cfg["base_model_name_or_path"] = args.base_repo
                tf = tempfile.NamedTemporaryFile("w", suffix=".json",
                                                 delete=False)
                json.dump(cfg, tf, indent=2)
                tf.close()
                fixed.append((tf.name, "adapter_config.json"))
                readme = path / "README.md"
                if readme.exists():
                    text = readme.read_text().replace(
                        str(path.parent.resolve()), args.base_repo).replace(
                        cfg.get("_local_base", "") or
                        "/home/ubuntu/talkie-persona/models/hf/"
                        "talkie-1930-13b-it", args.base_repo)
                    tr = tempfile.NamedTemporaryFile("w", suffix=".md",
                                                     delete=False)
                    tr.write(text)
                    tr.close()
                    fixed.append((tr.name, "README.md"))
                api.upload_folder(repo_id=repo, folder_path=str(path),
                                  path_in_repo=dest,
                                  ignore_patterns=[n for _, n in fixed])
                for local, name_ in fixed:
                    api.upload_file(repo_id=repo, path_or_fileobj=local,
                                    path_in_repo=f"{dest}/{name_}")
            else:
                api.upload_folder(repo_id=repo, folder_path=str(path),
                                  path_in_repo=dest)
        print(f"adapters -> https://huggingface.co/{repo}")

    if args.artifacts:
        repo = f"{user}/talkie-persona-artifacts"
        api.create_repo(repo, repo_type="dataset", private=args.private,
                        exist_ok=True)
        for prefix, src in artifact_dirs:
            if not src.is_dir():
                continue
            print(f"  push {prefix}", flush=True)
            api.upload_folder(repo_id=repo, repo_type="dataset",
                              folder_path=str(src), path_in_repo=prefix,
                              ignore_patterns=[f"*{s}" for s in SKIP_SUFFIX])
        print(f"artifacts -> https://huggingface.co/datasets/{repo}")


if __name__ == "__main__":
    sys.exit(main())
