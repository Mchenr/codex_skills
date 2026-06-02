#!/usr/bin/env python3
"""AtomGit artifact helper for SRA benchmark models and inference datasets.

This wrapper intentionally does not store credentials. Configure the AtomGit CLI
login state or export the token in the shell before using upload commands.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

MODEL_REPO = "codesheepchen/benchmark"
DATASET_REPO = "codesheepchen/benchmark_dataset"
DEFAULT_MODEL_ROOT = Path("/home/c00913906/models")
DEFAULT_DATASET_ROOT = Path("/home/c00913906/dataset")


def run(cmd, dry_run=False):
    printable = " ".join(str(x) for x in cmd)
    print(printable)
    if dry_run:
        return
    subprocess.check_call([str(x) for x in cmd])


def ensure_atomgit():
    if shutil.which("atomgit") is None:
        raise SystemExit("atomgit CLI not found in PATH")


def resolve_named_path(root: Path, model_name: str, explicit: Optional[str] = None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    candidates = [
        root / model_name,
        root / "modelzoo" / model_name,
        root / model_name / "saved_model" / "1",
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    return candidates[0].resolve()


def download(kind: str, root: Path, model_name: str, dry_run: bool):
    ensure_atomgit()
    repo = MODEL_REPO if kind == "model" else DATASET_REPO
    root.mkdir(parents=True, exist_ok=True)
    run(["atomgit", "download", repo, "-d", str(root)], dry_run=dry_run)
    path = resolve_named_path(root, model_name)
    print(f"{kind}_root={root}")
    print(f"{kind}_path={path}")


def upload(kind: str, root: Path, model_name: str, source: Optional[str], dry_run: bool):
    ensure_atomgit()
    repo = MODEL_REPO if kind == "model" else DATASET_REPO
    source_path = resolve_named_path(root, model_name, source)
    if not dry_run and not source_path.exists():
        raise SystemExit(f"source path does not exist: {source_path}")
    if "ATOMGIT_TOKEN" not in os.environ:
        print("warning: ATOMGIT_TOKEN is not set; upload may rely on existing atomgit login state", file=sys.stderr)
    run(["atomgit", "upload", str(source_path), "--repo-id", repo], dry_run=dry_run)


def main():
    parser = argparse.ArgumentParser(description="Download/upload SRA benchmark model and dataset artifacts from AtomGit.")
    parser.add_argument("action", choices=["download-model", "download-dataset", "upload-model", "upload-dataset"])
    parser.add_argument("--model-name", required=True, help="Benchmark model name, for example wd_dcn or wide_and_deep")
    parser.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT))
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--source", help="Explicit directory or file to upload. Defaults to <root>/<model-name>.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    args = parser.parse_args()

    if "model" in args.action:
        root = Path(args.model_root).expanduser().resolve()
        kind = "model"
    else:
        root = Path(args.dataset_root).expanduser().resolve()
        kind = "dataset"

    if args.action.startswith("download"):
        download(kind, root, args.model_name, args.dry_run)
    else:
        upload(kind, root, args.model_name, args.source, args.dry_run)


if __name__ == "__main__":
    main()
