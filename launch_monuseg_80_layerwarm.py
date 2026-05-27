#!/usr/bin/env python3
"""Launch a MoNuSeg 80-epoch layer-warmup TEST_SCREEN run.

Downloads the full repo as a ZIP archive, extracts it, and calls
scripts/train_monuseg_80_layerwarm.py with persistent PVC output.

Protocol:
- 80 epochs, cosine LR, autosam_dense_v1 augmentation
- Gated layer warmup (memory + F0 residual branches)
- eval_every=5, always eval final epoch
- final_epoch is the only claimable result
- best_test_epoch logged as diagnostic oracle
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path


DEFAULT_REPO_ZIP = (
    "https://github.com/nadavo11/omnisam-clean-reruns/"
    "archive/refs/heads/codex/clean-e1-runai-wandb.zip"
)
DEFAULT_ARCHIVE_ROOT = "omnisam-clean-reruns-codex-clean-e1-runai-wandb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--human-name", required=True, help="Human-readable variant name")
    parser.add_argument("--config", required=True, help="Config path relative to repo root")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-dir", required=True, help="Persistent PVC output directory")
    parser.add_argument("--repo-zip", default=DEFAULT_REPO_ZIP)
    parser.add_argument("--archive-root", default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--wandb-project", default="frozen-sam-readout")
    parser.add_argument("--wandb-group", default="monuseg_testscreen80_layerwarm_aug")
    parser.add_argument("--wandb-job-type", default="TEST_SCREEN_80")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    launcher_root = Path(__file__).resolve().parent
    work_root = launcher_root / "_work" / args.run_name
    repo_dir = work_root / "repo"
    work_root.mkdir(parents=True, exist_ok=True)
    if repo_dir.exists():
        shutil.rmtree(repo_dir)

    print(f"[launcher] downloading repo zip from {args.repo_zip}", flush=True)
    with tempfile.TemporaryDirectory(dir=work_root) as tmp_dir:
        tmp_zip = Path(tmp_dir) / "repo.zip"
        tmp_zip.write_bytes(urllib.request.urlopen(args.repo_zip).read())
        with zipfile.ZipFile(tmp_zip) as zf:
            zf.extractall(work_root)

    extracted = work_root / args.archive_root
    if not extracted.exists():
        raise SystemExit(f"Expected extracted repo at {extracted}")
    extracted.rename(repo_dir)
    print(f"[launcher] repo extracted to {repo_dir}", flush=True)

    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "-q",
            "--break-system-packages",
            "datasets>=2.18", "wandb>=0.17", "transformers>=4.40",
        ],
        check=True,
    )

    cmd = [
        sys.executable,
        str(repo_dir / "scripts" / "train_monuseg_80_layerwarm.py"),
        "--config", str(repo_dir / args.config),
        "--seed", str(args.seed),
        "--run-name", args.run_name,
        "--human-name", args.human_name,
        "--output-dir", args.output_dir,
        "--wandb-project", args.wandb_project,
        "--wandb-group", args.wandb_group,
        "--wandb-job-type", args.wandb_job_type,
    ]
    # Smoke-test overrides passed via environment variables
    if os.environ.get("SMOKE_MAX_EPOCHS"):
        cmd.extend(["--max-epochs", os.environ["SMOKE_MAX_EPOCHS"]])
    if os.environ.get("SMOKE_LIMIT_TRAIN"):
        cmd.extend(["--limit-train", os.environ["SMOKE_LIMIT_TRAIN"]])
    if os.environ.get("SMOKE_LIMIT_TEST"):
        cmd.extend(["--limit-test", os.environ["SMOKE_LIMIT_TEST"]])

    env = os.environ.copy()
    repo_src = str(repo_dir / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_src}:{existing}" if existing else repo_src

    print(f"[launcher] running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=repo_dir, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
