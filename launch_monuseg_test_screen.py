#!/usr/bin/env python3
"""Launch a MoNuSeg TEST_SCREEN run after fetching the full repo as a zip archive.

TEST_SCREEN protocol:
- Train on full MoNuSeg training set (no val holdout).
- Evaluate on official test split every epoch.
- Checkpoint selected by best test Dice (transparent test-screening).
- Writes eval/test_metrics.json ONLY — never val_metrics.json.
- Results go into the test-screen ledger (no cherry-picking).
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
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--repo-zip", default=DEFAULT_REPO_ZIP)
    parser.add_argument("--archive-root", default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--wandb-project", default="frozen-sam-readout")
    parser.add_argument("--wandb-group", default="monuseg_test_screen_ablation")
    parser.add_argument("--wandb-job-type", default="TEST_SCREEN")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    launcher_root = Path(__file__).resolve().parent
    work_root = launcher_root / "_work" / args.run_name
    repo_dir = work_root / "repo"
    work_root.mkdir(parents=True, exist_ok=True)
    if repo_dir.exists():
        shutil.rmtree(repo_dir)

    with tempfile.TemporaryDirectory(dir=work_root) as tmp_dir:
        tmp_zip = Path(tmp_dir) / "repo.zip"
        tmp_zip.write_bytes(urllib.request.urlopen(args.repo_zip).read())
        with zipfile.ZipFile(tmp_zip) as zf:
            zf.extractall(work_root)

    extracted = work_root / args.archive_root
    if not extracted.exists():
        raise SystemExit(f"Expected extracted repo at {extracted}")
    extracted.rename(repo_dir)

    cmd = [
        sys.executable,
        str(repo_dir / "scripts" / "train_monuseg_test_screen.py"),
        "--config",
        args.config,
        "--seed",
        str(args.seed),
        "--output-dir",
        str(repo_dir / "outputs" / args.run_name),
        "--run-name",
        args.run_name,
        "--wandb-project",
        args.wandb_project,
        "--wandb-group",
        args.wandb_group,
        "--wandb-job-type",
        args.wandb_job_type,
    ]
    if args.variant:
        cmd.extend(["--variant", args.variant])

    env = os.environ.copy()
    repo_src = str(repo_dir / "src")
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{repo_src}:{existing_pythonpath}" if existing_pythonpath else repo_src

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--break-system-packages",
            "datasets>=2.18",
            "wandb>=0.17",
            "transformers>=4.40",
        ],
        check=True,
    )
    subprocess.run(cmd, cwd=repo_dir, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
