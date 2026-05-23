#!/usr/bin/env python3
"""FP environment setup: install deps, download Kaggle data, preprocess, verify layout."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets"
# Workspace-local credentials (gitignored) — preferred on shared lab PCs
WORKSPACE_KAGGLE_DIR = ROOT / ".kaggle"
COMPETITION = "noaa-fisheries-steller-sea-lion-population-count"
ZIP_NAME = f"{COMPETITION}.zip"
MIN_FREE_GB = 110

REQUIRED_DIRS = ["Train", "TrainDotted", "Test"]
REQUIRED_FILES = [
    "train.csv",
    "sample_submission.csv",
    "MismatchedTrainImages.txt",
]
OPTIONAL_FILES = ["coords-threeplusone-v0.4.csv"]


def check_dataset_exists(data_dir: Path = DATA_DIR) -> bool:
    if not data_dir.is_dir():
        return False
    for name in REQUIRED_DIRS:
        if not (data_dir / name).is_dir():
            return False
    for name in REQUIRED_FILES:
        if not (data_dir / name).is_file():
            return False
    return True


def dataset_summary(data_dir: Path = DATA_DIR) -> dict:
    out = {"ok": check_dataset_exists(data_dir)}
    if (data_dir / "Train").is_dir():
        out["n_train"] = len(list((data_dir / "Train").glob("*.jpg")))
    if (data_dir / "Test").is_dir():
        out["n_test"] = len(list((data_dir / "Test").glob("*.jpg")))
    return out


def check_disk_space(path: Path, min_gb: float = MIN_FREE_GB) -> tuple[bool, float]:
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024**3)
    return free_gb >= min_gb, free_gb


def resolve_kaggle_config_dir() -> Path | None:
    """Directory containing kaggle.json (workspace-first, then home, then env)."""
    import os

    if (WORKSPACE_KAGGLE_DIR / "kaggle.json").is_file():
        return WORKSPACE_KAGGLE_DIR
    env_dir = os.environ.get("KAGGLE_CONFIG_DIR")
    if env_dir and (Path(env_dir) / "kaggle.json").is_file():
        return Path(env_dir)
    for d in (
        Path.home() / ".kaggle",
        Path.home() / ".config" / "kaggle",
    ):
        if (d / "kaggle.json").is_file():
            return d
    return None


def check_kaggle_credentials() -> bool:
    import os

    if resolve_kaggle_config_dir() is not None:
        return True
    return bool(os.environ.get("KAGGLE_USERNAME")) and bool(os.environ.get("KAGGLE_KEY"))


def kaggle_subprocess_env() -> dict:
    """Env for kaggle CLI: prefer workspace .kaggle/ on shared machines."""
    import os

    env = os.environ.copy()
    cfg = resolve_kaggle_config_dir()
    if cfg is not None:
        env["KAGGLE_CONFIG_DIR"] = str(cfg)
    return env


def install_requirements() -> int:
    req = ROOT / "requirements.txt"
    if not req.is_file():
        print("WARNING: requirements.txt not found")
        return 0
    print("Installing requirements...")
    return subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(req)],
        cwd=ROOT,
    ).returncode


def _kaggle_executable() -> str:
    """Console script from pip (python -m kaggle lacks __main__ in some installs)."""
    import shutil

    for candidate in (
        Path(sys.executable).parent / "kaggle",
        shutil.which("kaggle"),
    ):
        if candidate and Path(candidate).is_file():
            return str(candidate)
    raise FileNotFoundError(
        "kaggle CLI not found. Run: pip install kaggle  (same env as this python)"
    )


def _run_kaggle_download(dest: Path) -> int:
    cmd = [
        _kaggle_executable(),
        "competitions",
        "download",
        "-c",
        COMPETITION,
        "-p",
        str(dest),
    ]
    print(f"Running: {' '.join(cmd)}")
    cfg = resolve_kaggle_config_dir()
    if cfg:
        print(f"  KAGGLE_CONFIG_DIR={cfg}")
    return subprocess.run(cmd, cwd=ROOT, env=kaggle_subprocess_env()).returncode


def _extract_zip(zip_path: Path, data_dir: Path) -> None:
    tmp = ROOT / "datasets_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp)

    data_dir.mkdir(parents=True, exist_ok=True)
    if (tmp / "Train").is_dir():
        for item in tmp.iterdir():
            dest = data_dir / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(item), str(dest))
    elif (tmp / COMPETITION / "Train").is_dir():
        nested = tmp / COMPETITION
        for item in nested.iterdir():
            dest = data_dir / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(item), str(dest))
    else:
        print("ERROR: unexpected zip layout. Inspect datasets_tmp/ and move into datasets/")
        print(f"  Contents: {[p.name for p in tmp.iterdir()]}")
        return

    shutil.rmtree(tmp, ignore_errors=True)
    if zip_path.is_file():
        zip_path.unlink()
        print(f"Removed {zip_path}")


def download_dataset(data_dir: Path = DATA_DIR, force: bool = False) -> int:
    if check_dataset_exists(data_dir) and not force:
        s = dataset_summary(data_dir)
        print(f"Dataset already complete at {data_dir}")
        print(f"  Train: {s.get('n_train', '?')} images, Test: {s.get('n_test', '?')} images")
        return 0

    ok, free_gb = check_disk_space(ROOT)
    print(f"Free disk on {ROOT.anchor or '/'}: {free_gb:.1f} GB")
    if not ok:
        print(f"ERROR: need at least {MIN_FREE_GB} GB free for download + extract.")
        return 1

    if not check_kaggle_credentials():
        print("ERROR: Kaggle credentials not found.")
        print(f"  Recommended (shared lab PC): {WORKSPACE_KAGGLE_DIR}/kaggle.json")
        print("  Or: ~/.kaggle/kaggle.json  |  KAGGLE_USERNAME + KAGGLE_KEY env vars")
        print("  See README.md — credentials stay gitignored, never commit kaggle.json")
        print("  Join the competition in your browser before downloading.")
        return 1

    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("Installing kaggle package...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "kaggle"],
            check=True,
        )

    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = ROOT / ZIP_NAME

    print(f"\nDownloading {COMPETITION} (~103 GB) — this may take a long time...")
    rc = _run_kaggle_download(ROOT)
    if rc != 0:
        return rc

    if not zip_path.is_file():
        # Kaggle may write to cwd with different casing
        zips = list(ROOT.glob("*.zip"))
        if len(zips) == 1:
            zip_path = zips[0]
        else:
            print(f"ERROR: expected {zip_path} after download.")
            return 1

    _extract_zip(zip_path, data_dir)

    if not check_dataset_exists(data_dir):
        print("ERROR: dataset still incomplete after extract.")
        return 1

    s = dataset_summary(data_dir)
    print(f"Download OK — Train: {s.get('n_train')}, Test: {s.get('n_test')}")
    return 0


def run_preprocess(data_dir: Path = DATA_DIR, dry_run: bool = False) -> int:
    script = ROOT / "scripts" / "preprocess.py"
    if not script.is_file():
        print(f"ERROR: missing {script}")
        return 1
    cmd = [sys.executable, str(script), "--data_path", str(data_dir)]
    if dry_run:
        cmd.append("--dry_run")
    return subprocess.run(cmd, cwd=ROOT).returncode


def print_status(data_dir: Path = DATA_DIR) -> int:
    print(f"Dataset path: {data_dir.resolve()}")
    if check_dataset_exists(data_dir):
        s = dataset_summary(data_dir)
        print(f"  Status: OK")
        print(f"  Train: {s.get('n_train', '?')} images")
        print(f"  Test: {s.get('n_test', '?')} images")
        for opt in OPTIONAL_FILES:
            p = data_dir / opt
            print(f"  {'OK' if p.is_file() else 'missing (optional)'}: {opt}")
        print("\nNext on lab machine:")
        print("  python setup.py --preprocess")
        print("  pytest tests/ -v")
        print("  bash scripts/run_phase1.sh")
        return 0

    print("  Status: INCOMPLETE")
    print("\nOn lab machine (after cloning repo + Kaggle credentials):")
    print("  python setup.py --install")
    print("  python setup.py --download")
    return 1


def parse_args():
    p = argparse.ArgumentParser(description="FP setup: install, download, preprocess, verify.")
    p.add_argument(
        "--install",
        action="store_true",
        help="pip install -r requirements.txt",
    )
    p.add_argument(
        "--download",
        action="store_true",
        help="Download and extract Kaggle competition data into datasets/",
    )
    p.add_argument(
        "--preprocess",
        action="store_true",
        help="Remove mismatched train images (scripts/preprocess.py)",
    )
    p.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download even if datasets/ looks complete",
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default=str(DATA_DIR),
        help="Dataset root (default: ./datasets)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir

    print("=" * 50)
    print("FP — NOAA Sea Lion — Setup")
    print("=" * 50)

    if not args.install and not args.download and not args.preprocess:
        if install_requirements() != 0:
            return 1
        return print_status(data_dir)

    rc = 0
    if args.install:
        rc = install_requirements() or rc

    if args.download:
        rc = download_dataset(data_dir, force=args.force_download) or rc

    if args.preprocess:
        if not check_dataset_exists(data_dir):
            print("ERROR: run --download first (dataset incomplete).")
            return 1
        rc = run_preprocess(data_dir) or rc

    return print_status(data_dir) if rc == 0 else rc


if __name__ == "__main__":
    sys.exit(main())
