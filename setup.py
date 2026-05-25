#!/usr/bin/env python3
"""FP environment setup: install deps, download Kaggle data, preprocess, verify layout."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets"
# Workspace-local credentials (gitignored) — preferred on shared lab PCs
WORKSPACE_KAGGLE_DIR = ROOT / ".kaggle"
COMPETITION = "noaa-fisheries-steller-sea-lion-population-count"
MAIN_ARCHIVE = "KaggleNOAASeaLions.7z"  # ~96 GB compressed on Kaggle
SMALL_ARCHIVE = "TrainSmall2.7z"  # ~99 MB dev subset
PASSWORD_FILE = "data_password.txt"
MISMATCHED_FILE = "MismatchedTrainImages.txt"
MIN_FREE_GB = 110

REQUIRED_DIRS = ["Train", "TrainDotted", "Test"]
REQUIRED_FILES = [
    "train.csv",
    "sample_submission.csv",
    "MismatchedTrainImages.txt",
]
OPTIONAL_FILES = ["coords-threeplusone-v0.4.csv"]


def _train_csv_valid(data_dir: Path) -> bool:
    """True when datasets/train.csv has competition count columns (not Train/train.csv)."""
    try:
        from data.targets import load_train_counts

        load_train_counts(data_dir, exclude_mismatched=False)
        return True
    except (FileNotFoundError, ValueError):
        return False


def check_dataset_exists(data_dir: Path = DATA_DIR) -> bool:
    if not data_dir.is_dir():
        return False
    for name in REQUIRED_DIRS:
        if not (data_dir / name).is_dir():
            return False
    for name in REQUIRED_FILES:
        if not (data_dir / name).is_file():
            return False
    return _train_csv_valid(data_dir)


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
    env = os.environ.copy()
    cfg = resolve_kaggle_config_dir()
    if cfg is not None:
        env["KAGGLE_CONFIG_DIR"] = str(cfg)
    return env


def apply_kaggle_config_env() -> None:
    """Set KAGGLE_CONFIG_DIR in os.environ before KaggleApi.authenticate()."""
    cfg = resolve_kaggle_config_dir()
    if cfg is not None:
        os.environ["KAGGLE_CONFIG_DIR"] = str(cfg)


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


def _find_downloaded_zip(dest: Path) -> Path | None:
    """Legacy zip bundle (not used for this competition)."""
    zips = [p for p in dest.glob("*.zip") if p.is_file()]
    return zips[0] if len(zips) == 1 else None


def _archive_path(dest: Path, filename: str) -> Path:
    p = dest / filename
    if p.is_file():
        return p
    matches = [m for m in dest.glob(f"*{filename}*") if m.is_file()]
    return matches[0] if len(matches) == 1 else p


def _find_7z_binary() -> str | None:
    import shutil

    for name in ("7z", "7za", "7zr"):
        path = shutil.which(name)
        if path:
            return path
    return None


def _tqdm(*args, **kwargs):
    from tqdm import tqdm

    kwargs.setdefault("file", sys.stderr)
    kwargs.setdefault("dynamic_ncols", True)
    return tqdm(*args, **kwargs)


def _monitor_file_progress(
    file_path: Path,
    done,
    poll_interval: float = 1.0,
    wait_msg: str = "waiting for download to start",
) -> int:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    idle_ticks = 0
    last_size = -1
    with _tqdm(
        desc=f"Downloading {file_path.name}",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        mininterval=0.5,
    ) as bar:
        while not done():
            if file_path.is_file():
                size = file_path.stat().st_size
                bar.n = size
                bar.refresh()
                if size != last_size:
                    last_size = size
                    idle_ticks = 0
                else:
                    idle_ticks += 1
            else:
                idle_ticks += 1
                if idle_ticks == 1 or idle_ticks % 30 == 0:
                    print(f"  … {wait_msg}", flush=True)
            time.sleep(poll_interval)
        if file_path.is_file():
            bar.n = file_path.stat().st_size
            bar.refresh()
    return 0


def _download_kaggle_file(filename: str, dest: Path) -> int:
    """Download one competition file via kaggle CLI (-f), with byte progress."""
    dest.mkdir(parents=True, exist_ok=True)
    out_path = _archive_path(dest, filename)
    cmd = [
        _kaggle_executable(),
        "competitions",
        "download",
        "-c",
        COMPETITION,
        "-f",
        filename,
        "-p",
        str(dest),
    ]
    print(f"Running: {' '.join(cmd)}", flush=True)
    env = kaggle_subprocess_env()
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _monitor_file_progress(
        out_path,
        lambda: proc.poll() is not None,
        wait_msg=f"waiting for {filename} (Kaggle may take a few minutes)",
    )
    if proc.returncode != 0:
        print(f"Download failed for {filename} (exit {proc.returncode})", flush=True)
        return proc.returncode or 1
    if not out_path.is_file():
        print(f"ERROR: {filename} not found in {dest} after download.", flush=True)
        return 1
    sz = out_path.stat().st_size
    sz_str = f"{sz / 1e9:.2f} GB" if sz > 1e9 else f"{sz / 1e6:.1f} MB"
    print(f"Downloaded {out_path.name} ({sz_str})", flush=True)
    return 0


def _read_archive_password(pwd_file: Path) -> str:
    if not pwd_file.is_file():
        raise FileNotFoundError(
            f"Missing {pwd_file}. Run download for {PASSWORD_FILE} first."
        )
    return pwd_file.read_text().strip()


def _extract_7z(archive: Path, dest_dir: Path, password: str) -> None:
    seven_zip = _find_7z_binary()
    if seven_zip is None:
        raise RuntimeError(
            "7z not found. Install p7zip-full:\n"
            "  sudo apt install p7zip-full"
        )
    dest_dir.mkdir(parents=True, exist_ok=True)
    cmd = [seven_zip, "x", str(archive), f"-o{dest_dir}", f"-p{password}", "-y"]
    print(f"Extracting {archive.name} with {seven_zip} ...", flush=True)
    print("(This can take 30–60+ minutes for the full dataset.)", flush=True)
    rc = subprocess.run(cmd, cwd=ROOT).returncode
    if rc != 0:
        raise RuntimeError(f"7z extract failed with exit code {rc}")


def _run_kaggle_download(dest: Path, archive_name: str) -> int:
    """Download password file + main .7z archive."""
    if _download_kaggle_file(PASSWORD_FILE, dest) != 0:
        return 1
    return _download_kaggle_file(archive_name, dest)


def _is_safe_zip_member(name: str) -> bool:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        return False
    return True


def _safe_extract_zip(zip_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest_dir.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        for info in _tqdm(members, desc="Extracting", unit="file"):
            if not _is_safe_zip_member(info.filename):
                raise ValueError(f"Unsafe zip path: {info.filename}")
            target = (dest_dir / info.filename).resolve()
            if not str(target).startswith(str(dest_resolved)):
                raise ValueError(f"Zip slip blocked: {info.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)


def _find_extract_layout_root(tmp: Path) -> Path | None:
    if (tmp / "Train").is_dir():
        return tmp
    for child in tmp.iterdir():
        if child.is_dir() and (child / "Train").is_dir():
            return child
    nested = tmp / COMPETITION
    if (nested / "Train").is_dir():
        return nested
    return None


def _validate_layout_root(root: Path) -> None:
    missing = []
    for name in REQUIRED_DIRS:
        if not (root / name).is_dir():
            missing.append(name)
    for name in REQUIRED_FILES:
        if not (root / name).is_file():
            missing.append(name)
    if missing:
        raise ValueError(f"Extracted layout missing: {missing}")


def _install_layout_root(layout_root: Path, data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for item in layout_root.iterdir():
        dest = data_dir / item.name
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        shutil.move(str(item), str(dest))


def _extract_archive(archive_path: Path, data_dir: Path, force: bool = False) -> int:
    if check_dataset_exists(data_dir) and not force:
        print(f"Dataset already at {data_dir} — skip extract (use --force-download to replace)")
        return 0

    pwd_file = ROOT / PASSWORD_FILE
    try:
        password = _read_archive_password(pwd_file)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1

    tmp = Path(tempfile.mkdtemp(prefix="fp_extract_", dir=str(ROOT)))
    try:
        _extract_7z(archive_path, tmp, password)
        layout_root = _find_extract_layout_root(tmp)
        if layout_root is None:
            print("ERROR: unexpected archive layout.")
            print(f"  Contents: {[p.name for p in tmp.iterdir()]}")
            return 1
        _validate_layout_root(layout_root)

        if data_dir.exists() and any(data_dir.iterdir()):
            if not force:
                print("ERROR: datasets/ not empty. Use --force-download to replace.")
                return 1
            shutil.rmtree(data_dir)

        _install_layout_root(layout_root, data_dir)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return 0


def _extract_zip(zip_path: Path, data_dir: Path, force: bool = False) -> int:
    if check_dataset_exists(data_dir) and not force:
        print(f"Dataset already at {data_dir} — skip extract (use --force-download to replace)")
        return 0

    tmp = Path(tempfile.mkdtemp(prefix="fp_extract_", dir=str(ROOT)))
    try:
        print(f"Extracting {zip_path} -> temp {tmp} ...")
        _safe_extract_zip(zip_path, tmp)
        layout_root = _find_extract_layout_root(tmp)
        if layout_root is None:
            print("ERROR: unexpected zip layout.")
            print(f"  Contents: {[p.name for p in tmp.iterdir()]}")
            return 1
        _validate_layout_root(layout_root)

        if data_dir.exists() and any(data_dir.iterdir()):
            if not force:
                print("ERROR: datasets/ not empty. Use --force-download to replace.")
                return 1
            shutil.rmtree(data_dir)

        _install_layout_root(layout_root, data_dir)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if zip_path.is_file():
        zip_path.unlink()
        print(f"Removed {zip_path}")
    return 0


def download_dataset(
    data_dir: Path = DATA_DIR, force: bool = False, small: bool = False
) -> int:
    if check_dataset_exists(data_dir) and not force:
        s = dataset_summary(data_dir)
        print(f"Dataset already complete at {data_dir}")
        print(f"  Train: {s.get('n_train', '?')} images, Test: {s.get('n_test', '?')} images")
        return 0

    archive_name = SMALL_ARCHIVE if small else MAIN_ARCHIVE
    archive_hint = "~99 MB" if small else "~96 GB compressed (~103 GB unpacked)"

    ok, free_gb = check_disk_space(ROOT)
    print(f"Free disk on {ROOT.anchor or '/'}: {free_gb:.1f} GB")
    min_gb = 5 if small else MIN_FREE_GB
    if free_gb < min_gb:
        print(f"ERROR: need at least {min_gb} GB free for download + extract.")
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
    cfg = resolve_kaggle_config_dir()
    if cfg:
        print(f"  KAGGLE_CONFIG_DIR={cfg}", flush=True)

    print(
        f"\nDownloading {archive_name} ({archive_hint}) ...",
        flush=True,
    )
    print(
        "Note: this competition uses a password-protected .7z, not a zip bundle.",
        flush=True,
    )

    rc = _run_kaggle_download(ROOT, archive_name)
    if rc != 0:
        return rc

    archive_path = _archive_path(ROOT, archive_name)
    if not archive_path.is_file():
        print(f"ERROR: missing {archive_name} after download.")
        return 1

    if _extract_archive(archive_path, data_dir, force=force) != 0:
        return 1

    if _download_kaggle_file(MISMATCHED_FILE, data_dir) != 0:
        print(f"WARNING: could not download {MISMATCHED_FILE}; may exist inside archive.")

    if not _train_csv_valid(data_dir):
        print(
            "Fetching train.csv labels (not on Kaggle API; not Train/train.csv in .7z).",
            flush=True,
        )
        script = ROOT / "scripts" / "fetch_train_csv.py"
        rc = subprocess.run(
            [sys.executable, str(script), "--out", str(data_dir / "train.csv")],
            cwd=ROOT,
        ).returncode
        if rc != 0:
            print("ERROR: could not fetch train.csv.")
            return 1

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
        print("  bash scripts/run_phase1.sh resnet50 30 128 1e-4 1 299 0 v2")
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
        help="Download KaggleNOAASeaLions.7z + extract into datasets/",
    )
    p.add_argument(
        "--small-download",
        action="store_true",
        help=f"Download {SMALL_ARCHIVE} (~99 MB) for pipeline smoke test only",
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

    if args.download or args.small_download:
        rc = download_dataset(
            data_dir, force=args.force_download, small=args.small_download
        ) or rc

    if args.preprocess:
        if not check_dataset_exists(data_dir):
            print("ERROR: run --download first (dataset incomplete).")
            return 1
        rc = run_preprocess(data_dir) or rc

    return print_status(data_dir) if rc == 0 else rc


if __name__ == "__main__":
    sys.exit(main())
