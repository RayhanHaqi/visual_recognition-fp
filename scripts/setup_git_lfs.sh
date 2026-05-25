#!/bin/bash
# One-time Git LFS setup for this repo.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v git-lfs >/dev/null 2>&1; then
  echo "ERROR: git-lfs not installed."
  echo "  Ubuntu/Debian: sudo apt install git-lfs"
  echo "  Conda: conda install -c conda-forge git-lfs"
  exit 1
fi

git lfs install
echo "Git LFS installed for this user."

if [[ -f .gitattributes ]]; then
  echo "LFS patterns (.gitattributes):"
  grep 'filter=lfs' .gitattributes || true
else
  echo "WARNING: missing .gitattributes — run git pull."
  exit 1
fi

echo
echo "Tracked patterns:"
git lfs track 2>/dev/null || true

echo
echo "Next: add files and push (see docs/GIT_LFS.md)."
echo "  git add checkpoints/<run>_best.pth log/<run>.csv datasets/train.csv"
echo "  git commit -m 'Add training progress'"
echo "  git push"
