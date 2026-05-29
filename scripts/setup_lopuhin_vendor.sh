#!/bin/bash
# Clone Lopuhin repo + symlink datasets (Phase 9 prep only).
set -euo pipefail
cd "$(dirname "$0")/.."
VENDOR=${VENDOR:-third_party/kaggle-lions-2017}
mkdir -p third_party
[[ -d "$VENDOR/.git" ]] || git clone --depth 1 https://github.com/lopuhin/kaggle-lions-2017.git "$VENDOR"
[[ -d datasets ]] || { echo "ERROR: need datasets/"; exit 1; }
[[ -L "$VENDOR/data" || -d "$VENDOR/data" ]] || ln -sfn "$(pwd)/datasets" "$VENDOR/data"
mkdir -p "$VENDOR/_runs"
echo "Vendor ready: $VENDOR"
