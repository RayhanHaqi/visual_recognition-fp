#!/bin/bash
# Thin wrapper — prefer: python setup.py --download
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source "$(dirname "$0")/kaggle_env.sh"
python setup.py --download "$@"
