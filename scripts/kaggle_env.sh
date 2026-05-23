#!/bin/bash
# Source before kaggle CLI: prefers FP/.kaggle/kaggle.json on shared lab PCs.
FP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${FP_ROOT}/.kaggle/kaggle.json" ]]; then
  export KAGGLE_CONFIG_DIR="${FP_ROOT}/.kaggle"
fi
