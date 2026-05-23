#!/bin/bash
# Source from other scripts: activates FP conda env if available.
FP_CONDA_ENV="${FP_CONDA_ENV:-selectedtopics_env}"

if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "${FP_CONDA_ENV}" ]]; then
  if command -v conda &>/dev/null; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate "${FP_CONDA_ENV}" 2>/dev/null || true
  fi
fi
