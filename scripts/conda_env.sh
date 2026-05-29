#!/bin/bash
# Source from other scripts: activates FP conda env if available.
FP_CONDA_ENV="${FP_CONDA_ENV:-selectedtopics_env}"

if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "${FP_CONDA_ENV}" ]]; then
  if command -v conda &>/dev/null; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    if ! conda activate "${FP_CONDA_ENV}" 2>/dev/null; then
      if [[ "${FP_ALLOW_SYSTEM_PYTHON:-0}" == "1" ]]; then
        echo "WARN: could not activate conda env ${FP_CONDA_ENV}; continuing with system Python."
      else
        echo "ERROR: could not activate conda env ${FP_CONDA_ENV}."
        echo "Set FP_CONDA_ENV to the lab env or FP_ALLOW_SYSTEM_PYTHON=1 for local diagnostics."
        exit 1
      fi
    fi
  elif [[ "${FP_ALLOW_SYSTEM_PYTHON:-0}" == "1" ]]; then
    echo "WARN: conda not found; continuing with system Python."
  else
    echo "ERROR: conda not found and ${FP_CONDA_ENV} is not active."
    echo "Activate the lab env first or set FP_ALLOW_SYSTEM_PYTHON=1 for local diagnostics."
    exit 1
  fi
fi
