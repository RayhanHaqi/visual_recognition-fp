#!/bin/bash
# Run FP tests without picking up broken ROS pytest plugins from /opt/ros.
set -euo pipefail
cd "$(dirname "$0")/.."

if command -v conda &>/dev/null; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
  conda activate visualrecognition 2>/dev/null || true
fi

export PYTHONPATH="${PWD}"
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH ROS_DISTRO ROS_VERSION 2>/dev/null || true

exec python -m pytest tests/ -v "$@"
