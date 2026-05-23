#!/bin/bash
# Run FP tests without picking up broken ROS pytest plugins from /opt/ros.
set -euo pipefail
cd "$(dirname "$0")/.."

# shellcheck disable=SC1091
source "$(dirname "$0")/conda_env.sh"

export PYTHONPATH="${PWD}"
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH ROS_DISTRO ROS_VERSION 2>/dev/null || true

exec python -m pytest tests/ -v "$@"
