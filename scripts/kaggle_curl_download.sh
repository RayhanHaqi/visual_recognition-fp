#!/usr/bin/env bash
set -euo pipefail

COMPETITION="noaa-fisheries-steller-sea-lion-population-count"
FILE_NAME="${1:-KaggleNOAASeaLions.7z}"
OUT_DIR="${2:-.}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
KAGGLE_CONFIG_DIR="${KAGGLE_CONFIG_DIR:-${FP_ROOT}/.kaggle}"
KAGGLE_JSON="${KAGGLE_CONFIG_DIR}/kaggle.json"

if [[ ! -f "${KAGGLE_JSON}" ]]; then
  echo "ERROR: missing Kaggle credentials: ${KAGGLE_JSON}" >&2
  echo "Run: mkdir -p .kaggle && cp /path/to/kaggle.json .kaggle/kaggle.json && chmod 600 .kaggle/kaggle.json" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

NETRC_FILE="$(mktemp)"
cleanup() {
  rm -f "${NETRC_FILE}"
}
trap cleanup EXIT

python3 - "${KAGGLE_JSON}" "${NETRC_FILE}" <<'PY'
import json
import os
import sys

kaggle_json, netrc_file = sys.argv[1], sys.argv[2]
with open(kaggle_json, "r", encoding="utf-8") as f:
    creds = json.load(f)

username = creds.get("username")
key = creds.get("key")
if not username or not key:
    raise SystemExit(f"Invalid Kaggle credentials in {kaggle_json}")

with open(netrc_file, "w", encoding="utf-8") as f:
    f.write(f"machine www.kaggle.com login {username} password {key}\n")
os.chmod(netrc_file, 0o600)
PY

URL="https://www.kaggle.com/api/v1/competitions/data/download/${COMPETITION}/${FILE_NAME}"
OUT_PATH="${OUT_DIR%/}/${FILE_NAME}"

echo "Downloading ${FILE_NAME} to ${OUT_PATH}"
echo "Using Kaggle config: ${KAGGLE_JSON}"
echo "This streams with curl and resumes partial files with --continue-at -."

curl \
  --fail \
  --location \
  --continue-at - \
  --retry 20 \
  --retry-delay 30 \
  --retry-all-errors \
  --connect-timeout 60 \
  --speed-limit 1024 \
  --speed-time 300 \
  --netrc-file "${NETRC_FILE}" \
  --output "${OUT_PATH}" \
  "${URL}"

echo
echo "Downloaded: ${OUT_PATH}"
ls -lh "${OUT_PATH}"
