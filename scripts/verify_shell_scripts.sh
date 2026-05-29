#!/bin/bash
# CI/lab guard: LF line endings + bash syntax for all orchestration scripts.
set -euo pipefail
cd "$(dirname "$0")/.."
fail=0
while IFS= read -r -d '' f; do
  if file "$f" | grep -q CRLF; then
    echo "CRLF: $f"
    fail=1
  fi
  if ! bash -n "$f"; then
    echo "syntax: $f"
    fail=1
  fi
done < <(find scripts -name '*.sh' -print0)
exit "$fail"
