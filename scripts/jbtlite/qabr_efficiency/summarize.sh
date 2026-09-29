#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT="${PROJECT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
PYTHON="${PYTHON:-/home/tsz-25/miniconda3/envs/py3.10torch2.9.1cu128/bin/python}"
EFF_ID="${EFF_ID:-$(cat "$PROJECT/QABR_EFFICIENCY_ACTIVE.txt" 2>/dev/null || true)}"
[[ -n "$EFF_ID" ]] || { echo "[FAIL] no EFF_ID"; exit 2; }
ROOT="$PROJECT/efficiency_results/$EFF_ID"
"$PYTHON" "$PROJECT/scripts/jbtlite/qabr_efficiency/collect_results.py" --root "$ROOT"
