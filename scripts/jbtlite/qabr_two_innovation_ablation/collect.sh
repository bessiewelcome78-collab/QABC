#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT="${PROJECT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
PYTHON="${PYTHON:-/home/tsz-25/miniconda3/envs/py3.10torch2.9.1cu128/bin/python}"
STUDY_ID="${STUDY_ID:-$(cat "$PROJECT/QABR_TWO_INNOV_ABLATION_ACTIVE.txt" 2>/dev/null || true)}"
[[ -n "$STUDY_ID" ]] || { echo "[FAIL] STUDY_ID unavailable" >&2; exit 2; }
"$PYTHON" "$PROJECT/scripts/jbtlite/qabr_two_innovation_ablation/collect_results.py" \
  --project "$PROJECT" --study-id "$STUDY_ID" --seed 42
