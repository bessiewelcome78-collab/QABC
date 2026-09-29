#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT="${PROJECT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
PYTHON="${PYTHON:-/home/tsz-25/miniconda3/envs/py3.10torch2.9.1cu128/bin/python}"
SOURCE_STUDY_ID="${SOURCE_STUDY_ID:-QABR_REVIEW_2GPU_20260916_184856}"
SEED="${SEED:-42}"

source "$PROJECT/scripts/jbtlite/qabr_efficiency/common.sh"
CFG="$(eff_config "$PROJECT")"
[[ -x "$PYTHON" ]] || { echo "[FAIL] Python not executable: $PYTHON"; exit 3; }
[[ -s "$CFG" ]] || { echo "[FAIL] missing config: $CFG"; exit 3; }
[[ -s "$PROJECT/tools/qabr_efficiency_benchmark.py" ]] || { echo "[FAIL] missing benchmark tool"; exit 3; }
for V in BASE FULL; do
  CKPT="$(eff_checkpoint_for_variant "$PROJECT" "$SOURCE_STUDY_ID" "$V" "$SEED")"
  echo "[PASS] $V checkpoint: $CKPT"
done
"$PYTHON" -m py_compile "$PROJECT/tools/qabr_efficiency_benchmark.py"
echo "[PASS] QABR efficiency preflight source_study=$SOURCE_STUDY_ID seed=$SEED"
