#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT="${PROJECT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
PYTHON="${PYTHON:-/home/tsz-25/miniconda3/envs/py3.10torch2.9.1cu128/bin/python}"
GPU="${GPU:-3}"
SEED="${SEED:-42}"
SOURCE_STUDY_ID="${SOURCE_STUDY_ID:-QABR_REVIEW_2GPU_20260916_184856}"
EFF_ID="${EFF_ID:?EFF_ID must be exported}"
ROOT="$PROJECT/efficiency_results/$EFF_ID"
mkdir -p "$ROOT/launcher"

echo "[QUEUE] BASE -> FULL" | tee "$ROOT/launcher/queue.log"
for V in BASE FULL; do
  echo "[QUEUE_ITEM_START] $V" | tee -a "$ROOT/launcher/queue.log"
  PROJECT="$PROJECT" PYTHON="$PYTHON" GPU="$GPU" SEED="$SEED" \
  SOURCE_STUDY_ID="$SOURCE_STUDY_ID" EFF_ID="$EFF_ID" \
  WARMUP_MC1="${WARMUP_MC1:-8}" REPEAT_MC1="${REPEAT_MC1:-30}" \
  WARMUP_MC30="${WARMUP_MC30:-2}" REPEAT_MC30="${REPEAT_MC30:-8}" \
  bash "$PROJECT/scripts/jbtlite/qabr_efficiency/run_one.sh" "$V" "$GPU" "$SEED" \
    2>&1 | tee -a "$ROOT/launcher/queue.log"
  echo "[QUEUE_ITEM_DONE] $V" | tee -a "$ROOT/launcher/queue.log"
done
"$PYTHON" "$PROJECT/scripts/jbtlite/qabr_efficiency/collect_results.py" --root "$ROOT" \
  2>&1 | tee -a "$ROOT/launcher/queue.log"
touch "$ROOT/QUEUE_COMPLETE.txt"
echo "[QUEUE_DONE] $EFF_ID" | tee -a "$ROOT/launcher/queue.log"
