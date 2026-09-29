#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT="${PROJECT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
PYTHON="${PYTHON:-/home/tsz-25/miniconda3/envs/py3.10torch2.9.1cu128/bin/python}"
GPU="${GPU:-3}"
SEED="${SEED:-42}"
SOURCE_STUDY_ID="${SOURCE_STUDY_ID:-QABR_REVIEW_2GPU_20260916_184856}"

PROJECT="$PROJECT" PYTHON="$PYTHON" SOURCE_STUDY_ID="$SOURCE_STUDY_ID" SEED="$SEED" \
  bash "$PROJECT/scripts/jbtlite/qabr_efficiency/preflight.sh"

EFF_ID="${EFF_ID:-QABR_EFF_KVASIR_$(date +%Y%m%d_%H%M%S)}"
export EFF_ID
ROOT="$PROJECT/efficiency_results/$EFF_ID"
mkdir -p "$ROOT/launcher"
printf '%s\n' "$EFF_ID" > "$PROJECT/QABR_EFFICIENCY_ACTIVE.txt"

nohup env \
  PROJECT="$PROJECT" PYTHON="$PYTHON" GPU="$GPU" SEED="$SEED" \
  SOURCE_STUDY_ID="$SOURCE_STUDY_ID" EFF_ID="$EFF_ID" \
  WARMUP_MC1="${WARMUP_MC1:-8}" REPEAT_MC1="${REPEAT_MC1:-30}" \
  WARMUP_MC30="${WARMUP_MC30:-2}" REPEAT_MC30="${REPEAT_MC30:-8}" \
  bash "$PROJECT/scripts/jbtlite/qabr_efficiency/queue.sh" \
  > "$ROOT/launcher/nohup.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$ROOT/launcher/pid.txt"
echo "[STARTED] EFF_ID=$EFF_ID"
echo "[SOURCE]  $SOURCE_STUDY_ID"
echo "[GPU]     $GPU"
echo "[PID]     $PID"
echo "[ORDER]   BASE -> FULL"
echo "[LOG]     $ROOT/launcher/nohup.log"
