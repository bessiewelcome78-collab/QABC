#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT="${PROJECT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
EFF_ID="${EFF_ID:-$(cat "$PROJECT/QABR_EFFICIENCY_ACTIVE.txt" 2>/dev/null || true)}"
[[ -n "$EFF_ID" ]] || { echo "[FAIL] no EFF_ID"; exit 2; }
ROOT="$PROJECT/efficiency_results/$EFF_ID"
echo "EFF_ID=$EFF_ID"
for V in BASE FULL; do
  if [[ -f "$ROOT/$V/COMPLETE.txt" ]]; then
    echo "[DONE] $V"
  elif [[ -f "$ROOT/$V/run.log" ]]; then
    echo "[RUNNING/INCOMPLETE] $V"
  else
    echo "[WAIT] $V"
  fi
done
if [[ -f "$ROOT/QUEUE_COMPLETE.txt" ]]; then
  echo "[QUEUE] COMPLETE"
elif [[ -s "$ROOT/launcher/pid.txt" ]]; then
  PID="$(cat "$ROOT/launcher/pid.txt")"
  if kill -0 "$PID" 2>/dev/null; then echo "[QUEUE] RUNNING pid=$PID"; else echo "[QUEUE] STOPPED pid=$PID"; fi
fi
if [[ -f "$ROOT/FINAL_EFFICIENCY.csv" ]]; then
  echo
  column -s, -t "$ROOT/FINAL_EFFICIENCY.csv" || cat "$ROOT/FINAL_EFFICIENCY.csv"
  echo
  cat "$ROOT/FINAL_EFFICIENCY_VERDICT.txt"
fi
