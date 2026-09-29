#!/usr/bin/env bash
set -Eeuo pipefail
VARIANT="${1:?BASE|FULL}"
GPU="${2:?physical GPU index}"
SEED="${3:-42}"
PROJECT="${PROJECT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
PYTHON="${PYTHON:-/home/tsz-25/miniconda3/envs/py3.10torch2.9.1cu128/bin/python}"
SOURCE_STUDY_ID="${SOURCE_STUDY_ID:-QABR_REVIEW_2GPU_20260916_184856}"
EFF_ID="${EFF_ID:?EFF_ID must be exported}"
ROOT="$PROJECT/efficiency_results/$EFF_ID/$VARIANT"

source "$PROJECT/scripts/jbtlite/qabr_efficiency/common.sh"
CFG="$(eff_config "$PROJECT")"
CKPT="$(eff_checkpoint_for_variant "$PROJECT" "$SOURCE_STUDY_ID" "$VARIANT" "$SEED")"
mkdir -p "$ROOT"

if [[ -f "$ROOT/COMPLETE.txt" ]]; then
  echo "[SKIP] completed efficiency run: $VARIANT"
  exit 0
fi

eff_export_qabr_full_defaults
eff_export_runtime "$PROJECT" "$GPU" "$SEED"
OPTS=(
  DATASET.NAME Kvasir
  MODEL.QABR.BAND_KERNEL 5
  MODEL.QABR.MAX_LOGIT_DELTA 2.0
)
if [[ "$VARIANT" == "BASE" ]]; then
  OPTS+=(MODEL.QABR.ENABLED False)
elif [[ "$VARIANT" != "FULL" ]]; then
  echo "[FAIL] unsupported variant $VARIANT" >&2
  exit 2
fi

{
  echo "[CONTRACT] efficiency_id=$EFF_ID source_study=$SOURCE_STUDY_ID variant=$VARIANT seed=$SEED gpu=$GPU"
  echo "[CONTRACT] input=1x3x224x224 synthetic; prompt=Segment the polyp.; disk_io=excluded"
  echo "[CONTRACT] benchmark=MC1+MC30; warmup_mc1=${WARMUP_MC1:-8}; repeat_mc1=${REPEAT_MC1:-30}; warmup_mc30=${WARMUP_MC30:-2}; repeat_mc30=${REPEAT_MC30:-8}"
  echo "[CONTRACT] TF32=off; inference_mode=on; latency=wall-clock with CUDA synchronization"
  echo "[CONTRACT] qabr_conv_flops=2*executed_qabr_conv_macs; parameter-free ops excluded"
  echo "[CONTRACT] checkpoint=$CKPT"
  echo "[CONTRACT] config=$CFG"
} | tee "$ROOT/contract.log"

cd "$PROJECT"
"$PYTHON" -u tools/qabr_efficiency_benchmark.py \
  --config-file "$CFG" \
  --checkpoint "$CKPT" \
  --variant "$VARIANT" \
  --seed "$SEED" \
  --output-json "$ROOT/result.json" \
  --output-csv "$ROOT/result.csv" \
  --input-size 224 \
  --prompt "Segment the polyp." \
  --warmup-mc1 "${WARMUP_MC1:-8}" \
  --repeat-mc1 "${REPEAT_MC1:-30}" \
  --warmup-mc30 "${WARMUP_MC30:-2}" \
  --repeat-mc30 "${REPEAT_MC30:-8}" \
  "${OPTS[@]}" 2>&1 | tee "$ROOT/run.log"

touch "$ROOT/COMPLETE.txt"
echo "[DONE] efficiency $VARIANT"
