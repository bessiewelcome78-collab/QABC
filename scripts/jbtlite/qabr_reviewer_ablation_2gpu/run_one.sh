#!/usr/bin/env bash
set -Eeuo pipefail

DATASET="${1:?BUSI|Kvasir}"
VARIANT="${2:?ablation variant}"
GPU="${3:?physical GPU index}"
SEED="${4:-42}"

PROJECT="${PROJECT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
PYTHON="${PYTHON:-/home/tsz-25/miniconda3/envs/py3.10torch2.9.1cu128/bin/python}"
DATA_DIR="${DATA_DIR:-/home/tsz-25/MedCLIPSeg-main/data}"
STUDY_ID="${STUDY_ID:?STUDY_ID must be exported by the launcher}"
ROOT="$PROJECT/reviewer_ablation_results/$STUDY_ID/$DATASET/$VARIANT/seed$SEED"
TAG="QABR_REVIEW_${DATASET}_${VARIANT}_S${SEED}"

# shellcheck source=common.sh
source "$PROJECT/scripts/jbtlite/qabr_reviewer_ablation_2gpu/common.sh"
CFG="$(qabr_config_for_dataset "$PROJECT" "$DATASET")"

[[ -x "$PYTHON" ]] || { echo "[FAIL] missing Python: $PYTHON" >&2; exit 3; }
[[ -s "$CFG" ]] || { echo "[FAIL] missing config: $CFG" >&2; exit 3; }
qabr_check_dataset "$DATA_DIR" "$DATASET"

if [[ -f "$ROOT/COMPLETE.txt" ]]; then
  echo "[SKIP] completed run: $DATASET/$VARIANT/seed$SEED"
  exit 0
fi

if find "$ROOT/train" -type f \( -name '*_best_val.pth' -o -name '*_last_epoch.pth' \) -print -quit 2>/dev/null | grep -q .; then
  echo "[FAIL] partial checkpoints already exist in $ROOT; use a new STUDY_ID instead of overwriting" >&2
  exit 29
fi

mkdir -p "$ROOT"/{train,val,test,logs}
printf 'RUNNING dataset=%s variant=%s seed=%s gpu=%s\n' "$DATASET" "$VARIANT" "$SEED" "$GPU" > "$ROOT/RUNNING.txt"
trap 'rc=$?; printf "%s\n" "$rc" > "$ROOT/EXIT_CODE.txt"; exit "$rc"' EXIT

qabr_export_full_defaults
qabr_apply_variant "$VARIANT"
qabr_export_runtime "$PROJECT" "$GPU" "$SEED"

mapfile -d '' -t OVR < <(qabr_dataset_overrides "$DATA_DIR" "$DATASET" "$TAG")
OVR+=("${QABR_EXTRA_OPTS[@]}")

{
  echo "[CONTRACT] study=$STUDY_ID dataset=$DATASET variant=$VARIANT seed=$SEED physical_gpu=$GPU"
  echo "[CONTRACT] config=$CFG"
  echo "[CONTRACT] epochs=100 val_select=best_val val_mc=10 test_mc=30 test_once=1"
  echo "[CONTRACT] primary_nsd=true2d tolerance=2 tolerance_space=model"
  echo "[SWITCH] global_support=${QABR_ABL_GLOBAL_SUPPORT:-0} mask_only=${QABR_ABL_MASK_ONLY:-0}"
  echo "[SWITCH] no_direction=${QABR_ABL_NO_DIRECTIONAL_GEOMETRY:-0} no_semantic=${QABR_ABL_NO_SEMANTIC_DETAIL:-0} no_explicit_query=${QABR_ABL_NO_QUERY_ANCHOR:-0}"
  echo "[SWITCH] tangent=$QABR_V15_TANGENT_PROJECTION shadow_only=$QABR_V15_TRAIN_SHADOW_ONLY readiness=$QABR_V15_READINESS_GATE eval_scale=$QABR_V15_EVAL_SCALE"
  echo "[OVERRIDES] ${OVR[*]}"
} | tee "$ROOT/logs/contract.log"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[DRY_RUN] contract validated; training not started"
  touch "$ROOT/DRY_RUN_COMPLETE.txt"
  exit 0
fi

cd "$PROJECT"
"$PYTHON" -u train.py --config-file "$CFG" --output-dir "$ROOT/train" --seed "$SEED" \
  "${OVR[@]}" 2>&1 | tee "$ROOT/logs/train.log"

CKPT_DIR="$ROOT/train/$DATASET/trained_models/seed$SEED"
CKPT="$(find "$CKPT_DIR" -maxdepth 1 -type f -name '*_best_val.pth' -print | sort | tail -n1)"
[[ -n "$CKPT" && -s "$CKPT" ]] || { echo "[FAIL] best-Val checkpoint missing: $CKPT_DIR" >&2; exit 20; }
printf '%s\n' "$CKPT" > "$ROOT/LOCKED_CHECKPOINT.txt"

"$PYTHON" -u test.py --config-file "$CFG" --seed "$SEED" --split val --prompt_design original \
  --num-samples 10 --checkpoint "$CKPT" --export-mode base --inference-batch-size 1 \
  --output-dir "$ROOT/val" "${OVR[@]}" 2>&1 | tee "$ROOT/logs/val_mc10.log"
for MODE in true2d paper_legacy; do
  "$PYTHON" -u utils/eval.py --config-file "$CFG" --seed "$SEED" --split val \
    --prompt_design original --output-dir "$ROOT/val" \
    --csv-name "${VARIANT}_val_mc10_${MODE}.csv" --nsd-mode "$MODE" \
    --nsd-tolerance 2 --nsd-tolerance-space model "${OVR[@]}" \
    2>&1 | tee "$ROOT/logs/eval_val_mc10_${MODE}.log"
done

# Test is deliberately opened only after the best-Val checkpoint is locked.
"$PYTHON" -u test.py --config-file "$CFG" --seed "$SEED" --split test --prompt_design original \
  --num-samples 30 --checkpoint "$CKPT" --export-mode base --inference-batch-size 1 \
  --output-dir "$ROOT/test" "${OVR[@]}" 2>&1 | tee "$ROOT/logs/test_mc30.log"
for MODE in true2d paper_legacy; do
  "$PYTHON" -u utils/eval.py --config-file "$CFG" --seed "$SEED" --split test \
    --prompt_design original --output-dir "$ROOT/test" \
    --csv-name "${VARIANT}_test_mc30_${MODE}.csv" --nsd-mode "$MODE" \
    --nsd-tolerance 2 --nsd-tolerance-space model "${OVR[@]}" \
    2>&1 | tee "$ROOT/logs/eval_test_mc30_${MODE}.log"
done

printf 'COMPLETE dataset=%s variant=%s seed=%s checkpoint=%s\n' "$DATASET" "$VARIANT" "$SEED" "$CKPT" > "$ROOT/COMPLETE.txt"
echo "[DONE] $DATASET/$VARIANT/seed$SEED"
