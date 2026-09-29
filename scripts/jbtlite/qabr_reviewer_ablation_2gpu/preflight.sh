#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="${PROJECT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
PYTHON="${PYTHON:-/home/tsz-25/miniconda3/envs/py3.10torch2.9.1cu128/bin/python}"
DATA_DIR="${DATA_DIR:-/home/tsz-25/MedCLIPSeg-main/data}"
DIR="$PROJECT/scripts/jbtlite/qabr_reviewer_ablation_2gpu"
MATRIX="$PROJECT/configs/jbtlite/qabr_reviewer_ablation/ablation_matrix.tsv"

[[ -x "$PYTHON" ]] || { echo "[FAIL] Python not executable: $PYTHON" >&2; exit 3; }
[[ -s "$MATRIX" ]] || { echo "[FAIL] missing matrix: $MATRIX" >&2; exit 3; }
for cfg in \
  "$PROJECT/configs/jbtlite/qabr_v15/JBTL15_BUSI_QABR_PAPER100.yaml" \
  "$PROJECT/configs/jbtlite/qabr_v15_other3/JBTL15_Kvasir_QABR_FULL_PAPER100.yaml"; do
  [[ -s "$cfg" ]] || { echo "[FAIL] missing base config: $cfg" >&2; exit 3; }
done
for dataset in BUSI Kvasir; do
  for d in Train_Folder Val_Folder Test_Folder Prompts_Folder; do
    [[ -d "$DATA_DIR/$dataset/$d" ]] || { echo "[FAIL] missing $DATA_DIR/$dataset/$d" >&2; exit 3; }
  done
done

for script in "$DIR"/*.sh; do bash -n "$script"; done
"$PYTHON" -m py_compile "$DIR/collect_results.py"

grep -q 'QABR_V15_TANGENT_PROJECTION' "$PROJECT/trainers/qabr.py"
grep -q 'QABR_V15_TRAIN_SHADOW_ONLY' "$PROJECT/trainers/qabr.py"
grep -q 'QABR_V15_READINESS_GATE' "$PROJECT/trainers/qabr.py"
grep -q 'QABR_ABL_NO_DIRECTIONAL_GEOMETRY' "$PROJECT/trainers/qabr_v10_legacy.py"
grep -q 'QABR_ABL_NO_SEMANTIC_DETAIL' "$PROJECT/trainers/qabr_v10_legacy.py"
grep -q 'QABR_ABL_MASK_ONLY' "$PROJECT/trainers/qabr_v10_legacy.py"

echo "[PASS] reviewer ablation preflight"
