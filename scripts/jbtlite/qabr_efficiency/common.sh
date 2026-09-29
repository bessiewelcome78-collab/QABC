#!/usr/bin/env bash
set -Eeuo pipefail

eff_project_default() {
  cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd
}

eff_config() {
  local project="$1"
  printf '%s\n' "$project/configs/jbtlite/qabr_v15_other3/JBTL15_Kvasir_QABR_FULL_PAPER100.yaml"
}

eff_export_runtime() {
  local project="$1" gpu="$2" seed="$3"
  export CUDA_VISIBLE_DEVICES="$gpu"
  export PYTHONHASHSEED="$seed"
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
  export TOKENIZERS_PARALLELISM=false
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export PYTHONPATH="$project${PYTHONPATH:+:$PYTHONPATH}"
}

eff_export_qabr_full_defaults() {
  export QABR_V15_DEPLOY_START_FRAC=0.10
  export QABR_V15_DEPLOY_END_FRAC=0.30
  export QABR_V15_SHADOW_WEIGHT=1.00
  export QABR_V15_EVAL_SCALE=1.00
  export QABR_V15_ALPHA_READY=0.04
  export QABR_V15_CORR_CLIP=1.25
  export QABR_V15_CORR_SCALE=0.90
  export QABR_V15_TANGENT_PROJECTION=1
  export QABR_V15_TANGENT_PASSES=2
  export QABR_V15_READINESS_GATE=1
  export QABR_V15_TRAIN_SHADOW_ONLY=1
  export QABR_V15_DEBUG=0
  unset QABR_ABL_NO_QUERY_ANCHOR QABR_ABL_NO_SIGNED_DIRECTION
  unset QABR_ABL_GLOBAL_SUPPORT QABR_ABL_NO_DIRECTIONAL_GEOMETRY
  unset QABR_ABL_NO_SEMANTIC_DETAIL QABR_ABL_MASK_ONLY
}

eff_checkpoint_for_variant() {
  local project="$1" source_study="$2" variant="$3" seed="$4"
  local lock="$project/reviewer_ablation_results/$source_study/Kvasir/$variant/seed$seed/LOCKED_CHECKPOINT.txt"
  [[ -s "$lock" ]] || { echo "[FAIL] missing checkpoint lock: $lock" >&2; return 3; }
  local ckpt
  ckpt="$(head -n1 "$lock" | tr -d '\r')"
  [[ -s "$ckpt" ]] || { echo "[FAIL] checkpoint missing: $ckpt" >&2; return 3; }
  printf '%s\n' "$ckpt"
}
