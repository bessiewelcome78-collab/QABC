#!/usr/bin/env bash
set -Eeuo pipefail

qabr_project_default() {
  cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd
}

qabr_config_for_dataset() {
  local project="$1" dataset="$2"
  case "$dataset" in
    BUSI) printf '%s\n' "$project/configs/jbtlite/qabr_v15/JBTL15_BUSI_QABR_PAPER100.yaml" ;;
    Kvasir) printf '%s\n' "$project/configs/jbtlite/qabr_v15_other3/JBTL15_Kvasir_QABR_FULL_PAPER100.yaml" ;;
    *) echo "[FAIL] unsupported dataset: $dataset" >&2; return 2 ;;
  esac
}

qabr_dataset_overrides() {
  local data_dir="$1" dataset="$2" tag="$3"
  printf '%s\0' \
    DATASET.NAME "$dataset" \
    DATASET.TRAIN_PATH "$data_dir/$dataset/Train_Folder/" \
    DATASET.VAL_PATH "$data_dir/$dataset/Val_Folder/" \
    DATASET.TEST_PATH "$data_dir/$dataset/Test_Folder/" \
    DATASET.TEXT_PROMPT_PATH "$data_dir/$dataset/Prompts_Folder/" \
    TRAIN.RUN_TAG "$tag" \
    M1.RUN_TAG "$tag"
}

qabr_export_full_defaults() {
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
  export QABR_V15_DEBUG=1

  unset QABR_ABL_NO_QUERY_ANCHOR QABR_ABL_NO_SIGNED_DIRECTION
  unset QABR_ABL_GLOBAL_SUPPORT QABR_ABL_NO_DIRECTIONAL_GEOMETRY
  unset QABR_ABL_NO_SEMANTIC_DETAIL QABR_ABL_MASK_ONLY
}

qabr_apply_variant() {
  local variant="$1"
  QABR_EXTRA_OPTS=()
  case "$variant" in
    BASE)
      QABR_EXTRA_OPTS+=(MODEL.QABR.ENABLED False)
      ;;
    NAIVE_GLOBAL)
      export QABR_ABL_GLOBAL_SUPPORT=1
      export QABR_V15_TANGENT_PROJECTION=0
      ;;
    NO_LOCAL_SUPPORT)
      export QABR_ABL_GLOBAL_SUPPORT=1
      ;;
    LOCAL_ONLY)
      export QABR_ABL_MASK_ONLY=1
      ;;
    LOCAL_DIRECTION)
      export QABR_ABL_NO_SEMANTIC_DETAIL=1
      ;;
    LOCAL_SEMANTIC)
      export QABR_ABL_NO_DIRECTIONAL_GEOMETRY=1
      ;;
    NO_TANGENT)
      export QABR_V15_TANGENT_PROJECTION=0
      ;;
    DIRECT_RESIDUAL)
      export QABR_V15_TRAIN_SHADOW_ONLY=0
      ;;
    NO_READINESS)
      export QABR_V15_READINESS_GATE=0
      ;;
    FULL)
      ;;
    NO_EXPLICIT_QUERY)
      export QABR_ABL_NO_QUERY_ANCHOR=1
      ;;
    BAND3)
      QABR_EXTRA_OPTS+=(MODEL.QABR.BAND_KERNEL 3)
      ;;
    BAND7)
      QABR_EXTRA_OPTS+=(MODEL.QABR.BAND_KERNEL 7)
      ;;
    DELTA1)
      QABR_EXTRA_OPTS+=(MODEL.QABR.MAX_LOGIT_DELTA 1.0)
      ;;
    DELTA3)
      QABR_EXTRA_OPTS+=(MODEL.QABR.MAX_LOGIT_DELTA 3.0)
      ;;
    *)
      echo "[FAIL] unsupported training variant: $variant" >&2
      return 2
      ;;
  esac
}

qabr_export_runtime() {
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

qabr_check_dataset() {
  local data_dir="$1" dataset="$2"
  local d
  for d in Train_Folder Val_Folder Test_Folder Prompts_Folder; do
    [[ -d "$data_dir/$dataset/$d" ]] || {
      echo "[FAIL] missing dataset directory: $data_dir/$dataset/$d" >&2
      return 3
    }
  done
}
