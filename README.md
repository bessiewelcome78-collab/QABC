# QABC: Query-Anchored Boundary Correction for Vision-Language Medical Image Segmentation

Official implementation of **QABC (Query-Anchored Boundary Correction)**, a lightweight boundary refinement framework for text-conditioned medical image segmentation.

QABC treats the prediction of a pretrained vision-language segmenter as a **semantic anchor** and performs only constrained local correction around the predicted decision contour. Instead of re-predicting the entire mask, QABC focuses on boundary geometry while preserving the semantic localization ability of the base model.

---

## Overview

Vision-language medical segmentation models can provide reliable semantic localization from text queries, but their predicted masks may still contain local boundary misalignment.

QABC addresses this problem through three key designs:

1. **Query-Anchored Correction Evidence**  
   Local boundary support, uncertainty, image-edge evidence, signed contour-normal information, local logit contrast, and query-conditioned decoder features are combined to estimate boundary correction.

2. **Constrained Boundary Correction**  
   The predicted correction is spatially restricted to the contour neighborhood and bounded to prevent unstable global mask deformation. A mass-tangent projection further suppresses unintended foreground expansion or contraction.

3. **Zero-Forward Shadow-Residual Learning**  
   During training, the correction branch receives gradients without perturbing the forward prediction of the semantic anchor. At inference time, the learned residual is applied to refine the boundary.

The QABC correction head introduces only **8,882 trainable parameters**.

---

## Framework

The base vision-language segmentation model produces an initial logit map

\[
z_0
\]

and a query-conditioned decoder representation.

QABC constructs contour-local correction evidence from:

- local contour band;
- probability uncertainty;
- image-edge evidence;
- signed contour-normal geometry;
- query-conditioned decoder features;
- local logit contrast.

The correction head predicts a bounded residual that is constrained before being applied to the base prediction.

The final inference prediction is obtained by refining the base logits rather than re-predicting the complete segmentation mask.

---

## Repository Structure

```text
QABC/
├── configs/                 # Experiment configurations
├── datasets/                # Dataset loading and preprocessing
├── open_clip_lib/           # OpenCLIP / UniMedCLIP related utilities
├── repro/                   # Reproducibility utilities and experiment scripts
├── scripts/                 # Training / evaluation / analysis scripts
├── tools/                   # Auxiliary tools
├── trainers/                # Base segmenter and QABC implementation
│   ├── qabr.py
│   ├── qabr_v10_legacy.py
│   └── medclipseg_unimedclip.py
├── utils/                   # Evaluation, metrics, and common utilities
│   ├── eval.py
│   ├── main_utils.py
│   └── metrics_2d.py
├── train.py                 # Training entry point
├── test.py                  # Testing entry point
├── environment.yml          # Conda environment
├── requirements-lock.txt    # Exact Python package snapshot
└── README.md
