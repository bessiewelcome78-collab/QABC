# QABC: Query-Anchored Boundary Correction for Vision–Language Medical Image Segmentation

Official implementation of **QABC (Query-Anchored Boundary Correction)**, a lightweight contour refinement framework for vision–language medical image segmentation.

QABC is designed to improve boundary geometry while preserving the semantic localization of an existing text-conditioned segmenter. Instead of re-predicting the complete segmentation mask, QABC treats the base prediction as a semantic anchor and performs only constrained local correction around its decision contour.

---

## Overview

Vision–language medical segmentation models can provide strong semantic localization from text queries. However, semantically correct predictions may still exhibit local contour misalignment, especially around ambiguous or low-contrast boundaries.

Directly re-predicting the entire mask can unnecessarily disturb regions that are already correctly localized. QABC therefore reformulates refinement as a **query-anchored local boundary correction problem**.

<p align="center">
  <img src="assets/QABC_Motivation.png" width="90%">
</p>

<p align="center">
  <em>Motivation of QABC. Rather than replacing the base segmentation, QABC preserves its semantic prediction and only corrects local contour errors.</em>
</p>

> **Abstract:**  
> Vision–language medical segmentation can localize targets through text queries, yet reliable semantic localization does not necessarily guarantee precise boundary geometry. We propose **QABC**, a lightweight query-anchored boundary correction framework that preserves the base prediction as a semantic anchor and learns constrained local corrections around its decision contour. QABC integrates probability ambiguity, image-edge evidence, signed contour-normal geometry, local logit contrast, and query-conditioned decoder features to estimate bounded corrections. A mass-tangent projection suppresses unintended global foreground expansion or contraction, while zero-forward shadow-residual learning enables the correction branch to be optimized without perturbing the base forward prediction during training. QABC therefore improves local boundary alignment without re-predicting the complete segmentation mask and introduces only **8,882 additional trainable parameters**.

---

## Method

<p align="center">
  <img src="assets/QABC_Framework.png" width="100%">
</p>

<p align="center">
  <em>Overall architecture of QABC. The base vision–language segmentation prediction is preserved as a semantic anchor, while QABC learns constrained residual correction around the predicted decision contour.</em>
</p>

QABC consists of three main components.

### 1. Query-Anchored Correction Evidence

Given an image and a text query, the base vision–language segmenter produces an initial segmentation logit map and query-conditioned decoder features.

QABC constructs local correction evidence around the current decision contour using complementary signals including:

- contour-local support;
- predictive uncertainty;
- image-edge evidence;
- signed contour-normal geometry;
- local logit contrast;
- query-conditioned decoder features.

These signals jointly determine **where**, **in which direction**, and **whether** the current contour should be corrected.

### 2. Constrained Boundary Correction

A lightweight correction head predicts a bounded local residual from the constructed evidence.

The correction is restricted to the contour neighborhood rather than being applied globally. A mass-tangent projection further removes the common expansion/contraction component, reducing unintended global foreground-area drift.

The correction head contains only **8,882 trainable parameters**.

### 3. Zero-Forward Shadow-Residual Learning

Directly injecting an unconverged residual during early training may damage the semantic localization already provided by the base model.

QABC therefore adopts **zero-forward shadow-residual learning**. During training, the residual branch receives gradients while the forward prediction remains identical to the semantic anchor. During inference, the learned residual is activated to refine the decision contour.

This separates:

- **semantic localization**, provided by the base model; and
- **boundary geometry correction**, learned by QABC.

---

## Main Characteristics

QABC is designed around four principles:

1. **Semantic Preservation**  
   The existing vision–language prediction is treated as an anchor instead of being replaced.

2. **Contour-Local Refinement**  
   Corrections are concentrated around the predicted decision boundary.

3. **Geometry-Aware Constraints**  
   Directional and mass-tangent constraints suppress uncontrolled mask deformation.

4. **Lightweight Adaptation**  
   The correction branch adds only **8,882 trainable parameters**.

---

## Experimental Setup

We evaluate QABC on four medical image segmentation benchmarks covering different imaging modalities and anatomical targets:

| Dataset | Modality / Task |
|---|---|
| BUSI | Breast ultrasound lesion segmentation |
| BTMRI | Brain MRI tumor segmentation |
| ISIC | Dermoscopic skin-lesion segmentation |
| Kvasir-SEG | Endoscopic polyp segmentation |

The standard experimental setting uses:

```text
Input resolution : 224 × 224
Random seed      : 42
Training epochs  : 100
Batch size       : 24
Optimizer        : Adam
Learning rate    : 3e-4
LR schedule      : Cosine
Model selection  : Best validation checkpoint
Test inference   : MC30
