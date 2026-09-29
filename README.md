# QABC: Query-Anchored Boundary Correction via Directional Contour Signals for Medical Vision-Language Segmentation

Official implementation of **QABC (Query-Anchored Boundary Correction)**.

**Sizhe Tan**, **Shengchang Wang**, **Panpan Zheng**

School of Computer Science and Technology, Xinjiang University, Ürümqi 830046, China

[**Code**](https://github.com/bessiewelcome78-collab/QABC)

---

## Overview

Text-conditioned medical image segmentation can provide reliable semantic localization, but semantic correctness does not necessarily guarantee accurate boundary geometry. Local leakage, indentation, or contour displacement may remain around an otherwise correctly localized target.

QABC addresses this problem by preserving the existing query-conditioned prediction as a **semantic anchor** and learning only a constrained correction around its established decision contour, rather than re-predicting the complete segmentation mask.

<p align="center">
  <img src="assets/motivation.png" width="95%">
</p>

<p align="center">
  <em>
  Motivation for QABC: residual errors are contour-localized, motivating bounded correction around the established prediction rather than unconstrained full-mask updates.
  </em>
</p>

> **Abstract:**  
> Text-conditioned medical image segmentation has advanced rapidly, yet semantic correctness does not guarantee geometric accuracy: a model may identify the queried structure while retaining residual contour errors. Re-predicting the whole mask may repair these errors but can disturb already-correct regions. We introduce QABC, a lightweight query-anchored boundary correction framework that preserves the existing semantic prediction and edits only its contour neighborhood. QABC combines predictive uncertainty, normal-referenced image transitions, and query-conditioned local detail to predict bounded signed logit corrections. Local support restricts the edit region, while a first-order mass-tangent projection suppresses systematic foreground-mass drift. Zero-forward shadow-residual learning optimizes the correction branch while keeping the training forward prediction unchanged. Across four medical benchmarks, QABC consistently improves the matched host and attains the highest listed DSC and NSD with only 8,882 additional trainable parameters.

---

## Method

<p align="center">
  <img src="assets/framework.png" width="100%">
</p>

<p align="center">
  <em>
  QABC overview. The host prediction anchors local support; directional and query-conditioned evidence predicts a bounded correction, which is gated and mass-projected before shadow-residual training or explicit inference-time deployment.
  </em>
</p>

QABC preserves the query-conditioned prediction of the base segmenter as a semantic anchor and learns only a constrained local correction around its established decision contour.

### Query-Anchored Correction Evidence

Given a medical image \(I\) and text query \(q\), the base text-conditioned segmenter produces an image-resolution logit map \(z_0\) and a query-conditioned decoder feature \(F_d\), with

\[
p_0 = \sigma(z_0).
\]

Since \(p_0=0.5\) is equivalent to \(z_0=0\), this level set defines the current decision contour.

QABC constructs correction evidence from three complementary sources:

1. **Contour-localized support.**  
   A narrow support \(B\) restricts where correction is permitted, while

   \[
   u = 4p_0(1-p_0)
   \]

   describes predictive ambiguity around the current decision boundary.

2. **Normal-referenced directional geometry.**  
   Prediction-edge strength \(e_p\), image-edge strength \(e_I\), and the signed normal-referenced image transition \(s_n\) provide geometric evidence about the local contour and its potential correction direction.

3. **Query-conditioned local detail and logit contrast.**  
   Query-conditioned decoder information \(F_q\) is combined with multi-scale logit contrasts \(h_3\) and \(h_5\) to retain target-specific local evidence.

The complete correction evidence is

\[
E =
\operatorname{Concat}
(F_q,p_0,u,e_p,e_I,s_n,h_3,h_5),
\]

and the lightweight correction head predicts a bounded signed-logit proposal

\[
\delta = 2\tanh(f_\theta(E)).
\]

The correction head consists of a 3×3 Conv-GN-GELU block, a depthwise 3×3 Conv-GN-GELU block, and a 1×1 output convolution.

---

### Constrained Boundary Correction

The proposal \(\delta\) is not directly applied to the segmentation.

QABC first constructs an evidence-gated local support

\[
S =
B \odot
\left[
0.25 + 0.75
\max(u,B\odot e_I)
\right],
\]

and obtains

\[
d=\alpha(S\odot\delta),
\qquad
\alpha=\tanh(\eta),
\]

where \(\eta\) is initialized to zero so that QABC starts from an identity mapping.

Although the correction is contour-local, same-signed residuals may accumulate into systematic foreground expansion or contraction. QABC therefore applies a **first-order mass-tangent projection**.

With

\[
w_x=B_xp_{0,x}(1-p_{0,x}),
\]

the weighted common component is removed:

\[
\mu =
\frac{\sum_x w_x d_x}
{\sum_x w_x+\epsilon},
\qquad
r_x=B_x(d_x-\mu).
\]

This gives the first-order constraint

\[
\sum_x
p_{0,x}(1-p_{0,x})r_x
\approx 0,
\]

which suppresses dominant foreground-mass drift while still permitting positive and negative local contour corrections.

---

### Shadow-Residual Learning and Inference

Directly applying an immature residual during training may disturb an already meaningful semantic prediction.

QABC therefore separates **learning the correction** from **applying the correction** using a zero-forward shadow residual:

\[
r_{\mathrm{sh}} = r-\operatorname{sg}(r),
\]

\[
z_{\mathrm{train}}
=
z_0+r_{\mathrm{sh}}
=
z_0.
\]

Although the forward residual is zero, its gradient with respect to \(r\) remains active, allowing the correction branch to receive segmentation gradients without changing the host forward prediction.

At inference time, the learned residual is explicitly applied:

\[
z_{\mathrm{ref}}=z_0+\rho r,
\qquad
p_{\mathrm{ref}}=\sigma(z_{\mathrm{ref}}).
\]

The final segmentation is obtained by averaging 30 stochastic forward probability maps and thresholding the mean prediction at 0.5.

---

## Experimental Setup

### Datasets

QABC is evaluated on four medical image segmentation benchmarks:

- **BUSI** — breast ultrasound
- **BTMRI** — brain MRI
- **ISIC** — dermoscopy
- **Kvasir-SEG** — gastrointestinal endoscopy

Cross-domain evaluation is additionally conducted on:

- BUSI → BUID
- Kvasir-SEG → CVC-ColonDB
- Kvasir-SEG → CVC-ClinicDB
- Kvasir-SEG → BKAI
- BTMRI → BRISC

No target-domain adaptation is used for the cross-domain experiments.

### Host Model and Training Protocol

The host model uses:

- **UniMedCLIP ViT-B/16**
- **BiomedBERT**

The pretrained encoders are frozen.

The host training objective uses:

```text
Dice loss weight : 0.5
CE loss weight   : 0.5
CLIP loss weight : 0.1
Epochs           : 100
Optimizer        : Adam
Learning rate    : 3e-4
Batch size       : 24
LR schedule      : Cosine annealing
GPU              : NVIDIA A40
Test inference   : 30 stochastic passes
```

QABC uses an eight-channel detail projection and a 32-channel correction head.

---

## Main Results

### Protocol-Matched Comparison

DSC and NSD are reported in percentage (%).

| Method | BUSI DSC | BUSI NSD | BTMRI DSC | BTMRI NSD | ISIC DSC | ISIC NSD | Kvasir-SEG DSC | Kvasir-SEG NSD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MedCLIPSeg (Host) | 85.00 | 87.63 | 88.21 | 91.85 | 92.12 | 93.16 | 89.17 | 91.33 |
| **Host + QABC** | **85.95** | **88.58** | **88.51** | **92.27** | **92.38** | **93.41** | **90.39** | **92.54** |

The NSD values in this table follow the **benchmark NSD protocol**.

They are not numerically interchangeable with the stricter per-image two-pixel **true2D NSD** used in the ablation study below.

---

## Ablation Study

The matched component-removal ablation is performed on **Kvasir-SEG**.

| Variant | DSC ↑ | true2D NSD ↑ |
|---|---:|---:|
| Base | 89.17 | 59.81 |
| w/o Local Support | 89.37 | 61.79 |
| w/o Uncertainty Evidence | 89.32 | 61.45 |
| w/o Directional Geometry | 89.03 | 60.43 |
| w/o Semantic Detail | 89.61 | 61.49 |
| w/o Mass-Tangent | 89.19 | 61.93 |
| w/o Shadow Residual | 89.75 | 62.69 |
| **Full QABC** | **90.39** | **64.95** |

Full QABC improves the matched Base by **1.22 DSC** and **5.14 true2D NSD**.

Among the component-removal experiments, removing directional geometry produces the largest boundary degradation, reducing true2D NSD by 4.52 points.

---

## Qualitative Results

<p align="center">
  <img src="assets/qualitative.png" width="92%">
</p>

<p align="center">
  <em>
  QABC intermediate evidence and final predictions across four imaging modalities.
  </em>
</p>

The visualization shows the query-conditioned prediction, uncertainty, contour support, directional evidence, and final segmentation across BUSI, BTMRI, ISIC, and Kvasir-SEG.

---

## Efficiency

QABC is designed as a lightweight correction module.

Efficiency is measured at **224×224 resolution**, batch size 1, on one NVIDIA A40.

| Method | Extra Params | Extra GFLOPs / MC | MC30 Latency | Peak Memory |
|---|---:|---:|---:|---:|
| Host | 0 | 0 | 1262.80 ms | 887.44 MiB |
| Host + QABC | **+8,882** | **0.491** | **1508.32 ms (+19.44%)** | **889.19 MiB (+0.20%)** |

QABC therefore adds only **8,882 trainable parameters** and **0.491 GFLOPs per stochastic pass**, with a **0.20% peak-memory overhead**.

---

## Domain Generalization

<p align="center">
  <img src="assets/domain_generalization.png" width="95%">
</p>

<p align="center">
  <em>
  Source-to-target DSC without target-domain adaptation; source-domain columns are shown for reference.
  </em>
</p>

The QABC results reported in the paper are:

| Dataset | DSC (%) |
|---|---:|
| BUSI | 85.95 |
| BUID | 79.51 |
| Kvasir-SEG | 90.39 |
| ColonDB | 69.84 |
| ClinicDB | 78.54 |
| BKAI | 77.99 |
| BTMRI | 88.51 |
| BRISC | 81.33 |

---

## Repository Structure

```text
QABC/
├── assets/                 # Figures used in README
├── configs/                # Experiment configurations
├── datasets/               # Dataset loaders
├── open_clip_lib/          # OpenCLIP-related components
├── repro/                  # Reproducibility scripts
├── scripts/                # Experiment scripts
├── tools/                  # Auxiliary tools
├── trainers/
│   ├── qabr.py
│   ├── qabr_v10_legacy.py
│   └── medclipseg_unimedclip.py
├── utils/
│   ├── eval.py
│   ├── main_utils.py
│   └── metrics_2d.py
├── train.py
├── test.py
└── README.md
```

---

## Installation

Environment configuration files will be provided with the repository.

After `environment.yml` is added, the environment can be created with:

```bash
conda env create -f environment.yml
conda activate qabc
```

The exact software dependencies should follow the environment used for the reported experiments.

---

## Pretrained Models

The host uses **UniMedCLIP ViT-B/16** and **BiomedBERT**.

Large pretrained weights are not distributed directly in this Git repository.

Place the required pretrained models under the local checkpoint directory expected by the corresponding experiment configuration.

```text
checkpoints/
├── unimed_clip_vit_b16.pt
└── BiomedBERT-base-uncased-abstract/
```

---

## Data Preparation

Dataset files are not redistributed in this repository.

Please prepare the datasets locally according to the corresponding configuration files.

```text
data/
├── BUSI/
├── BTMRI/
├── ISIC/
└── Kvasir-SEG/
```

Cross-domain datasets used in the paper include BUID, CVC-ColonDB, CVC-ClinicDB, BKAI, and BRISC.

---

## Training and Evaluation

The main entry points are:

```bash
python train.py
```

and

```bash
python test.py
```

Experiment-specific configurations and scripts are provided under:

```text
configs/
repro/
scripts/
```

For exact reproduction, use the corresponding experiment configuration rather than manually modifying the implementation.

---

## Citation

If you find this work useful, please consider citing our paper.

Citation information will be updated after publication.

```bibtex
@inproceedings{tan2027qabc,
  title  = {QABC: Query-Anchored Boundary Correction via Directional Contour Signals for Medical Vision-Language Segmentation},
  author = {Tan, Sizhe and Wang, Shengchang and Zheng, Panpan},
  year   = {2027}
}
```

---

## Acknowledgements

QABC is developed on top of a vision-language medical segmentation host using UniMedCLIP ViT-B/16 and BiomedBERT.

We thank the authors of the related vision-language segmentation models, pretrained models, and public medical segmentation datasets used in this work.
