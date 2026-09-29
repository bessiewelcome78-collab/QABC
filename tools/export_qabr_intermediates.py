#!/usr/bin/env python3
"""Runtime visual exporter for the current JBT-Lite v15 QABR implementation.

This wrapper does NOT alter QABR outputs.  It monkey-patches only an eval-time
observer around ``trainers.qabr.QueryAnchoredBoundaryRefiner.forward`` and then
runs the project's existing ``test.py``.  Use --num-samples 1 and batch size 1
when collecting paper figures so one QABR call corresponds to one case.

Environment variables
---------------------
QABR_VIS_DIR       output root (default: ./qabr_intermediate_visuals)
QABR_VIS_MAX_CASES maximum number of QABR forward calls to export (default: 8)
QABR_VIS_DPI       PNG dpi (default: 180)

Example
-------
QABR_VIS_DIR=/tmp/qabr_vis QABR_VIS_MAX_CASES=8 \
python -u tools/export_qabr_intermediates.py \
  --config-file configs/jbtlite/qabr_v15/JBTL15_BUSI_QABR_PAPER100.yaml \
  --seed 42 --split val --num-samples 1 --checkpoint /path/to/best_val.pth \
  --export-mode base --inference-batch-size 1 --output-dir /tmp/qabr_vis_infer \
  DATASET.TRAIN_PATH /path/BUSI/Train_Folder/ \
  DATASET.VAL_PATH /path/BUSI/Val_Folder/ \
  DATASET.TEST_PATH /path/BUSI/Test_Folder/ \
  DATASET.TEXT_PROMPT_PATH /path/BUSI/Prompts_Folder/
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trainers.qabr import QueryAnchoredBoundaryRefiner


VIS_DIR = Path(os.environ.get("QABR_VIS_DIR", "./qabr_intermediate_visuals"))
MAX_CASES = max(1, int(os.environ.get("QABR_VIS_MAX_CASES", "8")))
DPI = max(80, int(os.environ.get("QABR_VIS_DPI", "180")))
_CALL_INDEX = 0
_ORIGINAL_FORWARD = QueryAnchoredBoundaryRefiner.forward


def _safe_name(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(x)).strip("_") or "case"


def _to_np(x: torch.Tensor) -> np.ndarray:
    a = x.detach().float().cpu().numpy()
    while a.ndim > 2:
        a = a[0]
    return a


def _minmax(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    lo = float(np.nanmin(a))
    hi = float(np.nanmax(a))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-8:
        return np.zeros_like(a, dtype=np.float32)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def _display_image(image: torch.Tensor | None, hw) -> np.ndarray:
    if image is None:
        return np.zeros(hw, dtype=np.float32)
    x = image.detach().float()
    if tuple(x.shape[-2:]) != tuple(hw):
        x = F.interpolate(x, size=hw, mode="bilinear", align_corners=False)
    x = x[0].mean(dim=0).cpu().numpy()
    # Input may be standardized; robust percentile scaling is safer than assuming [0,1].
    lo, hi = np.percentile(x, [1.0, 99.0])
    if hi - lo < 1e-8:
        return _minmax(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _save_map(path: Path, arr: np.ndarray, title: str, *, signed=False, v01=False, cmap=None):
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    arr = np.asarray(arr)
    if signed:
        vmax = float(np.nanmax(np.abs(arr)))
        vmax = max(vmax, 1e-6)
        im = ax.imshow(arr, cmap=cmap or "coolwarm", vmin=-vmax, vmax=vmax)
    elif v01:
        im = ax.imshow(arr, cmap=cmap or "magma", vmin=0.0, vmax=1.0)
    else:
        im = ax.imshow(arr, cmap=cmap or "viridis")
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _save_gray(path: Path, arr: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    ax.imshow(arr, cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _save_normal_overlay(path: Path, img: np.ndarray, p: np.ndarray,
                         nx: np.ndarray, ny: np.ndarray, hard_boundary: np.ndarray):
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
    try:
        ax.contour(p, levels=[0.5], colors=["yellow"], linewidths=1.4)
    except Exception:
        pass
    ys, xs = np.where(hard_boundary > 0.5)
    if len(xs) > 0:
        stride = max(1, len(xs) // 50)
        xs = xs[::stride]
        ys = ys[::stride]
        # image coordinates: +y is downward, so use ny directly with angles='xy'.
        ax.quiver(xs, ys, nx[ys, xs], ny[ys, xs], color="cyan",
                  angles="xy", scale_units="xy", scale=0.12,
                  width=0.004, headwidth=4.0, headlength=5.0)
    ax.set_title("预测轮廓与单位法向  n = ∇p / ||∇p||", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _save_overlay(path: Path, img: np.ndarray, p0: np.ndarray, pf: np.ndarray, band: np.ndarray):
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
    # editable band as a faint mask
    overlay = np.ma.masked_where(band <= 0.5, band)
    ax.imshow(overlay, cmap="autumn", alpha=0.18, vmin=0.0, vmax=1.0)
    try:
        c0 = ax.contour(p0, levels=[0.5], colors=["white"], linewidths=1.4, linestyles="--")
        cf = ax.contour(pf, levels=[0.5], colors=["red"], linewidths=1.6)
        c0.collections[0].set_label("coarse contour")
        cf.collections[0].set_label("refined contour")
        ax.legend(loc="lower right", fontsize=8, framealpha=0.8)
    except Exception:
        pass
    ax.set_title("粗轮廓 vs QABR 精细轮廓（淡色区域为可编辑带）", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _save_contact_sheet(path: Path, items):
    ncols = 4
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, 3.4 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, (title, arr, kind) in zip(axes, items):
        arr = np.asarray(arr)
        if kind == "gray":
            ax.imshow(arr, cmap="gray", vmin=0.0, vmax=1.0)
        elif kind == "signed":
            vmax = max(float(np.nanmax(np.abs(arr))), 1e-6)
            ax.imshow(arr, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        elif kind == "01":
            ax.imshow(arr, cmap="magma", vmin=0.0, vmax=1.0)
        else:
            ax.imshow(arr, cmap="viridis")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _extract_maps(module: QueryAnchoredBoundaryRefiner, args, kwargs, actual_result):
    decoder_features = args[0] if len(args) > 0 else kwargs.get("decoder_features")
    coarse_logits = module._find_base(args, kwargs)
    image = kwargs.get("image", args[2] if len(args) > 2 else None)
    output_size = kwargs.get("output_size", None)
    if output_size is None:
        output_size = image.shape[-2:] if image is not None else (
            coarse_logits.shape[-2] * 4, coarse_logits.shape[-1] * 4
        )
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    output_size = tuple(int(v) for v in output_size)

    base_hr = F.interpolate(coarse_logits, size=output_size, mode="bilinear", align_corners=False)
    p = torch.sigmoid(base_hr)
    uncertainty = (4.0 * p * (1.0 - p)).clamp(0.0, 1.0)

    px, py = module._finite_diff(p)
    eps = torch.finfo(p.dtype).eps
    mask_mag_raw = torch.sqrt(px.square() + py.square() + eps)
    mask_nx = px / mask_mag_raw.clamp_min(eps)
    mask_ny = py / mask_mag_raw.clamp_min(eps)
    coarse_edge = module._case_normalize(mask_mag_raw, robust_scale=2.0)

    hard = (p.detach() >= 0.5).to(p.dtype)
    hard_dilate = F.max_pool2d(hard, 3, stride=1, padding=1)
    hard_erode = 1.0 - F.max_pool2d(1.0 - hard, 3, stride=1, padding=1)
    hard_boundary = (hard_dilate - hard_erode).clamp(0.0, 1.0)
    band = F.max_pool2d(
        hard_boundary,
        kernel_size=module.band_kernel,
        stride=1,
        padding=module.band_kernel // 2,
    ).clamp(0.0, 1.0)

    image_edge, signed_normal = module._image_geometry(image, mask_nx, mask_ny, output_size)

    if module.detach_support:
        support_uncertainty = uncertainty.detach()
        support_band = band.detach()
        support_edge = coarse_edge.detach()
        image_edge_for_support = image_edge.detach()
        signed_normal_input = signed_normal.detach()
    else:
        support_uncertainty = uncertainty
        support_band = band
        support_edge = coarse_edge
        image_edge_for_support = image_edge
        signed_normal_input = signed_normal

    detail = module.detail_proj(decoder_features)
    detail = detail - F.avg_pool2d(detail, 3, stride=1, padding=1)
    detail = F.interpolate(detail, size=output_size, mode="bilinear", align_corners=False)
    detail_rms = torch.sqrt(detail.square().mean(dim=1, keepdim=True) + 1e-12)
    detail_rms = module._case_normalize(detail_rms, robust_scale=2.0)

    hp3 = torch.tanh(base_hr - F.avg_pool2d(base_hr, 3, stride=1, padding=1))
    hp5 = torch.tanh(base_hr - F.avg_pool2d(base_hr, 5, stride=1, padding=2))

    if module.ablate_no_query_anchor:
        p_input = torch.zeros_like(p)
        hp3_input = torch.zeros_like(hp3)
        hp5_input = torch.zeros_like(hp5)
    else:
        p_input, hp3_input, hp5_input = p, hp3, hp5

    detail_input = detail
    image_edge_input = image_edge.detach() if module.detach_support else image_edge
    if module.ablate_no_signed_direction:
        signed_normal_input = torch.zeros_like(signed_normal_input)
    if module.ablate_no_directional_geometry:
        image_edge_input = torch.zeros_like(image_edge_input)
        signed_normal_input = torch.zeros_like(signed_normal_input)
    if module.ablate_no_semantic_detail:
        detail_input = torch.zeros_like(detail_input)
        p_input = torch.zeros_like(p_input)
        hp3_input = torch.zeros_like(hp3_input)
        hp5_input = torch.zeros_like(hp5_input)
    if module.ablate_mask_only:
        detail_input = torch.zeros_like(detail_input)
        image_edge_input = torch.zeros_like(image_edge_input)
        signed_normal_input = torch.zeros_like(signed_normal_input)

    local_input = torch.cat([
        detail_input, p_input, support_uncertainty, support_edge,
        image_edge_input, signed_normal_input, hp3_input, hp5_input,
    ], dim=1)
    candidate_delta = module.max_logit_delta * torch.tanh(module.refine_head(local_input))
    local_evidence = torch.maximum(support_uncertainty, image_edge_for_support * support_band)
    support = support_band * (0.25 + 0.75 * local_evidence)
    alpha = torch.tanh(module.alpha)
    raw_corr = alpha * support * candidate_delta

    v15_band = module._hard_boundary_band(base_hr.detach())
    if module.v15_center:
        projected_corr, tangent_mass = module._probability_tangent_project(
            raw_corr, base_hr.detach(), v15_band
        )
    else:
        projected_corr = v15_band * raw_corr.clamp(-module.v15_corr_clip, module.v15_corr_clip)
        tangent_mass = projected_corr.new_zeros((projected_corr.shape[0], 1, 1, 1))
    projected_corr = projected_corr * module.v15_corr_scale

    deploy = float(module._time_deploy() * module._readiness() * module.v15_eval_scale)
    expected_final = base_hr + deploy * projected_corr
    actual_logits, _ = module._extract_tensor(actual_result)

    w = v15_band * (p * (1.0 - p)).detach()
    dims = tuple(range(2, raw_corr.ndim))
    mu = (w * raw_corr).sum(dim=dims, keepdim=True) / w.sum(dim=dims, keepdim=True).clamp_min(1e-6)

    return {
        "base_hr": base_hr,
        "p": p,
        "uncertainty": uncertainty,
        "mask_nx": mask_nx,
        "mask_ny": mask_ny,
        "coarse_edge": coarse_edge,
        "hard": hard,
        "hard_boundary": hard_boundary,
        "band": band,
        "image_edge": image_edge,
        "signed_normal": signed_normal,
        "detail_rms": detail_rms,
        "hp3": hp3,
        "hp5": hp5,
        "candidate_delta": candidate_delta,
        "support": support,
        "raw_corr": raw_corr,
        "v15_band": v15_band,
        "tangent_weight": w,
        "mu": mu,
        "projected_corr": projected_corr,
        "tangent_mass": tangent_mass,
        "deploy": deploy,
        "alpha_scalar": alpha.detach(),
        "expected_final": expected_final,
        "actual_final": actual_logits,
        "image": image,
    }


def _export_case(m, case_dir: Path):
    case_dir.mkdir(parents=True, exist_ok=True)
    hw = tuple(m["base_hr"].shape[-2:])
    img = _display_image(m["image"], hw)

    arrays = {k: _to_np(v) for k, v in m.items() if torch.is_tensor(v) and k != "image"}
    final_p = torch.sigmoid(m["actual_final"])
    expected_p = torch.sigmoid(m["expected_final"])
    arrays["final_p"] = _to_np(final_p)
    arrays["expected_p"] = _to_np(expected_p)
    arrays["input_gray"] = img

    np.savez_compressed(case_dir / "raw_maps.npz", **arrays)
    with open(case_dir / "meta.txt", "w", encoding="utf-8") as f:
        f.write(f"deploy={m['deploy']:.8f}\n")
        f.write(f"alpha={float(m['alpha_scalar'].flatten()[0]):+.8e}\n")
        f.write(f"tangent_mass_abs_max={float(m['tangent_mass'].abs().max()):.8e}\n")
        f.write(f"mu={float(m['mu'].flatten()[0]):+.8e}\n")
        f.write(f"actual_expected_max_abs={float((m['actual_final']-m['expected_final']).abs().max()):.8e}\n")

    _save_gray(case_dir / "00_input.png", img, "输入图像 I")
    _save_map(case_dir / "01_logit_z0.png", arrays["base_hr"], "语义锚点 / 高分辨率 logit  z0", signed=True)
    _save_map(case_dir / "02_probability_p.png", arrays["p"], "前景概率  p = sigmoid(z0)", v01=True)
    _save_map(case_dir / "03_uncertainty_u.png", arrays["uncertainty"], "QABR 不确定性  u = 4p(1-p)", v01=True)
    _save_gray(case_dir / "04_hard_mask.png", arrays["hard"], "当前硬分割  M = 1[p>=0.5]")
    _save_gray(case_dir / "05_hard_boundary.png", arrays["hard_boundary"], "硬轮廓  Dilate(M)-Erode(M)")
    _save_gray(case_dir / "06_boundary_band_B.png", arrays["band"], "可编辑边界带 B")
    _save_map(case_dir / "07_coarse_edge_ep.png", arrays["coarse_edge"], "预测轮廓强度 e_p", v01=True)
    _save_map(case_dir / "08_image_edge_eI.png", arrays["image_edge"], "图像边缘强度 e_I", v01=True)
    _save_map(case_dir / "09_signed_normal_sn.png", arrays["signed_normal"], "法向有符号图像变化 s_n", signed=True)
    _save_normal_overlay(case_dir / "10_normal_vectors.png", img, arrays["p"], arrays["mask_nx"], arrays["mask_ny"], arrays["hard_boundary"])
    _save_map(case_dir / "11_decoder_detail_D.png", arrays["detail_rms"], "压缩解码细节 D（通道 RMS）", v01=True)
    _save_map(case_dir / "12_hp3.png", arrays["hp3"], "查询锚点高通 h3", signed=True)
    _save_map(case_dir / "13_hp5.png", arrays["hp5"], "查询锚点高通 h5", signed=True)
    _save_map(case_dir / "14_candidate_delta.png", arrays["candidate_delta"], "有界候选修正 delta", signed=True)
    _save_map(case_dir / "15_support_S.png", arrays["support"], "局部支持门控 S", v01=True)
    _save_map(case_dir / "16_raw_correction_d.png", arrays["raw_corr"], "投影前局部修正 d", signed=True)
    _save_map(case_dir / "17_tangent_weight_w.png", arrays["tangent_weight"], "切空间权重 w = B p(1-p)", v01=True)
    _save_map(case_dir / "18_projected_correction_dhat.png", arrays["projected_corr"], "概率质量投影后修正 d_hat", signed=True)
    _save_map(case_dir / "19_final_probability.png", arrays["final_p"], "QABR 最终概率", v01=True)
    _save_overlay(case_dir / "20_coarse_vs_refined_overlay.png", img, arrays["p"], arrays["final_p"], arrays["band"])

    _save_contact_sheet(case_dir / "21_QABR_pipeline_contact_sheet.png", [
        ("Input I", img, "gray"),
        ("logit z0", arrays["base_hr"], "signed"),
        ("p", arrays["p"], "01"),
        ("u=4p(1-p)", arrays["uncertainty"], "01"),
        ("hard mask M", arrays["hard"], "gray"),
        ("boundary band B", arrays["band"], "gray"),
        ("coarse edge e_p", arrays["coarse_edge"], "01"),
        ("image edge e_I", arrays["image_edge"], "01"),
        ("signed normal s_n", arrays["signed_normal"], "signed"),
        ("decoder detail D", arrays["detail_rms"], "01"),
        ("h3", arrays["hp3"], "signed"),
        ("h5", arrays["hp5"], "signed"),
        ("candidate delta", arrays["candidate_delta"], "signed"),
        ("support S", arrays["support"], "01"),
        ("raw correction d", arrays["raw_corr"], "signed"),
        ("projected correction d_hat", arrays["projected_corr"], "signed"),
        ("tangent weight w", arrays["tangent_weight"], "01"),
        ("final p", arrays["final_p"], "01"),
    ])


def _wrapped_forward(self, *args, **kwargs):
    global _CALL_INDEX
    result = _ORIGINAL_FORWARD(self, *args, **kwargs)
    if self.training or _CALL_INDEX >= MAX_CASES:
        return result
    # Export only when explicitly enabled by invoking this wrapper.
    try:
        with torch.no_grad():
            maps = _extract_maps(self, args, kwargs, result)
        case_dir = VIS_DIR / f"case_{_CALL_INDEX:04d}"
        _export_case(maps, case_dir)
        print(f"[QABR_VIS] saved {case_dir}", flush=True)
        _CALL_INDEX += 1
    except Exception as exc:
        print(f"[QABR_VIS][WARN] export failed at call {_CALL_INDEX}: {type(exc).__name__}: {exc}", flush=True)
        _CALL_INDEX += 1
    return result


def main():
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    QueryAnchoredBoundaryRefiner.forward = _wrapped_forward
    # test.py owns the official config/checkpoint/data protocol. Keep all user
    # command-line arguments untouched and execute its main() in-process.
    import test as project_test
    project_test.main()


if __name__ == "__main__":
    main()
