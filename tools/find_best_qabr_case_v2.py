#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Find the clearest *validation* case where JBT-Lite v15 QABR helps.

The script scans BUSI / Kvasir / ISIC / BTMRI validation sets using the locked
seed-42 checkpoints, ranks cases by actual QABR improvement plus visible local
boundary movement, re-checks only the best candidates with a small MC average,
and exports the same full QABR intermediate visualization package as
``tools/export_qabr_intermediates.py``.

Important protocol note
-----------------------
This script deliberately selects qualitative examples on **Val only**.  It does
not inspect Test labels and it does not change the reported Test protocol.  The
224x224 Dice / surface-Dice values below are only case-selection diagnostics,
not paper-reported true2d metrics.

Expected project layout (defaults match the user's current project):
  formal_results_jbtlite_v15_seed42/<STUDY>/BUSI/QABR/seed42/
  formal_results_qabr_v15_other3/<STUDY>/{Kvasir,ISIC,BTMRI}/seed42/

Usage
-----
CUDA_VISIBLE_DEVICES=2 python -u tools/find_best_qabr_case.py \
  --project /home/tsz-25/MedCLIPSeg_JBTLite_QABR_V12_20260911 \
  --data-dir /home/tsz-25/MedCLIPSeg-main/data \
  --study-id JBTL15_EXACT_REPRO_S42_20260912_212152 \
  --scan-mc 1 --verify-mc 10 --preselect-k 8 --export-top-k 3
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Must be set before model construction because QABR reads these envs in __init__.
os.environ.setdefault("QABR_V15_DEPLOY_START_FRAC", "0.10")
os.environ.setdefault("QABR_V15_DEPLOY_END_FRAC", "0.30")
os.environ.setdefault("QABR_V15_SHADOW_WEIGHT", "1.00")
os.environ.setdefault("QABR_V15_EVAL_SCALE", "1.00")
os.environ.setdefault("QABR_V15_ALPHA_READY", "0.04")
os.environ.setdefault("QABR_V15_CORR_CLIP", "1.25")
os.environ.setdefault("QABR_V15_CORR_SCALE", "0.90")
os.environ.setdefault("QABR_V15_TANGENT_PROJECTION", "1")
os.environ.setdefault("QABR_V15_TANGENT_PASSES", "2")
os.environ.setdefault("QABR_V15_READINESS_GATE", "1")
os.environ.setdefault("QABR_V15_TRAIN_SHADOW_ONLY", "1")
os.environ.setdefault("QABR_V15_DEBUG", "0")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, default_collate

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATASET_SPECS = {
    "BUSI": {
        "config": "configs/jbtlite/qabr_v15/JBTL15_BUSI_QABR_PAPER100.yaml",
        "run_kind": "busi",
        "data_name": "BUSI",
    },
    "Kvasir": {
        "config": "configs/jbtlite/qabr_v15/JBTL15_Kvasir_FULL_PAPER100.yaml",
        "run_kind": "other3",
        "data_name": "Kvasir",
    },
    "ISIC": {
        "config": "configs/jbtlite/qabr_v15/JBTL15_ISIC_FULL_PAPER100.yaml",
        "run_kind": "other3",
        "data_name": "ISIC",
    },
    "BTMRI": {
        "config": "configs/jbtlite/qabr_v15/JBTL15_BTMRI_FULL_PAPER100.yaml",
        "run_kind": "other3",
        "data_name": "BTMRI",
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--study-id", default="JBTL15_EXACT_REPRO_S42_20260912_212152")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--datasets", nargs="+", default=["BUSI", "Kvasir", "ISIC", "BTMRI"],
                   choices=list(DATASET_SPECS))
    p.add_argument("--scan-mc", type=int, default=1,
                   help="Fast pass count for every Val case (1 recommended).")
    p.add_argument("--verify-mc", type=int, default=10,
                   help="MC passes used only for preselected cases.")
    p.add_argument("--preselect-k", type=int, default=8,
                   help="Fast-pass candidates rechecked per dataset.")
    p.add_argument("--export-top-k", type=int, default=3,
                   help="Verified candidates exported per dataset.")
    p.add_argument("--nsd-tolerance", type=int, default=2)
    p.add_argument("--max-cases-per-dataset", type=int, default=0,
                   help="0 scans the whole Val set; positive values are for smoke tests only.")
    p.add_argument("--output-dir", default="",
                   help="Default: <project>/qabr_case_search/<study-id>")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_checkpoint(project: Path, study_id: str, dataset: str) -> Tuple[Path, Path]:
    if dataset == "BUSI":
        run_dir = project / "formal_results_jbtlite_v15_seed42" / study_id / "BUSI" / "QABR" / "seed42"
    else:
        run_dir = project / "formal_results_qabr_v15_other3" / study_id / dataset / "seed42"
    lock = run_dir / "LOCKED_CHECKPOINT.txt"
    if not lock.is_file():
        raise FileNotFoundError(f"LOCKED_CHECKPOINT.txt not found: {lock}")
    raw = lock.read_text(encoding="utf-8").strip()
    if not raw:
        raise RuntimeError(f"Empty checkpoint lock: {lock}")
    ckpt = Path(raw)
    if not ckpt.is_absolute():
        ckpt = (project / ckpt).resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint referenced by lock does not exist: {ckpt}")
    return run_dir, ckpt


def load_visual_module(project: Path):
    path = project / "tools" / "export_qabr_intermediates.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Put export_qabr_intermediates.py in tools/ first."
        )
    spec = importlib.util.spec_from_file_location("qabr_visual_export", str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def load_model_and_dataset(project: Path, data_dir: Path, dataset: str, ckpt: Path, seed: int):
    # Imports happen after project is on sys.path.
    import test as project_test
    from datasets.dataloader import DatasetSegmentation, ValGenerator
    from utils.main_utils import load_cfg_from_cfg_file, read_text

    spec = DATASET_SPECS[dataset]
    cfg_path = project / spec["config"]
    cfg = load_cfg_from_cfg_file(str(cfg_path))
    ds = spec["data_name"]
    overrides = [
        "DATASET.NAME", ds,
        "DATASET.TRAIN_PATH", str(data_dir / ds / "Train_Folder"),
        "DATASET.VAL_PATH", str(data_dir / ds / "Val_Folder"),
        "DATASET.TEST_PATH", str(data_dir / ds / "Test_Folder"),
        "DATASET.TEXT_PROMPT_PATH", str(data_dir / ds / "Prompts_Folder"),
        "TRAIN.REPLAY_PUBLIC_REPO_MC_BURNIN", "false",
    ]
    cfg.merge_from_list(overrides)
    cfg.update({
        "seed": seed,
        "split": "val",
        "prompt_design": "original",
        "data_percentage": 100,
        "source_dataset": None,
        "output_dir": str(project / ".qabr_case_search_tmp"),
        "checkpoint": str(ckpt),
        "num_samples": 1,
        "export_mode": "base",
        "inference_batch_size": 1,
        "verifier_text_override": "",
        "allow_v15_identity_compat": False,
        "allow_v15_calibrator_compat": False,
    })

    set_seed(seed)
    model = project_test.build_model(cfg)
    checkpoint = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    incompatible = model.load_state_dict(dict(state), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint/model mismatch for {dataset}: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )

    cp_weight_source = str(checkpoint.get("weight_source", "unknown")).lower() if isinstance(checkpoint, dict) else "raw"
    m1_node = getattr(cfg, "M1", None)
    is_v393 = str(getattr(m1_node, "M1_LOSS_VERSION", "")).lower() == "v393_preserve_aware_edit_control" if m1_node is not None else False
    use_ema = (
        isinstance(checkpoint, dict)
        and "ema_shadow" in checkpoint
        and (cp_weight_source == "ema" or (cp_weight_source == "unknown" and not is_v393))
    )
    if use_ema:
        model_state = model.state_dict()
        matched = 0
        for k, v in checkpoint["ema_shadow"].items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k].copy_(v)
                matched += 1
        print(f"[{dataset}] deployed EMA shadow: matched={matched}", flush=True)

    epoch = int(checkpoint.get("deployment_epoch_override", checkpoint.get("epoch", -1))) if isinstance(checkpoint, dict) else -1
    if hasattr(model, "set_epoch"):
        model.set_epoch(epoch)
    model.eval().to(cfg.MODEL.DEVICE)

    transform = ValGenerator(output_size=[cfg.DATASET.SIZE, cfg.DATASET.SIZE])
    text_rows = read_text(str(Path(cfg.DATASET.TEXT_PROMPT_PATH) / "Val_text.xlsx"))
    valset = DatasetSegmentation(
        cfg.DATASET.VAL_PATH, cfg.DATASET.NAME, text_rows, transform,
        image_size=cfg.DATASET.SIZE,
    )
    return cfg, model, valset, epoch


def cpu_maps(m: Dict):
    out = {}
    for k, v in m.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu().clone()
        else:
            out[k] = v
    return out


def gt_from_batch(batch: Dict, device, hw: Tuple[int, int]):
    key = next((k for k in ("ground_truth_mask", "mask", "label", "labels") if k in batch), None)
    if key is None:
        raise KeyError(f"No GT key in batch. keys={list(batch.keys())}")
    gt = batch[key].to(device).float()
    if gt.ndim == 3:
        gt = gt[:, None]
    if gt.shape[1] != 1:
        gt = gt[:, :1]
    if tuple(gt.shape[-2:]) != tuple(hw):
        gt = F.interpolate(gt, size=hw, mode="nearest")
    return (gt > 0.5).float()


def dice_hard(prob: torch.Tensor, gt: torch.Tensor) -> float:
    if prob.ndim == 3:
        prob = prob[:, None]
    pred = (prob > 0.5).float()
    inter = (pred * gt).sum(dim=(1, 2, 3))
    den = pred.sum(dim=(1, 2, 3)) + gt.sum(dim=(1, 2, 3))
    both_empty = den == 0
    d = (2.0 * inter + 1e-6) / (den + 1e-6)
    d = torch.where(both_empty, torch.ones_like(d), d)
    return float(d[0].detach().cpu())


def surface_dice_proxy(prob: torch.Tensor, gt: torch.Tensor, tolerance: int = 2) -> float:
    if prob.ndim == 3:
        prob = prob[:, None]
    x = (prob > 0.5).float()
    y = gt.float()
    ex = 1.0 - F.max_pool2d(1.0 - x, 3, 1, 1)
    ey = 1.0 - F.max_pool2d(1.0 - y, 3, 1, 1)
    bx = (x - ex).clamp(0.0, 1.0)
    by = (y - ey).clamp(0.0, 1.0)
    r = max(0, int(tolerance))
    near_x = F.max_pool2d(bx, 2 * r + 1, 1, r)
    near_y = F.max_pool2d(by, 2 * r + 1, 1, r)
    close_x = (bx * near_y).sum(dim=(1, 2, 3))
    close_y = (by * near_x).sum(dim=(1, 2, 3))
    denom = bx.sum(dim=(1, 2, 3)) + by.sum(dim=(1, 2, 3))
    score = (close_x + close_y) / denom.clamp_min(1e-6)
    area_x = x.sum(dim=(1, 2, 3))
    area_y = y.sum(dim=(1, 2, 3))
    score = torch.where((area_x == 0) & (area_y == 0), torch.ones_like(score), score)
    score = torch.where((area_x == 0) ^ (area_y == 0), torch.zeros_like(score), score)
    return float(score[0].detach().cpu())


def case_metrics(maps: Dict, gt: torch.Tensor, tol: int):
    """Compute case-selection diagnostics on ONE device (CPU).

    The previous version mixed MC-mean CUDA tensors with representative-pass
    CPU maps, which caused ``changed & band`` to fail.  For qualitative case
    mining, CPU is cheap and makes the device contract explicit.
    """
    def _cpu(x):
        return x.detach().float().cpu() if torch.is_tensor(x) else x

    base_hr = _cpu(maps["base_hr"])
    final_hr = _cpu(maps["actual_final"])
    band = _cpu(maps["v15_band"]) > 0.5
    corr = _cpu(maps["projected_corr"])
    gt = _cpu(gt) > 0.5

    base_p = torch.sigmoid(base_hr)
    final_p = torch.sigmoid(final_hr)
    base_h = base_p > 0.5
    final_h = final_p > 0.5
    changed = base_h.ne(final_h)

    # Pixels that QABR actually repaired / damaged relative to GT.
    base_wrong = base_h.ne(gt)
    final_wrong = final_h.ne(gt)
    fixed = changed & base_wrong & (~final_wrong)
    harmed = changed & (~base_wrong) & final_wrong
    fixed_band = fixed & band
    harmed_band = harmed & band

    band_pixels = float(band.float().sum().clamp_min(1.0))
    changed_pixels = int(changed.sum())
    fixed_pixels = int(fixed.sum())
    harmed_pixels = int(harmed.sum())
    fixed_band_pixels = int(fixed_band.sum())
    harmed_band_pixels = int(harmed_band.sum())

    changed_band = float((changed & band).float().sum()) / band_pixels
    fixed_band_fraction = float(fixed_band.float().sum()) / band_pixels
    harmed_band_fraction = float(harmed_band.float().sum()) / band_pixels
    prob_shift_band = float((torch.abs(final_p - base_p) * band.float()).sum()) / band_pixels
    corr_rms_band = math.sqrt(float((corr.square() * band.float()).sum()) / band_pixels)
    correction_precision = fixed_pixels / max(changed_pixels, 1)
    net_fixed_pixels = fixed_pixels - harmed_pixels
    net_fixed_band_pixels = fixed_band_pixels - harmed_band_pixels

    db = dice_hard(base_p, gt.float())
    df = dice_hard(final_p, gt.float())
    nb = surface_dice_proxy(base_p, gt.float(), tolerance=tol)
    nf = surface_dice_proxy(final_p, gt.float(), tolerance=tol)
    dg = df - db
    ng = nf - nb
    return {
        "dice_base": db,
        "dice_final": df,
        "dice_gain": dg,
        "nsd_base": nb,
        "nsd_final": nf,
        "nsd_gain": ng,
        "changed_band_fraction": changed_band,
        "fixed_band_fraction": fixed_band_fraction,
        "harmed_band_fraction": harmed_band_fraction,
        "prob_shift_band": prob_shift_band,
        "corr_rms_band": corr_rms_band,
        "changed_pixels": changed_pixels,
        "fixed_pixels": fixed_pixels,
        "harmed_pixels": harmed_pixels,
        "net_fixed_pixels": net_fixed_pixels,
        "fixed_band_pixels": fixed_band_pixels,
        "harmed_band_pixels": harmed_band_pixels,
        "net_fixed_band_pixels": net_fixed_band_pixels,
        "correction_precision": correction_precision,
    }


def selection_tier(m: Dict) -> int:
    """Prefer cases that are both beneficial and visually interpretable."""
    dg, ng = m["dice_gain"], m["nsd_gain"]
    net = m.get("net_fixed_pixels", 0)
    prec = m.get("correction_precision", 0.0)
    # Tier 4: strong paper-quality example: both metrics improve and most changed
    # pixels move toward GT.
    if dg >= 0.003 and ng >= 0.010 and net > 0 and prec >= 0.60:
        return 4
    # Tier 3: non-harmful in both metrics with clear boundary gain and net repair.
    if dg >= 0.0 and ng >= 0.005 and net > 0 and prec >= 0.55:
        return 3
    # Tier 2: region/boundary gain is positive and edits are net beneficial.
    if dg >= 0.0 and ng >= 0.0 and net > 0:
        return 2
    # Tier 1: allow a tiny Dice tradeoff only for a strong boundary improvement.
    if ng >= 0.010 and dg >= -0.002 and net > 0:
        return 1
    return 0


def rank_score(m: Dict) -> float:
    """Rank by validity first, then visible *correct* movement.

    A large correction alone is not rewarded.  The visual terms only help after
    the edit is net beneficial w.r.t. GT.
    """
    tier = selection_tier(m)
    dg = max(m["dice_gain"], -0.01)
    ng = max(m["nsd_gain"], -0.02)
    net = max(m.get("net_fixed_pixels", 0), 0)
    prec = m.get("correction_precision", 0.0)
    visible = min(m["changed_band_fraction"] / 0.15, 1.0)
    shift = min(m["prob_shift_band"] / 0.05, 1.0)
    net_vis = min(net / 80.0, 1.0)
    return (
        tier * 1000.0
        + 180.0 * dg
        + 140.0 * ng
        + 18.0 * net_vis
        + 8.0 * prec
        + 6.0 * visible
        + 3.0 * shift
        + min(m["corr_rms_band"], 1.0)
    )


def one_forward(model, images, prompts, capture: Dict):
    capture.clear()
    tokenized = model.tokenizer(prompts).to(images.device)
    text_embeddings = model.text_model.transformer.embeddings.word_embeddings(tokenized).type(model.dtype)
    _ = model._forward_base_once(images, tokenized, text_embeddings)
    if "maps" not in capture:
        raise RuntimeError("QABR forward hook did not capture maps. Is MODEL.QABR.ENABLED=true?")
    return capture["maps"]


def evaluate_case_mc(model, sample: Dict, vis, tol: int, mc: int, seed: int):
    batch = default_collate([sample])
    images = batch["image"].to(next(model.parameters()).device)
    prompts = batch["text_prompt"]
    capture = {}

    def hook(mod, args, kwargs, result):
        with torch.no_grad():
            capture["maps"] = vis._extract_maps(mod, args, kwargs, result)

    handle = model.qabr.register_forward_hook(hook, with_kwargs=True)
    try:
        set_seed(seed)
        base_ps, final_ps = [], []
        rep_maps = None
        rep_strength = -1.0
        gt = None
        for _ in range(max(1, mc)):
            with torch.no_grad():
                maps = one_forward(model, images, prompts, capture)
            if gt is None:
                gt = gt_from_batch(batch, images.device, tuple(maps["base_hr"].shape[-2:]))
            bp = torch.sigmoid(maps["base_hr"])
            fp = torch.sigmoid(maps["actual_final"])
            base_ps.append(bp)
            final_ps.append(fp)
            band = maps["v15_band"] > 0.5
            denom = band.float().sum().clamp_min(1.0)
            strength = float((maps["projected_corr"].abs() * band.float()).sum().detach().cpu() / denom.detach().cpu())
            if strength > rep_strength:
                rep_strength = strength
                rep_maps = cpu_maps(maps)
        base_mean = torch.stack(base_ps, dim=0).mean(dim=0)
        final_mean = torch.stack(final_ps, dim=0).mean(dim=0)
        # Use the representative pass for geometric visibility diagnostics, but
        # overwrite quality metrics with MC-mean predictions.
        metrics = case_metrics({
            **rep_maps,
            "base_hr": torch.logit(base_mean.clamp(1e-6, 1 - 1e-6)).cpu(),
            "actual_final": torch.logit(final_mean.clamp(1e-6, 1 - 1e-6)).cpu(),
        }, gt.cpu(), tol)
        metrics["mc"] = max(1, mc)
        return metrics, rep_maps, gt.detach().cpu(), base_mean.detach().cpu(), final_mean.detach().cpu()
    finally:
        handle.remove()


def mask_name_of(sample: Dict, index: int) -> str:
    value = sample.get("mask_name", f"case_{index:04d}.png")
    if isinstance(value, (list, tuple)):
        value = value[0]
    return str(value)


def save_gt_overlay(path: Path, maps: Dict, gt: torch.Tensor, title: str):
    img = vis_display_image(maps)
    base = torch.sigmoid(maps["base_hr"]).detach().cpu().numpy()[0, 0]
    final = torch.sigmoid(maps["actual_final"]).detach().cpu().numpy()[0, 0]
    g = gt.detach().cpu().numpy()[0, 0]
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    try:
        ax.contour(g, levels=[0.5], colors=["#39ff14"], linewidths=2.0)
        ax.contour(base, levels=[0.5], colors=["white"], linewidths=1.5, linestyles="--")
        ax.contour(final, levels=[0.5], colors=["red"], linewidths=1.7)
    except Exception:
        pass
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def vis_display_image(maps: Dict):
    image = maps.get("image")
    hw = tuple(maps["base_hr"].shape[-2:])
    x = image.detach().float()
    if tuple(x.shape[-2:]) != hw:
        x = F.interpolate(x, size=hw, mode="bilinear", align_corners=False)
    x = x[0].mean(dim=0).cpu().numpy()
    lo, hi = np.percentile(x, [1.0, 99.0])
    if hi - lo < 1e-8:
        lo, hi = float(x.min()), float(x.max())
    return np.clip((x - lo) / max(hi - lo, 1e-8), 0, 1)


def save_selection_summary(path: Path, maps: Dict, gt: torch.Tensor, base_mean: torch.Tensor,
                           final_mean: torch.Tensor, metrics: Dict, dataset: str, mask_name: str):
    img = vis_display_image(maps)
    g = gt.numpy()[0, 0]
    bp = base_mean.numpy()[0, 0]
    fp = final_mean.numpy()[0, 0]
    u = (4 * bp * (1 - bp)).clip(0, 1)
    band = maps["v15_band"].numpy()[0, 0]
    sn = maps["signed_normal"].numpy()[0, 0]
    corr = maps["projected_corr"].numpy()[0, 0]

    fig, axes = plt.subplots(2, 4, figsize=(14.5, 7.2))
    axes = axes.ravel()
    panels = [
        ("Input", img, "gray", None),
        ("GT", g, "gray", (0, 1)),
        ("Base probability", bp, "magma", (0, 1)),
        ("QABR probability", fp, "magma", (0, 1)),
        ("Uncertainty 4p(1-p)", u, "magma", (0, 1)),
        ("Editable band B", band, "gray", (0, 1)),
        ("Signed normal evidence", sn, "coolwarm", (-1, 1)),
        ("Projected correction", corr, "coolwarm", None),
    ]
    for ax, (title, arr, cmap, lim) in zip(axes, panels):
        kwargs = {"cmap": cmap}
        if lim is not None:
            kwargs.update(vmin=lim[0], vmax=lim[1])
        elif cmap == "coolwarm":
            vmax = max(float(np.max(np.abs(arr))), 1e-6)
            kwargs.update(vmin=-vmax, vmax=vmax)
        ax.imshow(arr, **kwargs)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(
        f"{dataset} | {mask_name}\n"
        f"Dice {metrics['dice_base']:.4f}→{metrics['dice_final']:.4f} ({metrics['dice_gain']:+.4f}), "
        f"Surface {metrics['nsd_base']:.4f}→{metrics['nsd_final']:.4f} ({metrics['nsd_gain']:+.4f}), "
        f"changed-band={metrics['changed_band_fraction']:.3f}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)




def _np2(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    while x.ndim > 2:
        x = x[0]
    return x


def _sym_lim(arr, percentile=99.2):
    a = np.abs(np.asarray(arr, dtype=np.float32))
    v = float(np.percentile(a, percentile)) if a.size else 1.0
    return max(v, 1e-6)


def _contour(ax, arr, color, lw=2.0, ls='-'):
    try:
        ax.contour(arr, levels=[0.5], colors=[color], linewidths=lw, linestyles=ls)
    except Exception:
        pass


def _crop_box(gt, base, final, band, pad=18):
    mask = (gt > 0.5) | (base > 0.5) | (final > 0.5) | (band > 0.5)
    ys, xs = np.where(mask)
    H, W = mask.shape
    if len(xs) == 0:
        return 0, H, 0, W
    y0, y1 = max(0, int(ys.min()) - pad), min(H, int(ys.max()) + pad + 1)
    x0, x1 = max(0, int(xs.min()) - pad), min(W, int(xs.max()) + pad + 1)
    # Ensure a readable minimum crop.
    mh, mw = 72, 72
    if y1 - y0 < mh:
        c = (y0 + y1) // 2; y0=max(0,c-mh//2); y1=min(H,y0+mh); y0=max(0,y1-mh)
    if x1 - x0 < mw:
        c = (x0 + x1) // 2; x0=max(0,c-mw//2); x1=min(W,x0+mw); x0=max(0,x1-mw)
    return y0, y1, x0, x1


def save_vivid_framework(path: Path, maps: Dict, gt: torch.Tensor, base_mean: torch.Tensor,
                         final_mean: torch.Tensor, metrics: Dict, dataset: str, mask_name: str):
    """Publication-style evidence-chain figure using *actual* QABR tensors."""
    img = vis_display_image(maps)
    g = _np2(gt)
    bp = _np2(base_mean)
    fp = _np2(final_mean)
    z0 = _np2(maps['base_hr'])
    u = np.clip(4.0 * bp * (1.0 - bp), 0, 1)
    hard = _np2(maps['hard'])
    hb = _np2(maps['hard_boundary'])
    band = _np2(maps['v15_band'])
    eI = _np2(maps['image_edge'])
    sn = _np2(maps['signed_normal'])
    D = _np2(maps['detail_rms'])
    h3 = _np2(maps['hp3'])
    h5 = _np2(maps['hp5'])
    cand = _np2(maps['candidate_delta'])
    support = _np2(maps['support'])
    raw = _np2(maps['raw_corr'])
    w = _np2(maps['tangent_weight'])
    proj = _np2(maps['projected_corr'])
    nx = _np2(maps['mask_nx']); ny = _np2(maps['mask_ny'])

    fig, axes = plt.subplots(3, 6, figsize=(21.5, 11.2), constrained_layout=True)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])

    # ---------- Row 1: semantic anchor / where ----------
    ax=axes[0,0]; ax.imshow(img,cmap='gray',vmin=0,vmax=1); _contour(ax,g,'#39ff14',2.2); ax.set_title('Input + GT',fontsize=11,fontweight='bold')
    ax=axes[0,1]; v=_sym_lim(z0); im=ax.imshow(z0,cmap='turbo',vmin=-v,vmax=v); ax.set_title('Semantic logit  $z_0$',fontsize=11,fontweight='bold'); fig.colorbar(im,ax=ax,fraction=.046,pad=.02)
    ax=axes[0,2]; im=ax.imshow(bp,cmap='turbo',vmin=0,vmax=1); ax.set_title('Foreground probability  $p$',fontsize=11,fontweight='bold'); fig.colorbar(im,ax=ax,fraction=.046,pad=.02)
    ax=axes[0,3]; im=ax.imshow(u,cmap='inferno',vmin=0,vmax=1); _contour(ax,bp,'#00e5ff',1.0); ax.set_title('Uncertainty  $u=4p(1-p)$',fontsize=11,fontweight='bold'); fig.colorbar(im,ax=ax,fraction=.046,pad=.02)
    ax=axes[0,4]; ax.imshow(img,cmap='gray',vmin=0,vmax=1); ax.imshow(np.ma.masked_where(band<=.5,band),cmap='spring',vmin=0,vmax=1,alpha=.78); _contour(ax,bp,'white',1.1); ax.set_title('Editable boundary band  $B$',fontsize=11,fontweight='bold')
    ax=axes[0,5]; ax.imshow(img,cmap='gray',vmin=0,vmax=1); _contour(ax,g,'#39ff14',2.1); _contour(ax,bp,'#00e5ff',1.8,'--'); ax.set_title('Query-anchored support region',fontsize=11,fontweight='bold')

    # ---------- Row 2: directional / semantic evidence ----------
    ax=axes[1,0]; ax.imshow(img,cmap='gray',vmin=0,vmax=1); _contour(ax,bp,'#00e5ff',1.3)
    yy,xx=np.where(hb>.5)
    if len(xx)>0:
        step=max(1,len(xx)//42); xx=xx[::step]; yy=yy[::step]
        ax.quiver(xx,yy,nx[yy,xx],ny[yy,xx],color='#ff1744',angles='xy',scale_units='xy',scale=.14,width=.006,headwidth=4.5,headlength=5.5)
    ax.set_title('Contour normals  $\\mathbf{n}$',fontsize=11,fontweight='bold')
    ax=axes[1,1]; im=ax.imshow(eI,cmap='inferno',vmin=0,vmax=1); ax.set_title('Image edge  $e_I$',fontsize=11,fontweight='bold'); fig.colorbar(im,ax=ax,fraction=.046,pad=.02)
    ax=axes[1,2]; im=ax.imshow(sn,cmap='seismic',vmin=-1,vmax=1); _contour(ax,bp,'#ffe600',1.0); ax.set_title('Signed normal evidence  $s_n$',fontsize=11,fontweight='bold'); fig.colorbar(im,ax=ax,fraction=.046,pad=.02)
    ax=axes[1,3]; im=ax.imshow(D,cmap='plasma',vmin=0,vmax=1); ax.set_title('Semantic detail  $D$',fontsize=11,fontweight='bold'); fig.colorbar(im,ax=ax,fraction=.046,pad=.02)
    ax=axes[1,4]; v=_sym_lim(h3); im=ax.imshow(h3,cmap='seismic',vmin=-v,vmax=v); ax.set_title('Query anchor high-pass  $h_3$',fontsize=11,fontweight='bold'); fig.colorbar(im,ax=ax,fraction=.046,pad=.02)
    ax=axes[1,5]; v=_sym_lim(h5); im=ax.imshow(h5,cmap='seismic',vmin=-v,vmax=v); ax.set_title('Query anchor high-pass  $h_5$',fontsize=11,fontweight='bold'); fig.colorbar(im,ax=ax,fraction=.046,pad=.02)

    # ---------- Row 3: fusion / correction / projection ----------
    ax=axes[2,0]; v=_sym_lim(cand); im=ax.imshow(cand,cmap='seismic',vmin=-v,vmax=v); ax.set_title('Bounded candidate  $\\delta$',fontsize=11,fontweight='bold'); fig.colorbar(im,ax=ax,fraction=.046,pad=.02)
    ax=axes[2,1]; im=ax.imshow(support,cmap='inferno',vmin=0,vmax=1); ax.set_title('Adaptive support  $S$',fontsize=11,fontweight='bold'); fig.colorbar(im,ax=ax,fraction=.046,pad=.02)
    ax=axes[2,2]; v=_sym_lim(raw); im=ax.imshow(raw,cmap='seismic',vmin=-v,vmax=v); ax.set_title('Raw correction  $d$',fontsize=11,fontweight='bold'); fig.colorbar(im,ax=ax,fraction=.046,pad=.02)
    ax=axes[2,3]; im=ax.imshow(w,cmap='plasma',vmin=0,vmax=max(float(w.max()),1e-6)); ax.set_title('Tangent weight  $Bp(1-p)$',fontsize=11,fontweight='bold'); fig.colorbar(im,ax=ax,fraction=.046,pad=.02)
    ax=axes[2,4]; v=_sym_lim(proj); im=ax.imshow(proj,cmap='seismic',vmin=-v,vmax=v); ax.set_title('Projected correction  $\\hat d$',fontsize=11,fontweight='bold'); fig.colorbar(im,ax=ax,fraction=.046,pad=.02)
    ax=axes[2,5]; ax.imshow(img,cmap='gray',vmin=0,vmax=1); _contour(ax,g,'#39ff14',2.3); _contour(ax,bp,'#00e5ff',1.8,'--'); _contour(ax,fp,'#ff1744',2.0); ax.set_title('GT / Base / QABR',fontsize=11,fontweight='bold')
    ax.text(.02,.02,'GT',color='#39ff14',transform=ax.transAxes,fontsize=9,fontweight='bold',bbox=dict(facecolor='black',alpha=.55,pad=2,edgecolor='none'))
    ax.text(.19,.02,'Base',color='#00e5ff',transform=ax.transAxes,fontsize=9,fontweight='bold',bbox=dict(facecolor='black',alpha=.55,pad=2,edgecolor='none'))
    ax.text(.42,.02,'QABR',color='#ff1744',transform=ax.transAxes,fontsize=9,fontweight='bold',bbox=dict(facecolor='black',alpha=.55,pad=2,edgecolor='none'))

    # Row labels mimic the colorful paper framework.
    fig.text(.012,.785,'① Semantic Anchor + Where',color='#0d47a1',fontsize=13,fontweight='bold',rotation=90,va='center')
    fig.text(.012,.505,'② Direction + What',color='#ef6c00',fontsize=13,fontweight='bold',rotation=90,va='center')
    fig.text(.012,.215,'③ Fusion → Tangent Projection',color='#7b1fa2',fontsize=13,fontweight='bold',rotation=90,va='center')

    title=(f'QABR real intermediate evidence | {dataset} | {mask_name}\\n'
           f'Dice {metrics["dice_base"]:.4f}→{metrics["dice_final"]:.4f} ({metrics["dice_gain"]:+.4f})   '
           f'Surface {metrics["nsd_base"]:.4f}→{metrics["nsd_final"]:.4f} ({metrics["nsd_gain"]:+.4f})   '
           f'fixed/harmed={metrics.get("fixed_pixels",0)}/{metrics.get("harmed_pixels",0)}   '
           f'changed-band={metrics["changed_band_fraction"]:.3f}')
    fig.suptitle(title,fontsize=15,fontweight='bold')
    fig.savefig(path,dpi=260,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def save_zoom_evidence(path: Path, maps: Dict, gt: torch.Tensor, base_mean: torch.Tensor,
                       final_mean: torch.Tensor, metrics: Dict, dataset: str, mask_name: str):
    img=vis_display_image(maps); g=_np2(gt); bp=_np2(base_mean); fp=_np2(final_mean)
    band=_np2(maps['v15_band']); sn=_np2(maps['signed_normal']); cand=_np2(maps['candidate_delta'])
    raw=_np2(maps['raw_corr']); proj=_np2(maps['projected_corr']); support=_np2(maps['support'])
    u=np.clip(4*bp*(1-bp),0,1)
    y0,y1,x0,x1=_crop_box(g,bp,fp,band,pad=14)
    def C(a): return a[y0:y1,x0:x1]
    fig,axes=plt.subplots(2,4,figsize=(15.5,7.7),constrained_layout=True)
    axes=axes.ravel()
    # 0 overlay
    ax=axes[0]; ax.imshow(C(img),cmap='gray',vmin=0,vmax=1); _contour(ax,C(g),'#39ff14',2.4); _contour(ax,C(bp),'#00e5ff',1.9,'--'); _contour(ax,C(fp),'#ff1744',2.2); ax.set_title('Boundary relocation (zoom)',fontweight='bold')
    # 1 uncertainty
    im=axes[1].imshow(C(u),cmap='inferno',vmin=0,vmax=1); _contour(axes[1],C(bp),'#00e5ff',1.0); axes[1].set_title('Uncertainty $u$',fontweight='bold'); fig.colorbar(im,ax=axes[1],fraction=.046,pad=.02)
    # 2 band
    axes[2].imshow(C(img),cmap='gray',vmin=0,vmax=1); axes[2].imshow(np.ma.masked_where(C(band)<=.5,C(band)),cmap='spring',vmin=0,vmax=1,alpha=.82); axes[2].set_title('Editable band $B$',fontweight='bold')
    # 3 signed normal
    im=axes[3].imshow(C(sn),cmap='seismic',vmin=-1,vmax=1); _contour(axes[3],C(bp),'#ffe600',1.0); axes[3].set_title('Signed normal $s_n$',fontweight='bold'); fig.colorbar(im,ax=axes[3],fraction=.046,pad=.02)
    # 4 support
    im=axes[4].imshow(C(support),cmap='inferno',vmin=0,vmax=1); axes[4].set_title('Support $S$',fontweight='bold'); fig.colorbar(im,ax=axes[4],fraction=.046,pad=.02)
    # 5 candidate
    v=_sym_lim(C(cand)); im=axes[5].imshow(C(cand),cmap='seismic',vmin=-v,vmax=v); axes[5].set_title('Candidate $\\delta$',fontweight='bold'); fig.colorbar(im,ax=axes[5],fraction=.046,pad=.02)
    # 6 raw
    v=_sym_lim(C(raw)); im=axes[6].imshow(C(raw),cmap='seismic',vmin=-v,vmax=v); axes[6].set_title('Before projection $d$',fontweight='bold'); fig.colorbar(im,ax=axes[6],fraction=.046,pad=.02)
    # 7 projected
    v=_sym_lim(C(proj)); im=axes[7].imshow(C(proj),cmap='seismic',vmin=-v,vmax=v); axes[7].set_title('After tangent projection $\\hat d$',fontweight='bold'); fig.colorbar(im,ax=axes[7],fraction=.046,pad=.02)
    for ax in axes: ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f'{dataset} | {mask_name} | local QABR evidence chain | '
                 f'ΔDice={metrics["dice_gain"]:+.4f}, ΔSurface={metrics["nsd_gain"]:+.4f}',fontsize=13,fontweight='bold')
    fig.savefig(path,dpi=280,bbox_inches='tight',facecolor='white'); plt.close(fig)


def save_fixed_harmed(path: Path, maps: Dict, gt: torch.Tensor, base_mean: torch.Tensor,
                      final_mean: torch.Tensor, metrics: Dict, dataset: str, mask_name: str):
    img=vis_display_image(maps); g=_np2(gt)>0.5; bp=_np2(base_mean)>0.5; fp=_np2(final_mean)>0.5
    changed=bp^fp; fixed=changed & (bp!=g) & (fp==g); harmed=changed & (bp==g) & (fp!=g)
    overlay=np.zeros((*g.shape,4),dtype=np.float32)
    overlay[fixed]=[0.10,1.00,0.15,0.90]   # neon green
    overlay[harmed]=[1.00,0.05,0.15,0.92] # red
    fig,ax=plt.subplots(figsize=(6.6,6.2)); ax.imshow(img,cmap='gray',vmin=0,vmax=1); ax.imshow(overlay)
    _contour(ax,g.astype(float),'#ffe600',1.6); _contour(ax,bp.astype(float),'#00e5ff',1.3,'--'); _contour(ax,fp.astype(float),'white',1.3)
    ax.set_title(f'{dataset} | corrected pixels (green) vs harmed pixels (red)\\n'
                 f'fixed={metrics.get("fixed_pixels",0)}, harmed={metrics.get("harmed_pixels",0)}, '
                 f'precision={metrics.get("correction_precision",0):.2%}',fontsize=11,fontweight='bold')
    ax.axis('off'); fig.tight_layout(); fig.savefig(path,dpi=280,bbox_inches='tight',facecolor='white'); plt.close(fig)

def write_csv(path: Path, rows: List[Dict]):
    if not rows:
        return
    keys = [
        "dataset", "index", "mask_name", "tier", "rank_score", "mc",
        "dice_base", "dice_final", "dice_gain",
        "nsd_base", "nsd_final", "nsd_gain",
        "changed_band_fraction", "fixed_band_fraction", "harmed_band_fraction",
        "prob_shift_band", "corr_rms_band", "changed_pixels",
        "fixed_pixels", "harmed_pixels", "net_fixed_pixels",
        "fixed_band_pixels", "harmed_band_pixels", "net_fixed_band_pixels",
        "correction_precision", "gt_pixels",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in keys})


def scan_dataset(args, project: Path, data_dir: Path, out_root: Path, dataset: str, vis):
    run_dir, ckpt = resolve_checkpoint(project, args.study_id, dataset)
    print("=" * 90)
    print(f"[SCAN] {dataset}")
    print(f"run_dir={run_dir}")
    print(f"checkpoint={ckpt}")
    cfg, model, valset, epoch = load_model_and_dataset(project, data_dir, dataset, ckpt, args.seed)
    if not getattr(model, "qabr_enabled", False) or getattr(model, "qabr", None) is None:
        raise RuntimeError(f"{dataset}: QABR is not enabled in the selected config")
    print(f"[{dataset}] val cases={len(valset)} | deployment_epoch={epoch}")

    ds_out = out_root / dataset
    ds_out.mkdir(parents=True, exist_ok=True)

    capture = {}
    def hook(mod, f_args, f_kwargs, result):
        with torch.no_grad():
            capture["maps"] = vis._extract_maps(mod, f_args, f_kwargs, result)
    hook_handle = model.qabr.register_forward_hook(hook, with_kwargs=True)

    fast_rows = []
    fast_candidates = []
    loader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=0)
    max_cases = args.max_cases_per_dataset if args.max_cases_per_dataset > 0 else len(valset)
    set_seed(args.seed)
    try:
        for idx, batch in enumerate(loader):
            if idx >= max_cases:
                break
            images = batch["image"].to(cfg.MODEL.DEVICE)
            gt_pixels = None
            # Skip empty-GT cases for qualitative boundary visualization.
            gt_probe = gt_from_batch(batch, images.device, (cfg.DATASET.SIZE, cfg.DATASET.SIZE))
            gt_pixels = int(gt_probe.sum().detach().cpu())
            if gt_pixels == 0:
                continue

            base_list, final_list = [], []
            rep_maps = None
            rep_strength = -1.0
            for _ in range(max(1, args.scan_mc)):
                with torch.no_grad():
                    maps = one_forward(model, images, batch["text_prompt"], capture)
                gt = gt_from_batch(batch, images.device, tuple(maps["base_hr"].shape[-2:]))
                bp = torch.sigmoid(maps["base_hr"])
                fp = torch.sigmoid(maps["actual_final"])
                base_list.append(bp)
                final_list.append(fp)
                band = maps["v15_band"] > 0.5
                denom = band.float().sum().clamp_min(1.0)
                strength = float((maps["projected_corr"].abs() * band.float()).sum().detach().cpu() / denom.detach().cpu())
                if strength > rep_strength:
                    rep_strength = strength
                    rep_maps = cpu_maps(maps)
            base_mean = torch.stack(base_list).mean(0)
            final_mean = torch.stack(final_list).mean(0)
            metric_maps = {
                **rep_maps,
                "base_hr": torch.logit(base_mean.clamp(1e-6, 1 - 1e-6)).cpu(),
                "actual_final": torch.logit(final_mean.clamp(1e-6, 1 - 1e-6)).cpu(),
            }
            m = case_metrics(metric_maps, gt.cpu(), args.nsd_tolerance)
            m.update({
                "dataset": dataset,
                "index": idx,
                "mask_name": str(batch["mask_name"][0]),
                "gt_pixels": gt_pixels,
                "mc": max(1, args.scan_mc),
            })
            m["tier"] = selection_tier(m)
            m["rank_score"] = rank_score(m)
            fast_rows.append(m)
            fast_candidates.append(m)
            if (idx + 1) % 50 == 0:
                print(
                    f"[{dataset}] {idx+1}/{min(len(valset), max_cases)} | "
                    f"current ΔDice={m['dice_gain']:+.4f} ΔSurf={m['nsd_gain']:+.4f} "
                    f"band-change={m['changed_band_fraction']:.3f}", flush=True
                )
    finally:
        hook_handle.remove()

    fast_rows.sort(key=lambda r: r["rank_score"], reverse=True)
    write_csv(ds_out / "all_val_cases_fast_scan.csv", fast_rows)
    shortlist = fast_rows[: max(1, args.preselect_k)]
    print(f"[{dataset}] fast shortlist:")
    for r in shortlist:
        print(
            f"  idx={r['index']:4d} {r['mask_name']} tier={r['tier']} "
            f"ΔDice={r['dice_gain']:+.4f} ΔSurf={r['nsd_gain']:+.4f} "
            f"changedBand={r['changed_band_fraction']:.3f} score={r['rank_score']:.3f}"
        )

    verified = []
    for order, r in enumerate(shortlist):
        idx = int(r["index"])
        sample = valset[idx]
        verify_seed = args.seed * 100000 + idx
        metrics, maps, gt, base_mean, final_mean = evaluate_case_mc(
            model, sample, vis, args.nsd_tolerance, args.verify_mc, verify_seed
        )
        row = {
            **metrics,
            "dataset": dataset,
            "index": idx,
            "mask_name": mask_name_of(sample, idx),
            "gt_pixels": int(gt.sum().item()),
        }
        row["tier"] = selection_tier(row)
        row["rank_score"] = rank_score(row)
        row["_maps"] = maps
        row["_gt"] = gt
        row["_base_mean"] = base_mean
        row["_final_mean"] = final_mean
        verified.append(row)
        print(
            f"[{dataset} VERIFY] idx={idx:4d} {row['mask_name']} MC={args.verify_mc} "
            f"tier={row['tier']} ΔDice={row['dice_gain']:+.4f} ΔSurf={row['nsd_gain']:+.4f} "
            f"changedBand={row['changed_band_fraction']:.3f} score={row['rank_score']:.3f}", flush=True
        )

    verified.sort(key=lambda r: r["rank_score"], reverse=True)
    write_csv(ds_out / "verified_candidates.csv", verified)

    exported = []
    for rank, row in enumerate(verified[: max(1, args.export_top_k)], start=1):
        case_dir = ds_out / f"rank{rank:02d}_idx{int(row['index']):04d}_{Path(row['mask_name']).stem}"
        if case_dir.exists():
            shutil.rmtree(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)
        vis._export_case(row["_maps"], case_dir)
        # Save MC-mean base/final arrays too.
        np.savez_compressed(
            case_dir / "mc_mean_predictions.npz",
            base_p=row["_base_mean"].numpy(),
            final_p=row["_final_mean"].numpy(),
            gt=row["_gt"].numpy(),
        )
        save_gt_overlay(
            case_dir / "22_gt_coarse_refined_overlay.png", row["_maps"], row["_gt"],
            f"{dataset} | GT=green, Base=white dashed, QABR=red"
        )
        save_selection_summary(
            case_dir / "23_selected_case_summary.png", row["_maps"], row["_gt"],
            row["_base_mean"], row["_final_mean"], row, dataset, row["mask_name"]
        )
        save_vivid_framework(
            case_dir / "24_QABR_framework_evidence_chain_vivid.png", row["_maps"], row["_gt"],
            row["_base_mean"], row["_final_mean"], row, dataset, row["mask_name"]
        )
        save_zoom_evidence(
            case_dir / "25_QABR_local_zoom_evidence.png", row["_maps"], row["_gt"],
            row["_base_mean"], row["_final_mean"], row, dataset, row["mask_name"]
        )
        save_fixed_harmed(
            case_dir / "26_QABR_fixed_vs_harmed_pixels.png", row["_maps"], row["_gt"],
            row["_base_mean"], row["_final_mean"], row, dataset, row["mask_name"]
        )
        with (case_dir / "selection_metrics.json").open("w", encoding="utf-8") as f:
            clean = {k: v for k, v in row.items() if not k.startswith("_")}
            json.dump(clean, f, indent=2, ensure_ascii=False)
        row["_case_dir"] = str(case_dir)
        exported.append(row)

    # Free GPU before the next dataset.
    del model
    torch.cuda.empty_cache()
    return exported, fast_rows


def main():
    args = parse_args()
    if args.scan_mc < 1 or args.verify_mc < 1:
        raise ValueError("--scan-mc and --verify-mc must be >= 1")
    project = Path(args.project).resolve()
    data_dir = Path(args.data_dir).resolve()
    if not project.is_dir():
        raise FileNotFoundError(project)
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)
    sys.path.insert(0, str(project))

    out_root = Path(args.output_dir).resolve() if args.output_dir else (
        project / "qabr_case_search" / args.study_id
    )
    out_root.mkdir(parents=True, exist_ok=True)
    vis = load_visual_module(project)

    all_exported = []
    all_fast = []
    for ds in args.datasets:
        exported, fast = scan_dataset(args, project, data_dir, out_root, ds, vis)
        all_exported.extend(exported)
        all_fast.extend(fast)

    all_fast.sort(key=lambda r: r["rank_score"], reverse=True)
    write_csv(out_root / "ALL_DATASETS_fast_ranking.csv", all_fast)

    if not all_exported:
        raise RuntimeError("No non-empty validation candidate was exported.")
    all_exported.sort(key=lambda r: r["rank_score"], reverse=True)
    write_csv(out_root / "ALL_DATASETS_verified_top_candidates.csv", all_exported)
    winner = all_exported[0]
    winner_src = Path(winner["_case_dir"])
    winner_dst = out_root / "BEST_OVERALL"
    if winner_dst.exists():
        shutil.rmtree(winner_dst)
    shutil.copytree(winner_src, winner_dst)
    with (winner_dst / "WINNER.txt").open("w", encoding="utf-8") as f:
        f.write(
            f"dataset={winner['dataset']}\n"
            f"index={winner['index']}\n"
            f"mask_name={winner['mask_name']}\n"
            f"tier={winner['tier']}\n"
            f"rank_score={winner['rank_score']:.8f}\n"
            f"dice={winner['dice_base']:.6f}->{winner['dice_final']:.6f} ({winner['dice_gain']:+.6f})\n"
            f"surface_proxy={winner['nsd_base']:.6f}->{winner['nsd_final']:.6f} ({winner['nsd_gain']:+.6f})\n"
            f"changed_band_fraction={winner['changed_band_fraction']:.6f}\n"
            f"prob_shift_band={winner['prob_shift_band']:.6f}\n"
            f"corr_rms_band={winner['corr_rms_band']:.6f}\n"
            f"fixed_pixels={winner.get('fixed_pixels',0)}\n"
            f"harmed_pixels={winner.get('harmed_pixels',0)}\n"
            f"net_fixed_pixels={winner.get('net_fixed_pixels',0)}\n"
            f"correction_precision={winner.get('correction_precision',0):.6f}\n"
        )

    print("\n" + "=" * 90)
    print("BEST OVERALL VALIDATION CASE")
    print("=" * 90)
    print(f"dataset   : {winner['dataset']}")
    print(f"index     : {winner['index']}")
    print(f"mask_name : {winner['mask_name']}")
    print(f"tier      : {winner['tier']}")
    print(f"Dice      : {winner['dice_base']:.4f} -> {winner['dice_final']:.4f} ({winner['dice_gain']:+.4f})")
    print(f"Surface   : {winner['nsd_base']:.4f} -> {winner['nsd_final']:.4f} ({winner['nsd_gain']:+.4f})")
    print(f"changed B : {winner['changed_band_fraction']:.4f}")
    print(f"OUTPUT    : {winner_dst}")
    print("\nOpen these first:")
    print(winner_dst / "20_coarse_vs_refined_overlay.png")
    print(winner_dst / "21_QABR_pipeline_contact_sheet.png")
    print(winner_dst / "22_gt_coarse_refined_overlay.png")
    print(winner_dst / "23_selected_case_summary.png")
    print(winner_dst / "24_QABR_framework_evidence_chain_vivid.png")
    print(winner_dst / "25_QABR_local_zoom_evidence.png")
    print(winner_dst / "26_QABR_fixed_vs_harmed_pixels.png")
    print("=" * 90)


if __name__ == "__main__":
    main()
