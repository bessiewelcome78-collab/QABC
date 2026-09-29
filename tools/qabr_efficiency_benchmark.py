#!/usr/bin/env python3
"""Controlled efficiency benchmark for QABR vs its matched host.

Measures model parameter counts, batch-1 model-forward latency, CUDA peak
allocated memory, and the *incremental convolutional compute* inside QABR.
The latency benchmark uses a synthetic 224x224 image and a fixed text prompt,
so disk I/O and data-loader variability are excluded.  Both MC1 and the paper's
MC30 inference protocol are timed.

The QABR Conv FLOP count intentionally covers only Conv2d/Linear operations
inside the QABR module.  Parameter-free interpolation/pooling/elementwise
geometry is not folded into this number, so the table must label it as
"extra QABR conv FLOPs" rather than total-model FLOPs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from utils.main_utils import load_cfg_from_cfg_file
from trainers.medclipseg_unimedclip import build_medclipseg_unimedclip


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config-file", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--variant", choices=["BASE", "FULL"], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--prompt", default="Segment the polyp.")
    p.add_argument("--warmup-mc1", type=int, default=8)
    p.add_argument("--repeat-mc1", type=int, default=30)
    p.add_argument("--warmup-mc30", type=int, default=2)
    p.add_argument("--repeat-mc30", type=int, default=8)
    p.add_argument("opts", nargs=argparse.REMAINDER)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Match the paper's no-TF32 setup; cudnn benchmark is useful for stable
    # inference timing once shapes are fixed.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True


def numel(module: nn.Module) -> int:
    return sum(int(p.numel()) for p in module.parameters())


def trainable_numel(module: nn.Module) -> int:
    return sum(int(p.numel()) for p in module.parameters() if p.requires_grad)


def load_checkpoint(model: nn.Module, checkpoint_path: str) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw_state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    incompatible = model.load_state_dict(dict(raw_state), strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint/model mismatch in efficiency benchmark. "
            f"missing={missing[:30]} unexpected={unexpected[:30]}"
        )
    checkpoint_epoch = int(checkpoint.get("epoch", -1)) if isinstance(checkpoint, dict) else -1
    deployment_epoch = (
        int(checkpoint.get("deployment_epoch_override", checkpoint_epoch))
        if isinstance(checkpoint, dict) else checkpoint_epoch
    )
    if hasattr(model, "set_epoch"):
        model.set_epoch(deployment_epoch)
    return {
        "checkpoint_epoch": checkpoint_epoch,
        "deployment_epoch": deployment_epoch,
        "checkpoint_role": str(checkpoint.get("checkpoint_role", "")) if isinstance(checkpoint, dict) else "",
        "weight_source": str(checkpoint.get("weight_source", "raw")) if isinstance(checkpoint, dict) else "raw",
    }


def _first_tensor(x: Any) -> torch.Tensor | None:
    if torch.is_tensor(x):
        return x
    if isinstance(x, (tuple, list)):
        for item in x:
            t = _first_tensor(item)
            if t is not None:
                return t
    if isinstance(x, dict):
        for item in x.values():
            t = _first_tensor(item)
            if t is not None:
                return t
    return None


def qabr_conv_compute(model: nn.Module, image: torch.Tensor, prompt: str) -> Dict[str, float]:
    """Count learnable Conv2d/Linear MACs actually executed inside QABR for MC1."""
    qabr = getattr(model, "qabr", None)
    if qabr is None:
        return {
            "qabr_conv_macs_per_mc": 0.0,
            "qabr_conv_flops_per_mc": 0.0,
            "qabr_conv_gmacs_per_mc": 0.0,
            "qabr_conv_gflops_per_mc": 0.0,
        }

    counter = {"macs": 0}
    handles = []

    def conv_hook(module: nn.Conv2d, inputs: Tuple[Any, ...], output: Any):
        out = _first_tensor(output)
        if out is None:
            return
        # one output element performs kH*kW*(Cin/groups) multiply-accumulates
        kernel_mul = int(module.kernel_size[0]) * int(module.kernel_size[1]) * (int(module.in_channels) // int(module.groups))
        counter["macs"] += int(out.numel()) * kernel_mul

    def linear_hook(module: nn.Linear, inputs: Tuple[Any, ...], output: Any):
        out = _first_tensor(output)
        if out is None:
            return
        counter["macs"] += int(out.numel()) * int(module.in_features)

    for sub in qabr.modules():
        if isinstance(sub, nn.Conv2d):
            handles.append(sub.register_forward_hook(conv_hook))
        elif isinstance(sub, nn.Linear):
            handles.append(sub.register_forward_hook(linear_hook))

    try:
        with torch.inference_mode():
            _ = model(image, [prompt], num_samples=1, compute_m1=False)
        torch.cuda.synchronize()
    finally:
        for h in handles:
            h.remove()

    macs = float(counter["macs"])
    flops = 2.0 * macs  # conventional MAC = one multiply + one add
    return {
        "qabr_conv_macs_per_mc": macs,
        "qabr_conv_flops_per_mc": flops,
        "qabr_conv_gmacs_per_mc": macs / 1e9,
        "qabr_conv_gflops_per_mc": flops / 1e9,
    }


def benchmark_latency(
    model: nn.Module,
    image: torch.Tensor,
    prompt: str,
    mc: int,
    warmup: int,
    repeat: int,
) -> Dict[str, float]:
    if repeat <= 0:
        raise ValueError("repeat must be > 0")

    with torch.inference_mode():
        for _ in range(max(0, warmup)):
            out = model(image, [prompt], num_samples=mc, compute_m1=False)
            del out
    torch.cuda.synchronize()

    # Clear only unused cached blocks. Model weights remain allocated.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline_alloc = float(torch.cuda.memory_allocated())
    baseline_reserved = float(torch.cuda.memory_reserved())

    times_ms: List[float] = []
    with torch.inference_mode():
        for _ in range(repeat):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(image, [prompt], num_samples=mc, compute_m1=False)
            torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000.0)
            del out

    peak_alloc = float(torch.cuda.max_memory_allocated())
    peak_reserved = float(torch.cuda.max_memory_reserved())
    mean_ms = float(statistics.mean(times_ms))
    median_ms = float(statistics.median(times_ms))
    std_ms = float(statistics.stdev(times_ms)) if len(times_ms) > 1 else 0.0
    return {
        f"latency_mc{mc}_mean_ms": mean_ms,
        f"latency_mc{mc}_median_ms": median_ms,
        f"latency_mc{mc}_std_ms": std_ms,
        f"throughput_mc{mc}_cases_per_s": 1000.0 / mean_ms,
        f"memory_mc{mc}_baseline_alloc_mib": baseline_alloc / (1024.0 ** 2),
        f"memory_mc{mc}_baseline_reserved_mib": baseline_reserved / (1024.0 ** 2),
        f"memory_mc{mc}_peak_alloc_mib": peak_alloc / (1024.0 ** 2),
        f"memory_mc{mc}_peak_reserved_mib": peak_reserved / (1024.0 ** 2),
        f"memory_mc{mc}_activation_peak_mib": max(0.0, peak_alloc - baseline_alloc) / (1024.0 ** 2),
        f"timed_repeats_mc{mc}": int(repeat),
        f"warmup_repeats_mc{mc}": int(warmup),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the paper efficiency benchmark")
    set_seed(args.seed)

    cfg = load_cfg_from_cfg_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.MODEL.DEVICE = "cuda"

    model = build_medclipseg_unimedclip(cfg)
    ckpt_meta = load_checkpoint(model, args.checkpoint)
    model.eval().to("cuda")

    total_params = numel(model)
    trainable_params = trainable_numel(model)
    qabr = getattr(model, "qabr", None)
    qabr_params = numel(qabr) if qabr is not None else 0
    qabr_trainable = trainable_numel(qabr) if qabr is not None else 0

    if args.variant == "BASE" and qabr_params != 0:
        raise RuntimeError(f"BASE unexpectedly contains QABR parameters: {qabr_params}")
    if args.variant == "FULL" and qabr_params != 8882:
        raise RuntimeError(
            f"Paper claims 8,882 QABR parameters, but instantiated FULL has {qabr_params}. "
            "Refusing to produce an efficiency table from a mismatched model."
        )

    image = torch.randn(1, 3, args.input_size, args.input_size, device="cuda", dtype=torch.float32)

    # Execute one hook-counted MC1 forward before latency timing. This does not
    # affect the final statistics because timing has independent warm-up passes.
    compute = qabr_conv_compute(model, image, args.prompt)

    mc1 = benchmark_latency(
        model, image, args.prompt, 1,
        warmup=args.warmup_mc1, repeat=args.repeat_mc1,
    )
    mc30 = benchmark_latency(
        model, image, args.prompt, 30,
        warmup=args.warmup_mc30, repeat=args.repeat_mc30,
    )

    result: Dict[str, Any] = {
        "variant": args.variant,
        "seed": args.seed,
        "input_size": args.input_size,
        "batch_size": 1,
        "device_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "checkpoint": os.path.abspath(args.checkpoint),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "qabr_params": qabr_params,
        "qabr_trainable_params": qabr_trainable,
        "total_params_m": total_params / 1e6,
        "trainable_params_m": trainable_params / 1e6,
        "checkpoint_epoch": ckpt_meta["checkpoint_epoch"],
        "deployment_epoch": ckpt_meta["deployment_epoch"],
        "weight_source": ckpt_meta["weight_source"],
        **compute,
        **mc1,
        **mc30,
    }

    out_json = Path(args.output_json)
    out_csv = Path(args.output_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)

    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"[OK] wrote {out_json}")
    print(f"[OK] wrote {out_csv}")


if __name__ == "__main__":
    main()
