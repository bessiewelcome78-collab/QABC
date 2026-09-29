#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, re
from pathlib import Path

VARIANTS = [
    ("Innovation1_Evidence", "no_uncertainty_evidence"),
    ("Innovation1_Evidence", "no_directional_geometry"),
    ("Innovation1_Evidence", "no_semantic_detail"),
    ("Innovation2_Constraint", "no_local_constraint"),
    ("Innovation2_Constraint", "no_mass_tangent"),
    ("Innovation2_Constraint", "no_zero_forward"),
]
DATASETS = ["BUSI", "Kvasir"]

def read(path: Path) -> str:
    return path.read_text(errors="replace") if path.is_file() else ""

def last_metric(text: str, name: str):
    matches = re.findall(rf"Average {re.escape(name)}[^:]*:\s*([0-9.]+)%", text)
    return float(matches[-1]) if matches else None

def best_epoch(text: str):
    for pat in (r"best_val_epoch=(\d+)", r"Best.*epoch[^0-9]*(\d+)"):
        m = re.findall(pat, text, flags=re.I)
        if m:
            return int(m[-1])
    return None

def fmt(x):
    return "NA" if x is None else f"{x:.2f}"

ap = argparse.ArgumentParser()
ap.add_argument("--project", default="/home/tsz-25/MedCLIPSeg_JBTLite_QABR_V12_20260911")
ap.add_argument("--study-id", required=True)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

project = Path(args.project)
base = project / "qabr_two_innovation_ablation_results" / args.study_id
rows = []
for dataset in DATASETS:
    for group, variant in VARIANTS:
        root = base / dataset / variant / f"seed{args.seed}"
        tr = read(root / "logs/train.log")
        true2d = read(root / "logs/eval_test_mc30_true2d.log")
        legacy = read(root / "logs/eval_test_mc30_paper_legacy.log")
        val = read(root / "logs/eval_val_mc10_true2d.log")
        done = (root / "DONE").is_file()
        rows.append({
            "dataset": dataset,
            "innovation": group,
            "variant": variant,
            "seed": args.seed,
            "best_val_epoch": best_epoch(tr),
            "val_dsc": last_metric(val, "DSC"),
            "val_true2d_nsd": last_metric(val, "NSD"),
            "test_dsc": last_metric(true2d, "DSC"),
            "test_true2d_nsd": last_metric(true2d, "NSD"),
            "test_legacy_nsd": last_metric(legacy, "NSD"),
            "status": "COMPLETE" if done else ("PARTIAL" if root.exists() else "PENDING"),
            "root": str(root),
        })

out = base / "QABR_TWO_INNOV_ABLATION_SUMMARY.csv"
out.parent.mkdir(parents=True, exist_ok=True)
fields = list(rows[0].keys())
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); w.writerows(rows)

print("dataset  innovation              variant                       TestDSC  true2D   legacy   status")
print("-" * 104)
for r in rows:
    print(f"{r['dataset']:<8} {r['innovation']:<23} {r['variant']:<29} "
          f"{fmt(r['test_dsc']):>7} {fmt(r['test_true2d_nsd']):>8} {fmt(r['test_legacy_nsd']):>8}  {r['status']}")
print(f"\nCSV={out}")
print("NOTE: FULL and BASE are intentionally absent; merge existing paper anchors later if desired.")
