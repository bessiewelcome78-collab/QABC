#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import re
import statistics
from pathlib import Path


LABELS = {
    "BASE": "Base / w/o QABR",
    "NAIVE_GLOBAL": "Naive Global Refiner",
    "NO_LOCAL_SUPPORT": "w/o Local Boundary Support",
    "LOCAL_ONLY": "Local Support Only",
    "LOCAL_DIRECTION": "Local + Direction (w/o Semantic)",
    "LOCAL_SEMANTIC": "Local + Semantic (w/o Direction)",
    "NO_TANGENT": "w/o Mass-Tangent Projection",
    "DIRECT_RESIDUAL": "Direct Residual Training",
    "NO_READINESS": "w/o Readiness Gate",
    "FULL": "Full QABR",
    "NO_EXPLICIT_QUERY": "w/o Explicit Query Channels",
    "BAND3": "Band width 3",
    "BAND7": "Band width 7",
    "DELTA1": "Max displacement 1",
    "DELTA3": "Max displacement 3",
    "EVAL_SCALE_050": "Inference scale 0.5",
    "EVAL_SCALE_070": "Inference scale 0.7",
    "EVAL_SCALE_100": "Inference scale 1.0",
}


def read_cases(path: Path | None):
    if path is None or not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        try:
            out.append({
                "case_id": row.get("Case_ID") or row.get("Name") or "",
                "dsc": float(row["DSC"]),
                "nsd": float(row["NSD"]),
                "gt_area": float(row.get("GT_Area_Fraction", "nan")),
                "pred_area": float(row.get("Pred_Area_Fraction", "nan")),
            })
        except (KeyError, TypeError, ValueError):
            continue
    return out


def find_csv(run: Path, split: str, mode: str):
    matches = sorted(run.glob(f"{split}/**/*_{split}_*_{mode}.csv"))
    return matches[-1] if matches else None


def mean(values):
    values = list(values)
    return statistics.fmean(values) if values else None


def fmt(v, digits=4):
    return "" if v is None else f"{v:.{digits}f}"


def parse_epoch(run: Path):
    path = run / "logs" / "test_mc30.log"
    if not path.exists():
        return ""
    text = path.read_text(errors="ignore")
    found = re.findall(r"Checkpoint epoch=(\d+)", text)
    return found[-1] if found else ""


def summarize_run(dataset, variant, seed, run):
    val_true = read_cases(find_csv(run, "val", "true2d"))
    val_legacy = read_cases(find_csv(run, "val", "paper_legacy"))
    test_true = read_cases(find_csv(run, "test", "true2d"))
    test_legacy = read_cases(find_csv(run, "test", "paper_legacy"))
    positive = [r for r in test_true if r["gt_area"] > 0]
    empty = [r for r in test_true if r["gt_area"] == 0]
    status = "COMPLETE" if (run / "COMPLETE.txt").exists() else ("PARTIAL" if run.exists() else "MISSING")
    return {
        "dataset": dataset,
        "variant": variant,
        "label": LABELS.get(variant, variant),
        "seed": seed,
        "status": status,
        "best_epoch": parse_epoch(run),
        "val_dsc": mean(r["dsc"] for r in val_true),
        "val_true2d_nsd": mean(r["nsd"] for r in val_true),
        "val_legacy_nsd": mean(r["nsd"] for r in val_legacy),
        "test_dsc": mean(r["dsc"] for r in test_true),
        "test_true2d_nsd": mean(r["nsd"] for r in test_true),
        "test_legacy_nsd": mean(r["nsd"] for r in test_legacy),
        "n_cases": len(test_true),
        "n_positive": len(positive),
        "n_empty": len(empty),
        "positive_dsc": mean(r["dsc"] for r in positive),
        "positive_true2d_nsd": mean(r["nsd"] for r in positive),
        "empty_clear_rate": mean(1.0 if r["pred_area"] == 0 else 0.0 for r in empty),
        "empty_mean_pred_area": mean(r["pred_area"] for r in empty),
        "true_cases": test_true,
    }


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def paired_stats(base_cases, full_cases, bootstrap=10000):
    b = {r["case_id"]: r for r in base_cases}
    f = {r["case_id"]: r for r in full_cases}
    ids = sorted(set(b) & set(f))
    dd = [f[i]["dsc"] - b[i]["dsc"] for i in ids]
    dn = [f[i]["nsd"] - b[i]["nsd"] for i in ids]
    if not ids:
        return None
    rng = random.Random(20260916)
    boot_d, boot_n = [], []
    count = len(ids)
    for _ in range(bootstrap):
        sample = [rng.randrange(count) for _ in range(count)]
        boot_d.append(statistics.fmean(dd[i] for i in sample))
        boot_n.append(statistics.fmean(dn[i] for i in sample))
    return {
        "n": count,
        "delta_dsc": statistics.fmean(dd),
        "dsc_ci_low": percentile(boot_d, 0.025),
        "dsc_ci_high": percentile(boot_d, 0.975),
        "delta_true2d_nsd": statistics.fmean(dn),
        "nsd_ci_low": percentile(boot_n, 0.025),
        "nsd_ci_high": percentile(boot_n, 0.975),
        "dsc_improve_rate": mean(x > 0 for x in dd),
        "dsc_harm_rate": mean(x < 0 for x in dd),
        "nsd_improve_rate": mean(x > 0 for x in dn),
        "nsd_harm_rate": mean(x < 0 for x in dn),
    }


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--study-id", required=True)
    args = ap.parse_args()
    root = Path(args.project) / "reviewer_ablation_results" / args.study_id
    root.mkdir(parents=True, exist_ok=True)

    records = []
    for dataset in ("BUSI", "Kvasir"):
        dsroot = root / dataset
        if not dsroot.exists():
            continue
        for variant_dir in sorted(p for p in dsroot.iterdir() if p.is_dir()):
            for seed_dir in sorted(variant_dir.glob("seed*")):
                try:
                    seed = int(seed_dir.name[4:])
                except ValueError:
                    continue
                records.append(summarize_run(dataset, variant_dir.name, seed, seed_dir))

    anchors = {(r["dataset"], r["seed"], r["variant"]): r for r in records}
    for r in records:
        base = anchors.get((r["dataset"], r["seed"], "BASE"))
        full = anchors.get((r["dataset"], r["seed"], "FULL"))
        r["gain_vs_base_dsc"] = None if not base or r["test_dsc"] is None or base["test_dsc"] is None else r["test_dsc"] - base["test_dsc"]
        r["gain_vs_base_true2d_nsd"] = None if not base or r["test_true2d_nsd"] is None or base["test_true2d_nsd"] is None else r["test_true2d_nsd"] - base["test_true2d_nsd"]
        r["drop_from_full_dsc"] = None if not full or r["test_dsc"] is None or full["test_dsc"] is None else full["test_dsc"] - r["test_dsc"]
        r["drop_from_full_true2d_nsd"] = None if not full or r["test_true2d_nsd"] is None or full["test_true2d_nsd"] is None else full["test_true2d_nsd"] - r["test_true2d_nsd"]

    numeric = [
        "val_dsc", "val_true2d_nsd", "val_legacy_nsd", "test_dsc", "test_true2d_nsd",
        "test_legacy_nsd", "positive_dsc", "positive_true2d_nsd", "empty_clear_rate",
        "empty_mean_pred_area", "gain_vs_base_dsc", "gain_vs_base_true2d_nsd",
        "drop_from_full_dsc", "drop_from_full_true2d_nsd",
    ]
    output_rows = []
    for r in records:
        row = {k: v for k, v in r.items() if k != "true_cases"}
        for key in numeric:
            row[key] = fmt(row.get(key))
        output_rows.append(row)

    fields = [
        "dataset", "variant", "label", "seed", "status", "best_epoch",
        "val_dsc", "val_true2d_nsd", "val_legacy_nsd", "test_dsc", "test_true2d_nsd",
        "test_legacy_nsd", "gain_vs_base_dsc", "gain_vs_base_true2d_nsd",
        "drop_from_full_dsc", "drop_from_full_true2d_nsd", "n_cases", "n_positive",
        "n_empty", "positive_dsc", "positive_true2d_nsd", "empty_clear_rate",
        "empty_mean_pred_area",
    ]
    write_csv(root / "FINAL_RESULTS.csv", output_rows, fields)

    core_order = ["BASE", "NAIVE_GLOBAL", "NO_LOCAL_SUPPORT", "LOCAL_ONLY", "LOCAL_DIRECTION", "LOCAL_SEMANTIC", "NO_TANGENT", "DIRECT_RESIDUAL", "NO_READINESS", "FULL"]
    paper = [r for r in output_rows if int(r["seed"]) == 42 and r["variant"] in core_order]
    paper.sort(key=lambda r: (r["dataset"], core_order.index(r["variant"])))
    write_csv(root / "FINAL_PAPER_ABLATION.csv", paper, fields)

    multi = []
    for dataset in ("BUSI", "Kvasir"):
        for variant in ("BASE", "FULL"):
            vals = [r for r in records if r["dataset"] == dataset and r["variant"] == variant and r["status"] == "COMPLETE" and r["test_dsc"] is not None]
            if not vals:
                continue
            dsc = [r["test_dsc"] for r in vals]
            nsd = [r["test_true2d_nsd"] for r in vals]
            multi.append({
                "dataset": dataset, "variant": variant, "seeds": " ".join(str(r["seed"]) for r in vals),
                "n": len(vals), "dsc_mean": fmt(mean(dsc)), "dsc_std": fmt(statistics.stdev(dsc) if len(dsc) > 1 else 0.0),
                "true2d_nsd_mean": fmt(mean(nsd)), "true2d_nsd_std": fmt(statistics.stdev(nsd) if len(nsd) > 1 else 0.0),
            })
    write_csv(root / "FINAL_MULTI_SEED.csv", multi, ["dataset", "variant", "seeds", "n", "dsc_mean", "dsc_std", "true2d_nsd_mean", "true2d_nsd_std"])

    paired = []
    for dataset in ("BUSI", "Kvasir"):
        seeds = sorted({r["seed"] for r in records if r["dataset"] == dataset})
        for seed in seeds:
            b = anchors.get((dataset, seed, "BASE"))
            f = anchors.get((dataset, seed, "FULL"))
            if not b or not f:
                continue
            stats = paired_stats(b["true_cases"], f["true_cases"])
            if stats:
                paired.append({"dataset": dataset, "seed": seed, **{k: fmt(v) if isinstance(v, float) else v for k, v in stats.items()}})
    pair_fields = ["dataset", "seed", "n", "delta_dsc", "dsc_ci_low", "dsc_ci_high", "delta_true2d_nsd", "nsd_ci_low", "nsd_ci_high", "dsc_improve_rate", "dsc_harm_rate", "nsd_improve_rate", "nsd_harm_rate"]
    write_csv(root / "FINAL_PAIRED_BOOTSTRAP.csv", paired, pair_fields)

    complete = sum(r["status"] == "COMPLETE" for r in records)
    print(f"study={args.study_id} runs={len(records)} complete={complete}")
    print(root / "FINAL_RESULTS.csv")
    print(root / "FINAL_PAPER_ABLATION.csv")
    print(root / "FINAL_MULTI_SEED.csv")
    print(root / "FINAL_PAIRED_BOOTSTRAP.csv")


if __name__ == "__main__":
    main()
