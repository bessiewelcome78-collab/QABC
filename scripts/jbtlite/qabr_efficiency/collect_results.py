#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path


def fmt(x, n=2):
    return f"{float(x):.{n}f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    args = p.parse_args()
    root = Path(args.root)
    rows = {}
    for variant in ("BASE", "FULL"):
        path = root / variant / "result.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows[variant] = json.loads(path.read_text())

    b, f = rows["BASE"], rows["FULL"]
    delta_total = int(f["total_params"]) - int(b["total_params"])
    delta_trainable = int(f["trainable_params"]) - int(b["trainable_params"])
    if int(f["qabr_params"]) != 8882 or delta_total != 8882:
        raise RuntimeError(
            f"8,882-parameter claim FAILED: qabr_params={f['qabr_params']} total_delta={delta_total}"
        )
    if delta_trainable != 8882:
        raise RuntimeError(
            f"Trainable-parameter delta is not 8,882: {delta_trainable}"
        )

    base_l = float(b["latency_mc30_mean_ms"])
    full_l = float(f["latency_mc30_mean_ms"])
    latency_delta = full_l - base_l
    latency_pct = 100.0 * latency_delta / base_l
    base_mem = float(b["memory_mc30_peak_alloc_mib"])
    full_mem = float(f["memory_mc30_peak_alloc_mib"])
    mem_delta = full_mem - base_mem
    mem_pct = 100.0 * mem_delta / base_mem

    table = [
        {
            "Method": "Base / w/o QABR",
            "Total Params (M)": fmt(b["total_params_m"], 3),
            "Trainable Params (M)": fmt(b["trainable_params_m"], 3),
            "Extra Params": "0",
            "Extra QABR Conv GFLOPs/MC": "0.000",
            "MC1 Latency (ms)": fmt(b["latency_mc1_mean_ms"], 2),
            "MC30 Latency (ms)": fmt(b["latency_mc30_mean_ms"], 2),
            "MC30 Peak Mem (MiB)": fmt(b["memory_mc30_peak_alloc_mib"], 1),
        },
        {
            "Method": "Full QABR",
            "Total Params (M)": fmt(f["total_params_m"], 3),
            "Trainable Params (M)": fmt(f["trainable_params_m"], 3),
            "Extra Params": f"+{delta_total:,}",
            "Extra QABR Conv GFLOPs/MC": fmt(f["qabr_conv_gflops_per_mc"], 3),
            "MC1 Latency (ms)": fmt(f["latency_mc1_mean_ms"], 2),
            "MC30 Latency (ms)": fmt(f["latency_mc30_mean_ms"], 2),
            "MC30 Peak Mem (MiB)": fmt(f["memory_mc30_peak_alloc_mib"], 1),
        },
    ]

    out_csv = root / "FINAL_EFFICIENCY.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(table[0].keys()))
        writer.writeheader(); writer.writerows(table)

    verdict = root / "FINAL_EFFICIENCY_VERDICT.txt"
    verdict.write_text(
        "QABR efficiency verification\n"
        f"parameter_claim_8882=PASS\n"
        f"base_total_params={int(b['total_params']):,}\n"
        f"full_total_params={int(f['total_params']):,}\n"
        f"delta_total_params={delta_total:,}\n"
        f"delta_trainable_params={delta_trainable:,}\n"
        f"qabr_params={int(f['qabr_params']):,}\n"
        f"qabr_conv_gflops_per_mc={float(f['qabr_conv_gflops_per_mc']):.6f}\n"
        f"qabr_conv_gflops_mc30={30*float(f['qabr_conv_gflops_per_mc']):.6f}\n"
        f"mc30_latency_base_ms={base_l:.4f}\n"
        f"mc30_latency_full_ms={full_l:.4f}\n"
        f"mc30_latency_delta_ms={latency_delta:.4f}\n"
        f"mc30_latency_overhead_pct={latency_pct:.4f}\n"
        f"mc30_peak_mem_base_mib={base_mem:.4f}\n"
        f"mc30_peak_mem_full_mib={full_mem:.4f}\n"
        f"mc30_peak_mem_delta_mib={mem_delta:.4f}\n"
        f"mc30_peak_mem_overhead_pct={mem_pct:.4f}\n",
        encoding="utf-8",
    )

    latex = root / "FINAL_EFFICIENCY_LATEX_ROWS.txt"
    latex.write_text(
        f"Base / w/o QABR & {table[0]['Total Params (M)']} & {table[0]['Trainable Params (M)']} & 0 & 0.000 & {table[0]['MC30 Latency (ms)']} & {table[0]['MC30 Peak Mem (MiB)']} \\\\\n"
        f"Full QABR & {table[1]['Total Params (M)']} & {table[1]['Trainable Params (M)']} & +{delta_total:,} & {table[1]['Extra QABR Conv GFLOPs/MC']} & {table[1]['MC30 Latency (ms)']} & {table[1]['MC30 Peak Mem (MiB)']} \\\\\n",
        encoding="utf-8",
    )

    print("=== FINAL EFFICIENCY TABLE ===")
    for r in table:
        print(r)
    print("=== VERDICT ===")
    print(verdict.read_text())
    print(out_csv)
    print(verdict)
    print(latex)

if __name__ == "__main__":
    main()
