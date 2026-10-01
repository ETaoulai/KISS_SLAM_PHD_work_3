#!/usr/bin/env python3
"""Paired comparison of two arms across sequences, from a results_table.py csv.

    python scripts/compare_arms.py <results csv> "<arm A>" "<arm B>" [<arm A2> <arm B2> ...]

Works on both results_table.py outputs: the default one (official protocol, evo: RPE 1 m and 1 s, APE) and --legacy
(RPE 1 s, ATE, KITTI of #041-#060).

The unit is the sequence: each arm's mean over its runs on that sequence, paired by sequence.  Per metric:
how many sequences each arm wins (lower is better; path: closer to the GT), the median of (A - B) / B, and a
two-sided Wilcoxon signed-rank test on the paired values (exact for n <= 25; with 8 sequences the smallest
possible p is 2/256 = 0.0078, reached only when one arm wins all 8).
"""
import csv
import sys

import numpy as np
from scipy.stats import wilcoxon

METRICS = [("rpe_t", "RPE 1 s translation"), ("rpe_r", "RPE 1 s rotation"), ("ate", "ATE"),
           ("excess", "path length vs GT (|%|)"), ("z_rmse", "z RMSE"), ("kitti", "KITTI")]
# A results_table.py csv of the official protocol (rpe1s_* present): RPE over 1 m and over 1 s, APE of evo, no KITTI.
METRICS_OFFICIAL = [("rpe_t", "RPE 1 m translation"), ("rpe_r", "RPE 1 m rotation"), ("rpe1s_t", "RPE 1 s translation"),
                    ("rpe1s_r", "RPE 1 s rotation"), ("rte", "RTE 100-800 m (KITTI)"), ("rre", "RRE 100-800 m (KITTI)"), ("ate", "APE (evo)"), ("excess", "path length vs GT (|%|)"),
                    ("z_rmse", "z RMSE")]


def main():
    rows = list(csv.DictReader(open(sys.argv[1])))
    pairs = sys.argv[2:]
    metric_list = METRICS_OFFICIAL if rows and "rpe1s_t" in rows[0] else METRICS
    for a, b in zip(pairs[::2], pairs[1::2]):
        seqs = [s for s in dict.fromkeys(r["sequence"] for r in rows)
                if {a, b} <= {r["arm"] for r in rows if r["sequence"] == s}]
        val = lambda arm, s, m: float(next(r[m] for r in rows if r["sequence"] == s and r["arm"] == arm))
        print(f"\n{a}  vs  {b}   ({len(seqs)} sequences: {', '.join(seqs)})")
        print(f"  {'metric':26s} {'A wins':>7s} {'B wins':>7s} {'median (A-B)/B':>15s} {'Wilcoxon p':>11s}")
        for m, name in metric_list:
            x = np.array([[val(a, s, m), val(b, s, m)] for s in seqs])
            x = x[np.isfinite(x).all(1)]
            if m == "excess":
                x = np.abs(x)
            if len(x) < 3:
                print(f"  {name:26s} {'(fewer than 3 sequences)':>44s}")
                continue
            d = x[:, 0] - x[:, 1]
            p = wilcoxon(x[:, 0], x[:, 1]).pvalue if np.any(d != 0) else 1.0
            print(f"  {name:26s} {int((d < 0).sum()):7d} {int((d > 0).sum()):7d} {100 * np.median(d / x[:, 1]):+14.1f}% {p:11.4f}")


if __name__ == "__main__":
    main()
