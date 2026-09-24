#!/usr/bin/env python3
"""All results in one table: every sequence x arm, mean ± σ over its runs, against the ground truth.

    python scripts/results_table.py [<data root>] [--out=<prefix>] [--offset=best] [--extra=/home/photogrammetry/kiss_runs]

Reads the run folders under <data root>/runs (default /media/photogrammetry/A26C3DDF6C3DAF431/data) with the
evaluation of scripts/evaluate_ncd.py, and writes <prefix>.md and <prefix>.csv (default <data root>/runs/results_all).
Sequences: Newer College 2020 01_short (#041), the five of 2021 (#043), Oxford Spires christ-church-02 / -03 full
recordings.  --offset=best: every run is scored at its own best time shift (evaluate_ncd.best_offset, #045),
default <prefix> then results_all_best_offset.  Arms are the run-folder names up to the first "_" (kiss, sift, surf); folders that do not exist are skipped.
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_ncd import evaluate, load_gt  # noqa: E402

ROOT = Path(next((a for a in sys.argv[1:] if not a.startswith("--")), "/media/photogrammetry/A26C3DDF6C3DAF431/data"))
OFFSET = "best" if "--offset=best" in sys.argv else 0.0
OUT = Path(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")),
                ROOT / "runs" / ("results_all_best_offset" if OFFSET == "best" else "results_all")))
NC, RUNS = ROOT / "newer_college", ROOT / "runs"
EXTRA = Path(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--extra=")), "/home/photogrammetry/kiss_runs"))

# (dataset, sequence, sensor, ground truth, frame, runs folder)
SEQUENCES = [("Newer College 2020", "01_short", "Ouster OS1-64", NC / "2020/01_short_experiment", "ncd2020",
              RUNS / "newer_college_01_short")]
for seq, gt in [("quad_easy", "collection 1 - newer college/ground_truth/tum_format/gt-nc-quad-easy.csv"),
                ("stairs", "collection 1 - newer college/ground_truth/tum_format/gt-nc-stairs.csv"),
                ("cloister", "collection 2 - newer college/ground_truth/tum_format/gt-nc-cloister.csv"),
                ("math_easy", "collection 3 - maths institute/ground_truth/tum_format/gt_math_easy.csv"),
                ("underground_easy", "collection 4 - underground mine/ground truth/tum_format/easy_gt_state_tum.csv")]:
    SEQUENCES.append(("Newer College 2021", seq, "Ouster OS0-128", NC / "2021" / gt, "ncd2021", RUNS / "newer_college_2021" / seq))
for n in (2, 3):
    SEQUENCES.append(("Oxford Spires", f"christ-church-0{n}", "Hesai QT64",
                      ROOT / f"oxford_spires/2024-03-18-christ-church-0{n}/ground_truth/gt-tum_church_{n}.txt", "spires",
                      RUNS / "oxford_spires_full" / f"church_0{n}"))

ARMS = {"kissncd": "KISS-SLAM, kiss_icp NCD loader", "kissnodeskew": "KISS-SLAM, no deskew", "kissdetail": "KISS-SLAM, indoor_detail", "kiss": "KISS-SLAM",
        "sift": "i3 + SIFT", "surf": "i3 + SURF", "surftrans": "i3 + SURF, translation only",
        "surfrot": "i3 + SURF, rotation only", "surfsmooth3": "i3 + SURF, rotation smoothed (3)"}
METRICS = [("ate", "ATE [m]", "{:.3f}"), ("rpe_t", "RPE 1 s [cm]", "{:.2f}"), ("rpe_r", "RPE 1 s [°]", "{:.3f}"),
           ("path", "path [m]", "{:.1f}"), ("excess", "path vs GT [%]", "{:+.1f}"), ("z_rmse", "z RMSE [m]", "{:.3f}"),
           ("kitti", "KITTI [%]", "{:.2f}"), ("fail", "image fails", "{:.0f}"), ("offset", "time shift [s]", "{:+.3f}")]


def main():
    rows = []
    for dataset, seq, sensor, gt, frame, folder in SEQUENCES:
        # Also the same folder under EXTRA (runs written to ext4 after the ntfs3 kernel bug, #046); a run counts
        # only if its log ends with the "wall" line of /usr/bin/time, i.e. it finished (the crashed ones did not).
        found = {}
        for base in (folder, EXTRA / folder.relative_to(RUNS)):
            for p in (sorted(base.glob("*_*")) if base.exists() else []):
                log = p.parent / f"{p.name}.log"
                if p.is_dir() and log.exists() and "\nwall " in log.read_text(errors="replace").replace("\r", "\n"):
                    found[p.name] = p
        runs = [found[k] for k in sorted(found)]
        if not runs:
            print(f"skip {seq}: no runs in {folder}")
            continue
        gt_t, gt_T, _ = load_gt(gt, frame)
        res = {r.name: evaluate(gt_t, gt_T, r, OFFSET) for r in runs}
        for v in res.values():
            v["excess"] = 100 * (v["path"] / v["gt_path"] - 1)
        gt_path = next(iter(res.values()))["gt_path"]
        for arm in ARMS:
            vs = [v for n, v in res.items() if n.split("_")[0] == arm]
            if not vs:
                continue
            row = dict(dataset=dataset, sequence=seq, sensor=sensor, gt_path=gt_path, arm=ARMS[arm], runs=len(vs))
            for c, _, _ in METRICS:
                x = np.array([v[c] for v in vs], float)
                row[c], row[c + "_sd"] = np.nanmean(x) if np.isfinite(x).any() else np.nan, (np.std(x, ddof=1) if len(vs) > 1 else np.nan)
            rows.append(row)
        print(f"{seq}: {len(runs)} runs")

    with open(OUT.with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "sequence", "sensor", "gt_path_m", "arm", "runs"] + sum([[c, c + "_sd"] for c, _, _ in METRICS], []))
        for r in rows:
            w.writerow([r["dataset"], r["sequence"], r["sensor"], f"{r['gt_path']:.1f}", r["arm"], r["runs"]]
                       + sum([[f"{r[c]:.4f}", "" if np.isnan(r[c + "_sd"]) else f"{r[c + '_sd']:.4f}"] for c, _, _ in METRICS], []))

    def cell(r, c, f):
        if np.isnan(r[c]):
            return "—"
        s, sd = f.format(r[c]), r[c + "_sd"]
        return s + (" ± " + f.replace("+", "").format(sd) if not np.isnan(sd) and sd > 1e-9 else "")

    lines = ["# Results against the ground truth" + (" — each run at its best time shift" if OFFSET == "best" else ""), "",
             ("Every run is scored at the time shift (added to its scan stamps) that minimises its rotation RPE over 1 s, "
              "searched in -0.15..+0.25 s (last column).  Arms differ in which instant of the sweep a pose stands for, and "
              "the rotation RPE of a hand-held sensor doubles within 50 ms of shift (#045).  " if OFFSET == "best" else
              "Scan stamps as recorded (shift 0).  "),
             "KISS-SLAM default config (the setting of the KISS-SLAM paper), except the arms \"KISS-SLAM, indoor_detail\" "
             "(configs/indoor_detail.yaml: voxel 0.25 m, max range 50 m, local maps 15 m) and \"KISS-SLAM, no deskew\" "
             "(configs/kiss_paper_nodeskew.yaml: paper config, deskew off).  Mean ± σ over the runs of each arm "
             "(KISS-SLAM is deterministic: its runs are identical).  ATE: RMSE after a rigid alignment.  RPE over 1 s.  "
             "KITTI: relative translation error over 100-800 m segments (undefined below 100 m).  "
             "*Best value per sequence in bold* (lower is better; path: closest to the GT).  "
             "Per CLAUDE.md, judge by RPE and path length: the ATE of a single run is not a measurement (#037).", ""]
    head = "| Dataset | Sequence (GT path) | Arm | runs | " + " | ".join(h for _, h, _ in METRICS) + " |"
    lines += [head, "|" + "---|" * (4 + len(METRICS))]
    for key in dict.fromkeys((r["dataset"], r["sequence"]) for r in rows):
        grp = [r for r in rows if (r["dataset"], r["sequence"]) == key]
        best = {c: min((r for r in grp if not np.isnan(r[c])), key=lambda r: abs(r[c]) if c == "excess" else r[c], default=None)
                for c, _, _ in METRICS if c not in ("path", "fail", "offset")}
        for i, r in enumerate(grp):
            cells = []
            for c, _, f in METRICS:
                s = cell(r, c, f)
                cells.append(f"**{s}**" if best.get(c) is r and len(grp) > 1 else s)
            first = f"{r['dataset']} | {r['sequence']} ({r['gt_path']:.0f} m)" if i == 0 else " | "
            lines.append(f"| {first} | {r['arm']} | {r['runs']} | " + " | ".join(cells) + " |")
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(f"→ {OUT.with_suffix('.md')}, {OUT.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
