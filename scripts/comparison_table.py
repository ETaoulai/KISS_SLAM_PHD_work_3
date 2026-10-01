#!/usr/bin/env python3
"""Per-sequence comparison of all methods (#083-#085) from a results_table.py csv, as Markdown.

    python scripts/comparison_table.py <results csv> <out.md>

APE (each dataset's official protocol) for every sequence and method, best per sequence in bold, failures (APE > 5 m)
marked; then RTE / RRE (KITTI 100-800 m, NCD / Spires only) and per-method medians, worst case and failure counts.
"""
import csv
import math
import statistics as st
import sys

ARMS = [("i3 + SURF, two starting points, range image when intensity fails", "Ours SLAM"),
        ("i3 + SURF, two starts + range fallback, odometry only (no loop closures, replay_backend none, #083)", "Ours odo"),
        ("KISS-SLAM", "KISS-SLAM"), ("KISS-SLAM, no deskew", "KISS no deskew"),
        ("GenZ-ICP (odometry, own config, #083)", "GenZ-ICP"), ("MAD-ICP (odometry, own config, #083)", "MAD-ICP"),
        ("DLO (LiDAR-only odometry, authors' config with imu false, #085)", "DLO"),
        ("CT-ICP (continuous-time odometry, robust low-inertia profile, #083)", "CT-ICP"),
        ("Traj-LO (continuous-time odometry, own config, #083)", "Traj-LO"),
        ("FAST-LIO2 (LiDAR-inertial, reference, #083)", "FAST-LIO2 (IMU)"),
        ("COIN-LIO (LiDAR-inertial, intensity + IMU, reference, #083)", "COIN-LIO (IMU)")]
FAIL = 5.0


def num(x):
    try:
        v = float(x)
        return v
    except (TypeError, ValueError):
        return None


def main():
    rows = list(csv.DictReader(open(sys.argv[1])))
    names = dict(ARMS)
    t, meta = {}, {}
    for x in rows:
        if x["arm"] in names:
            t.setdefault(x["sequence"], {})[names[x["arm"]]] = x
            meta[x["sequence"]] = x["dataset"]
    cols = [n for _, n in ARMS]

    def table(metric, fmt, title, only_defined=False):
        out = [f"### {title}", "", "| dataset | sequence | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
        for s, v in t.items():
            vals = {a: num(v[a][metric]) if a in v else None for a in cols}
            if only_defined and not any(x is not None and math.isfinite(x) for x in vals.values()):
                continue
            ok = [x for x in vals.values() if x is not None and math.isfinite(x) and (metric != "ate" or x <= FAIL)]
            best = min(ok) if ok else None

            def cell(a):
                x = vals[a]
                if a not in v:
                    return "—"
                if x is None or not math.isfinite(x):
                    return "fail" if metric == "ate" else "—"
                txt = fmt.format(x) if x < 100 else f"{x:.0f}"
                if metric == "ate" and x > FAIL:
                    return f"*{txt}* ✗"
                return f"**{txt}**" if x == best else txt
            out.append(f"| {meta[s]} | {s} | " + " | ".join(cell(a) for a in cols) + " |")
        return out

    out = ["# Comparison with other methods — all sequences (#083–#085)", "",
           "*Each dataset's official protocol (#061). APE in m; **bold** = best per sequence (among non-failures); ✗ = failure "
           f"(APE > {FAIL:g} m); — = not run (method cannot read the data: Traj-LO / FAST-LIO2 / DLO no .pcd for 01_short; COIN-LIO "
           "Ouster OS0-128 only; Traj-LO long experiment > 46 GB memory; MAD-ICP segfault on the long experiment). Ours: SLAM mean of "
           "4 seeds; all other methods one run (deterministic). The other methods are odometry only — compare them with “Ours odo”. "
           "FAST-LIO2 and COIN-LIO use an IMU (reference, not competitors).*", ""]
    out += table("ate", "{:.3f}", "APE [m]")
    out += [""] + table("rte", "{:.2f}", "RTE [%] (KITTI 100–800 m; NCD / Spires with ≥ 100 m of path)", only_defined=True)
    out += [""] + table("rpe_t", "{:.1f}", "RPE 1 m translation [cm] (NCD / Spires)", only_defined=True)
    out += ["", "### Summary per method", "",
            "| method | sequences | median APE [m] | worst APE [m] | failures (APE > 5 m) | median RPE 1 m [cm] | median RTE [%] | median RRE [°/100 m] |",
            "|---|---|---|---|---|---|---|---|"]
    for a in cols:
        v = [t[s][a] for s in t if a in t[s]]
        ate = [num(x["ate"]) for x in v]
        fin = [x for x in ate if x is not None and math.isfinite(x)]
        fails = sum(1 for x in ate if x is None or not math.isfinite(x) or x > FAIL)
        med = lambda k: [num(x[k]) for x in v if num(x[k]) is not None and math.isfinite(num(x[k]))]
        rp, rt, rr = med("rpe_t"), med("rte"), med("rre")
        worst = max(fin) if fin else float("nan")
        out.append(f"| {a} | {len(v)} | {st.median(fin):.3f} | {worst:.3f}" + (" ✗" if worst > FAIL else "") + f" | {fails} | "
                   f"{st.median(rp):.1f} | {st.median(rt):.2f} | {st.median(rr):.2f} |" if fin else f"| {a} | {len(v)} | — | — | {fails} | — | — | — |")
    open(sys.argv[2], "w").write("\n".join(out) + "\n")
    print(f"{sys.argv[2]}: {len(t)} sequences x {len(cols)} methods")


if __name__ == "__main__":
    main()
