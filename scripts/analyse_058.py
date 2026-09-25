"""#058 analysis: two starts on 16 sequences, range variants, and the two-start decisions against the ground truth."""
import sys, csv, glob, re
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.argv = ["x"]; sys.path.insert(0, "/home/photogrammetry/Kiss_SLAM-main/scripts")
import results_table as T
from evaluate_ncd import evaluate, best_offset
from evaluate_gt import find_tum, load_tum, interpolate
X = Path("/home/photogrammetry/kiss_runs"); D = Path("/media/photogrammetry/A26C3DDF6C3DAF431/data/runs")
REL = {"christ-church-02": "oxford_spires_full/church_02", "christ-church-03": "oxford_spires_full/church_03", "keble-college-03": "oxford_spires_full/keble_03",
       "observatory-quarter-01": "oxford_spires_full/observatory_01", "blenheim-palace-02": "oxford_spires_full/blenheim_02",
       "bodleian-library-02": "oxford_spires_full/bodleian_02", "01_short": "newer_college_01_short"}
rel = lambda s: REL.get(s, f"newer_college_2021/{s}")
def runs(seq, arm):
    out = []
    for base in (D, X):
        for r in sorted((base / rel(seq)).glob(f"{arm}_s*")):
            lg = r.parent / f"{r.name}.log"
            if r.is_dir() and lg.exists() and "\nwall " in lg.read_text(errors="replace").replace("\r", "\n"):
                out.append(r)
    return out
seqs = {d[1]: d for d in T.SEQUENCES}

if "--only4" not in sys.argv[0:0] and False:
  pass
print("== 1. Two starts vs plain SURF (seed 0): ATE m / RPE translation cm / RPE rotation deg; scans registered twice, CV kept, extra time")
for s, d in seqs.items():
    t, G, _ = T.load_gt(d[3], d[4]); a = runs(s, "surf"); b = runs(s, "surftwo") + runs(s, "surftwolog")
    a = [r for r in a if r.name == "surf_s0"]; b = [r for r in b if r.name.endswith("_s0")][:1]
    if not a or not b: continue
    va, vb = evaluate(t, G, a[0], "best"), evaluate(t, G, b[0], "best")
    lg = (b[0].parent / f"{b[0].name}.log").read_text(errors="replace")
    m = re.search(r"two starting points: (\d+) scans registered twice, constant-velocity start kept in (\d+)(?:; extra time ([\d.]+) s of ([\d.]+) s in process_scan \(([\d.]+) %\))?", lg)
    extra = f"{m[1]:>4s} twice, CV {m[2]:>3s}" + (f", +{m[5]} % time" if m and m[5] else "") if m else "   0 twice"
    print(f"   {s:22s} SURF {va['ate']:6.3f} / {va['rpe_t']:5.2f} / {va['rpe_r']:.3f}   two starts {vb['ate']:6.3f} / {vb['rpe_t']:5.2f} / {vb['rpe_r']:.3f}   {extra}")

print("\n== 2. Rerun with logging equals the earlier two-start run (poses max |diff|)")
for s in ("blenheim-palace-02", "keble-college-03", "underground_hard"):
    a = [r for r in runs(s, "surftwo") if r.name == "surftwo_s0"]; b = runs(s, "surftwolog")
    if a and b:
        pa, pb = (np.load(glob.glob(f"{r}/latest/*_poses.npy")[0]) for r in (a[0], b[0])); print(f"   {s:22s} {np.abs(pa - pb).max():.1e}")

print("\n== 3. Range variants, ATE m per seed (RPE translation cm)")
for s in ("blenheim-palace-02", "keble-college-03", "underground_hard"):
    t, G, _ = T.load_gt(seqs[s][3], seqs[s][4]); line = []
    for label, arm in (("two starts", "surftwo"), ("gate + range fallback", "surfrangefb"), ("three starts (+range)", "surftworange")):
        v = [evaluate(t, G, r, "best") for r in runs(s, arm)]
        line.append(f"{label}: " + ", ".join(f"{x['ate']:.2f} ({x['rpe_t']:.1f})" for x in v))
    rf = [re.search(r"range image used for (\d+) scans", (r.parent / f"{r.name}.log").read_text(errors="replace")) for r in runs(s, "surfrangefb")]
    print(f"   {s:22s} " + " | ".join(line) + f" | range used in {[int(m[1]) for m in rf if m]} scans")

print("\n== 4. Two-start decisions vs the ground truth (runs surftwolog_s0 and surftworange_s*)")
ang = lambda rv: float(np.linalg.norm(rv))
for s in ("blenheim-palace-02", "keble-college-03", "underground_hard"):
    t, G, _ = T.load_gt(seqs[s][3], seqs[s][4])
    for r in runs(s, "surftwolog") + runs(s, "surftworange")[:1]:
        f = sorted(r.glob("*/two_start.csv"))
        if not f: continue
        rows = list(csv.DictReader(open(f[-1])))
        st, est = load_tum(find_tum(r)); sh = best_offset(t, G, st, est); ok, g = interpolate(t, G, st + sh)
        idx = np.full(len(st), -1); idx[ok] = np.arange(ok.sum())
        names = [k[:-3] for k in rows[0] if k.endswith("_rx")]
        right, total, kept_count, err_kept, err_best, err_image, big = 0, 0, {}, [], [], [], []
        for row in rows:
            k = int(row["scan"])
            if k < 1 or idx[k] < 1 or idx[k - 1] < 0: continue
            Dg = np.linalg.inv(g[idx[k - 1]]) @ g[idx[k]]
            errs = {}
            for n in names:
                if row.get(f"{n}_rx", "") == "": continue          # this start was not available (e.g. range failed)
                Mk = np.eye(4); Mk[:3, :3] = R.from_rotvec(np.radians([float(row[f"{n}_r{a}"]) for a in "xyz"])).as_matrix()
                Mk[:3, 3] = [float(row[f"{n}_t{a}"]) for a in "xyz"]
                E = np.linalg.inv(Dg) @ Mk; errs[n] = np.degrees(ang(R.from_matrix(E[:3, :3]).as_rotvec())) + 10 * np.linalg.norm(E[:3, 3])
            closest = min(errs, key=errs.get); total += 1; right += row["kept"] == closest
            kept_count[row["kept"]] = kept_count.get(row["kept"], 0) + 1
            err_kept.append(errs[row["kept"]]); err_best.append(errs[closest]); err_image.append(errs["image"])
            if errs["image"] > 10: big.append((k, row["kept"], round(errs["image"], 1), round(errs[row["kept"]], 1)))
        print(f"   {s:22s} {r.name:15s} {total} decisions, kept {kept_count}; kept = the start closest to the truth in {right}/{total} ({100*right/max(total,1):.0f} %)")
        print(f"   {'':22s} {'':15s} error of the result (deg + 10 x m): image start p50 {np.median(err_image):.2f} p99 {np.percentile(err_image,99):.2f} | kept p50 {np.median(err_kept):.2f} p99 {np.percentile(err_kept,99):.2f} | best possible p99 {np.percentile(err_best,99):.2f}")
        print(f"   {'':22s} {'':15s} scans where the image start converged > 10 off: {len(big)}; kept instead: {sum(1 for b in big if b[1] != 'image')}; e.g. {big[:5]}")
