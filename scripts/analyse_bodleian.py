import sys, csv, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.argv = ["x"]; sys.path.insert(0, "/home/photogrammetry/Kiss_SLAM-main/scripts")
import results_table as T
from evaluate_ncd import best_offset
from evaluate_gt import find_tum, load_tum, interpolate
X = Path("/home/photogrammetry/kiss_runs/oxford_spires_full")
def decisions(seq, run):
    d = next(x for x in T.SEQUENCES if x[1] == seq); t, G, _ = T.load_gt(d[3], d[4])
    r = X / run; rows = list(csv.DictReader(open(sorted(r.glob("*/two_start.csv"))[-1])))
    if "image_rx" not in rows[0]: return None
    st, est = load_tum(find_tum(r)); sh = best_offset(t, G, st, est); ok, g = interpolate(t, G, st + sh)
    idx = np.full(len(st), -1); idx[ok] = np.arange(ok.sum()); out = []
    for row in rows:
        k = int(row["scan"])
        if k < 1 or idx[k] < 1 or idx[k - 1] < 0: continue
        Dg = np.linalg.inv(g[idx[k - 1]]) @ g[idx[k]]; e = {}
        for n in ("image", "cv"):
            M = np.eye(4); M[:3, :3] = R.from_rotvec(np.radians([float(row[f"{n}_r{a}"]) for a in "xyz"])).as_matrix(); M[:3, 3] = [float(row[f"{n}_t{a}"]) for a in "xyz"]
            E = np.linalg.inv(Dg) @ M; e[n] = (np.degrees(np.linalg.norm(R.from_matrix(E[:3, :3]).as_rotvec())), 100 * np.linalg.norm(E[:3, 3]))
        out.append(dict(t=st[k] - st[0], kept=row["kept"], fi=float(row["fit_image"]), fc=float(row["fit_cv"]), dis=float(row["disagree_deg"]),
                        ei=e["image"], ec=e["cv"]))
    return out
score = lambda e: e[0] + 0.1 * e[1]                     # deg + (cm / 10) = deg + 10 x m, as in #058
for seq, runs in (("bodleian-library-02", ["bodleian_02/surftwo_s1", "bodleian_02/surftwo_s2", "bodleian_02/surftwo_s3"]),
                  ("blenheim-palace-02", ["blenheim_02/surftwolog_s0", "blenheim_02/surftwo_s3"])):
    allsw = []
    for run in runs:
        D = decisions(seq, run)
        if D is None: continue
        sw = [x for x in D if x["kept"] == "cv"]
        helpful = [x for x in sw if score(x["ec"]) < score(x["ei"]) - 1]; harmful = [x for x in sw if score(x["ec"]) > score(x["ei"]) + 1]
        early = [x for x in D if x["t"] < 60]
        print(f"{seq} {run.split('/')[1]:14s}: {len(D)} decisions, switched to CV {len(sw)}: helpful {len(helpful)}, harmful {len(harmful)}, neutral {len(sw)-len(helpful)-len(harmful)} | first 60 s: {len(early)} decisions, {sum(x['kept']=='cv' for x in early)} switches")
        allsw += [(x, "helpful" if x in helpful else "harmful" if x in harmful else "neutral") for x in sw]
    for kind in ("helpful", "harmful"):
        xs = [x for x, k in allsw if k == kind]
        if not xs: continue
        gain = np.array([(x["fi"] - x["fc"]) / x["fi"] for x in xs]); dis = np.array([x["dis"] for x in xs])
        ei_r = np.array([x["ei"][0] for x in xs]); ec_r = np.array([x["ec"][0] for x in xs]); ei_t = np.array([x["ei"][1] for x in xs]); ec_t = np.array([x["ec"][1] for x in xs])
        print(f"   {kind:8s} switches ({len(xs)}): fit gain of CV p10/p50/p90 {np.percentile(gain,10)*100:5.1f} / {np.percentile(gain,50)*100:5.1f} / {np.percentile(gain,90)*100:5.1f} %"
              f" | disagreement p50 {np.median(dis):5.1f} deg | rotation error image -> CV p50 {np.median(ei_r):5.2f} -> {np.median(ec_r):5.2f} deg, translation {np.median(ei_t):5.1f} -> {np.median(ec_t):5.1f} cm")
