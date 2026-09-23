"""Η αναδιάρθρωση του process_scan δεν αλλάζει τίποτα πέρα από το σκόπιμο.

    python tests/test_replace_mode.py

Συγκρίνει τον τρέχοντα κώδικα με το kiss_slam/slam.py του commit 1409b64 (REF, το
τελευταίο πριν από την αναδιάρθρωση), τροφοδοτώντας και τα δύο με τα ίδια πραγματικά scans:

(a) baseline νέο == baseline REF             → η αναδιάρθρωση δεν άγγιξε το baseline
(b) "replace" που κρατά ΟΛΑ τα σημεία == baseline → το αντίγραφο του register_frame είναι πιστό
(c) "refine" νέο == "refine" REF, όταν στον νέο κώδικα επαναφερθεί προσωρινά η
    παλιά αντιστοίχιση intensity → η αναδιάρθρωση δεν άλλαξε τίποτα άλλο. Η ίδια
    η διόρθωση αναμένεται να αλλάζει το refine και αναφέρεται πληροφοριακά.
"""
import sys, subprocess, importlib.util, warnings, tempfile, copy
from pathlib import Path
import numpy as np
warnings.simplefilter("ignore")

from kiss_icp.datasets import dataset_factory
from kiss_slam.config import load_config
from kiss_slam.tools.point_cloud2 import read_point_cloud
import kiss_slam.slam as NEW

TOL = 1e-9   # δύο ίδια runs του KISS διαφέρουν ήδη ~1e-14 (multithreading)

# Καρφωμένο στο τελευταίο commit ΠΡΙΝ από την αναδιάρθρωση και τη διόρθωση του intensity.
# (Με "HEAD" το τεστ έχανε το νόημά του μόλις γινόταν commit η ίδια η αλλαγή.)
REF = "1409b64"
head_src = subprocess.run(["git", "show", f"{REF}:kiss_slam/slam.py"], capture_output=True, text=True, check=True).stdout
tmp = Path(tempfile.mkdtemp()) / "slam_head.py"; tmp.write_text(head_src)
spec = importlib.util.spec_from_file_location("slam_head", tmp)
HEAD = importlib.util.module_from_spec(spec); spec.loader.exec_module(HEAD)


class _RefLoopCloser(HEAD.LoopCloser):
    """Το REF slam.py περιμένει την παλιά επιστροφή του LoopCloser.compute (#014 την άλλαξε σε λίστα)."""

    def compute(self, query_id, points, local_map_graph):
        accepted = super().compute(query_id, points, local_map_graph)
        return (True, *accepted[0]) if accepted else (False, -1, query_id, np.eye(4))


HEAD.LoopCloser = _RefLoopCloser

def run(module, cfg, jump, n):
    ds = dataset_factory(dataloader="rosbag", data_dir=Path("data/church_02_cut.bag"),
                         sequence=None, topic="/hesai/pandar", meta=None)
    ds.read_point_cloud = read_point_cloud
    slam = module.KissSLAM(cfg)
    for i in range(jump + n):
        xyz, ts, it = ds[i]
        if i >= jump:
            slam.process_scan(xyz, ts, it if cfg.intensity.enabled else None)
    return np.array(slam.poses), slam

def cfg_with(**kw):
    c = load_config(Path("configs/indoor_fast.yaml"))
    for k, v in kw.items():
        setattr(c.intensity, k, v)
    return c

ok = True
def report(name, a, b):
    global ok
    d = np.abs(a - b).max()
    passed = d < TOL
    ok &= passed
    print(f"  {name:62s} max|Δ| = {d:.1e}   {'PASS' if passed else 'FAIL'}")

print("(a) baseline: νέο vs REF  [scans 0–299]")
pa, _ = run(NEW,  cfg_with(enabled=False), 0, 300)
pb, _ = run(HEAD, cfg_with(enabled=False), 0, 300)
report("baseline νέο == baseline REF", pa, pb)

print("(b) replace που κρατά όλα τα σημεία vs baseline  [scans 0–299]")
pr, s_r = run(NEW, cfg_with(enabled=True, mode="replace", keep_ratio=1.0), 0, 300)
report("replace(keep_ratio=1.0) == baseline", pr, pa)
kept = np.mean([m["n_filtered_pts"] / m["n_source_pts"] for m in s_r.icp_metrics_log])
print(f"  {'(έλεγχος: κράτησε ' + f'{100*kept:.1f}% των σημείων' + ')':62s}")

print("(c) refine: νέο vs REF  [scans 900–1099]")
# Η διόρθωση του intensity ΣΚΟΠΙΜΑ αλλάζει το refine όπου υπάρχουν σημεία > 50 m.
# Για να ελεγχθεί ότι η αναδιάρθρωση δεν άλλαξε ΤΙΠΟΤΑ ΑΛΛΟ, επαναφέρεται προσωρινά
# η παλιά αντιστοίχιση (ακατέργαστο intensity, χωρίς αφαίρεση) στον νέο κώδικα.
_fixed = NEW._align_intensity_to_preprocessed
NEW._align_intensity_to_preprocessed = lambda deskewed_all, raw, pre, mx, mn: (raw, False)
pn_old, _ = run(NEW, cfg_with(enabled=True, mode="refine"), 900, 200)
NEW._align_intensity_to_preprocessed = _fixed
ph, _ = run(HEAD, cfg_with(enabled=True), 900, 200)
report("refine νέο (με την ΠΑΛΙΑ αντιστοίχιση) == refine REF", pn_old, ph)

pn, s_n = run(NEW, cfg_with(enabled=True, mode="refine"), 900, 200)
d_fix = np.abs(pn - ph).max()
print(f"  {'(πληροφοριακά) πόσο αλλάζει το refine η ίδια η διόρθωση:':62s} max|Δ| = {d_fix:.1e}")

print("(d) διαγνωστικό model deviation: δεν είναι πια αντίγραφο της κίνησης")
m = s_n.icp_metrics_log[50]
diff = abs(m["motion_trans_m"] - m["model_dev_trans_m"])
print(f"  frame 950: κίνηση={m['motion_trans_m']:.4f} m, διόρθωση ICP={m['model_dev_trans_m']:.4f} m   "
      f"{'PASS' if diff > 1e-6 else 'FAIL'}")
ok &= diff > 1e-6

print("\nRESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
