#!/usr/bin/env python3
"""Φταίει η κίνηση ή η γεωμετρία για το σφάλμα στροφής στη σκάλα;

Το KISS προβλέπει κάθε σάρωση με σταθερή ταχύτητα (last_pose @ last_delta) και με την
ίδια υπόθεση κάνει το deskew. Όταν η κίνηση ΑΛΛΑΖΕΙ απότομα από σάρωση σε σάρωση, και τα
δύο αστοχούν. Άρα, αν φταίει η κίνηση, το σφάλμα στροφής ανά σάρωση πρέπει να ακολουθεί
την «αλλαγή στροφής» δ_k = γωνία(Δ_{k-1}^{-1} Δ_k), όπου Δ_k η σχετική κίνηση του GT.

Κρίσιμο ερώτημα: με την ΙΔΙΑ αλλαγή κίνησης, είναι το σφάλμα μεγαλύτερο στη σκάλα;
  - όχι → η κίνηση εξηγεί το σφάλμα της σκάλας· κανένα intensity δεν θα βοηθήσει·
  - ναι → η γεωμετρία της σκάλας συνεισφέρει· η Ι-1 στοχεύει σωστά.
Υπολογίζεται με γραμμικό μοντέλο  σφάλμα ~ a + b·δ + c_άνω·[σκάλα πάνω] + c_κάτω·[σκάλα κάτω],
με διαστήματα εμπιστοσύνης από block bootstrap (οι διαδοχικές σαρώσεις δεν είναι ανεξάρτητες).

    python scripts/analyze_motion_vs_error.py [run_dir] [--offset-ms -70]

--offset-ms: μετατόπιση των χρονοσφραγίδων μας πριν την παρεμβολή του GT. Οι πόζες του
KISS αντιστοιχούν στο ΤΕΛΟΣ της σάρωσης (deskew με exp((stamp-1)·ω), Preprocessing.cpp) και
η χρονοσφραγίδα τους είναι ο χρόνος εγγραφής στο bag (~105 ms μετά την αρχή) → σωστές ±5 ms.
Παρ' όλα αυτά, το σφάλμα ανά σάρωση ελαχιστοποιείται σταθερά στα −70…−75 ms σε τρία runs και
δύο μετρικές → πιθανότατα σύμβαση χρόνου του GT. Βλ. docs/experiment_log.md, #010.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_gt import base_to_lidar, find_tum, interpolate, load_tum  # noqa: E402

_args = [a for a in sys.argv[1:] if not a.startswith("--offset-ms")]
RUN = _args[0] if _args and not _args[0].lstrip("-").isdigit() else "runs/indoor_detail_base_overlapfix"
OFFSET = next((float(a.split("=", 1)[1]) for a in sys.argv[1:] if a.startswith("--offset-ms=")), 0.0) / 1000
GT = "gt/church_02_gt-tum.txt"
FIG = Path("docs/figures/motion_vs_error.png")  # με --offset-ms: _offset πριν την κατάληξη
SEGMENTS = {"ισόγειο": [(0, 820), (2258, 2402)], "όροφος": [(990, 2014)],
            "σκάλα πάνω": [(820, 990)], "σκάλα κάτω": [(2014, 2258)]}
BLOCK, N_BOOT = 20, 2000


def angle(R):
    return np.degrees(np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))


def main():
    gs, gT = load_tum(GT)
    gT = base_to_lidar(gT)
    st, est = load_tum(find_tum(RUN))
    ok, G = interpolate(gs, gT, st + OFFSET)
    E = est[ok]
    dt = np.median(np.diff(st[ok]))

    inv = np.linalg.inv
    dG = inv(G[:-1]) @ G[1:]                      # κίνηση GT ανά σάρωση
    dE = inv(E[:-1]) @ E[1:]                      # εκτιμώμενη κίνηση ανά σάρωση
    err = inv(dG) @ dE                            # σφάλμα σχετικής κίνησης ανά σάρωση
    rate = angle(dG[:, :3, :3]) / dt              # ρυθμός στροφής (°/s)
    change = np.r_[np.nan, angle((inv(dG[:-1]) @ dG[1:])[:, :3, :3])]   # αλλαγή στροφής (°/σάρωση)
    rot_err = angle(err[:, :3, :3])               # σφάλμα στροφής (°/σάρωση)
    Rw = G[:-1, :3, :3]
    tilt_err = np.degrees(np.arccos(np.clip(
        np.einsum("nij,njk,nlk->nil", Rw, err[:, :3, :3], Rw)[:, 2, 2], -1, 1)))

    n = len(rot_err)
    seg = np.full(n, "", dtype=object)
    for name, spans in SEGMENTS.items():
        for a, b in spans:
            seg[a:min(b, n)] = name
    valid = ~np.isnan(change)

    print(f"run: {RUN}   ({n} σχετικές κινήσεις, dt = {dt*1000:.0f} ms, μετατόπιση χρόνου {OFFSET*1000:+.0f} ms)\n")
    print(f"{'τμήμα':12}{'σαρώσεις':>9}{'ρυθμός στροφής':>16}{'αλλαγή στροφής':>16}{'σφάλμα στροφής':>16}{'σφάλμα κλίσης':>15}")
    print(f"{'':12}{'':>9}{'(°/s)':>16}{'(°/σάρωση)':>16}{'(°/σάρωση)':>16}{'(°/σάρωση)':>15}")
    for name in SEGMENTS:
        m = (seg == name) & valid
        print(f"{name:12}{m.sum():>9}{rate[m].mean():>16.1f}{change[m].mean():>16.3f}"
              f"{rot_err[m].mean():>16.3f}{tilt_err[m].mean():>15.3f}")

    r_c, _ = spearmanr(change[valid], rot_err[valid])
    r_r, _ = spearmanr(rate[valid], rot_err[valid])
    print(f"\nΣυσχέτιση Spearman με το σφάλμα στροφής:  αλλαγή στροφής ρ = {r_c:+.2f} · ρυθμός στροφής ρ = {r_r:+.2f}")

    # Ίδια αλλαγή κίνησης → διαφέρει το σφάλμα ανάμεσα σε σκάλα και ορόφους;
    edges = np.nanpercentile(change[valid], [0, 25, 50, 75, 90, 100])
    print("\nΣφάλμα στροφής (°/σάρωση) ανά κλάση αλλαγής στροφής — ίδια κίνηση, διαφορετικό περιβάλλον:")
    print(f"{'αλλαγή στροφής (°)':>22}{'όροφοι':>14}{'σκάλα πάνω':>14}{'σκάλα κάτω':>14}")
    floors = np.isin(seg, ["ισόγειο", "όροφος"])
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = valid & (change >= lo) & (change <= hi)
        cells = []
        for m in (floors, seg == "σκάλα πάνω", seg == "σκάλα κάτω"):
            k = b & m
            cells.append(f"{rot_err[k].mean():.3f} ({k.sum()})" if k.sum() >= 5 else "—")
        print(f"{f'{lo:.2f}–{hi:.2f}':>22}" + "".join(f"{c:>14}" for c in cells))

    # Γραμμικό μοντέλο + block bootstrap
    X = np.column_stack([np.ones(n), change, seg == "σκάλα πάνω", seg == "σκάλα κάτω"]).astype(float)[valid]
    y = rot_err[valid]
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    rng = np.random.default_rng(0)
    nb = len(y) // BLOCK
    boots = []
    for _ in range(N_BOOT):
        starts = rng.integers(0, len(y) - BLOCK, nb)
        idx = (starts[:, None] + np.arange(BLOCK)).ravel()
        boots.append(np.linalg.lstsq(X[idx], y[idx], rcond=None)[0])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    names = ["σταθερός όρος", "ανά 1° αλλαγής στροφής", "επιπλέον στη σκάλα πάνω", "επιπλέον στη σκάλα κάτω"]
    print("\nΜοντέλο: σφάλμα στροφής ~ a + b·αλλαγή + c·[σκάλα]   (95 % διάστημα, block bootstrap)")
    for nm, c, l, h in zip(names, coef, lo, hi):
        print(f"  {nm:26}{c:+.3f}°   [{l:+.3f}, {h:+.3f}]")
    for i, nm in ((2, "σκάλα πάνω"), (3, "σκάλα κάτω")):
        m = (seg == nm) & valid
        excess_vs_floor = rot_err[m].mean() - rot_err[floors & valid].mean()
        print(f"  {nm}: υπερβάλλον σφάλμα έναντι ορόφων {excess_vs_floor:+.3f}° → εξηγείται από την κίνηση "
              f"{excess_vs_floor - coef[i]:+.3f}°, από το περιβάλλον {coef[i]:+.3f}°")

    fig_path = FIG if OFFSET == 0 else FIG.with_name(f"{FIG.stem}_offset{OFFSET*1000:+.0f}ms.png")
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    colors = {"ισόγειο": "#9e9e9e", "όροφος": "#616161", "σκάλα πάνω": "#1f77b4", "σκάλα κάτω": "#d62728"}
    ax = axes[0]
    for name, c in colors.items():
        m = (seg == name) & valid
        ax.scatter(change[m], rot_err[m], s=4, alpha=0.35, color=c, label=name)
    xs = np.linspace(0, np.nanpercentile(change, 99), 50)
    ax.plot(xs, coef[0] + coef[1] * xs, "k--", lw=1, label="μοντέλο (όροφοι)")
    ax.set_xlim(0, np.nanpercentile(change, 99)); ax.set_ylim(0, np.nanpercentile(rot_err, 99))
    ax.set_xlabel("αλλαγή στροφής της μονάδας (°/σάρωση, από GT)"); ax.set_ylabel("σφάλμα στροφής ICP (°/σάρωση)")
    ax.legend(fontsize=8, markerscale=3); ax.grid(alpha=0.3)
    ax = axes[1]
    k = np.arange(n)
    ax.plot(k, rot_err, lw=0.5, color="#d62728", label="σφάλμα στροφής ICP")
    ax.plot(k, change, lw=0.5, color="#1f77b4", alpha=0.7, label="αλλαγή στροφής (GT)")
    for name in ("σκάλα πάνω", "σκάλα κάτω"):
        for a, b in SEGMENTS[name]:
            ax.axvspan(a, b, color="#ffe082", alpha=0.4)
    ax.set_xlabel("σάρωση (κίτρινο = σκάλα)"); ax.set_ylabel("°/σάρωση")
    ax.set_ylim(0, np.nanpercentile(np.r_[rot_err, change[valid]], 99.5))
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_path, dpi=90)
    print(f"\nεικόνα → {fig_path}")


if __name__ == "__main__":
    main()
