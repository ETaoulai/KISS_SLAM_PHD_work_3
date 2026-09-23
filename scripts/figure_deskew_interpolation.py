#!/usr/bin/env python3
"""Σχήμα: τι σημαίνει «η θέση του σαρωτή τη στιγμή s» μέσα στη σάρωση (ομοιόμορφα vs καμπύλη, #033).

Τρία πάνελ: (α) πού ήταν ο σαρωτής στο 30 % της σάρωσης με ομοιόμορφη κατανομή· (β) γωνία ως προς τον χρόνο,
ευθεία (KISS) έναντι καμπύλης με επιτάχυνση (δικό μας μοντέλο «car»)· (γ) τι σημαίνει η διαφορά για ένα σημείο.
Οι στροφές είναι υπερβολικές για να φαίνονται· τα νούμερα στους τίτλους είναι τα πραγματικά.

    python scripts/figure_deskew_interpolation.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = Path("docs/figures/deskew_interpolation.png")
TOTAL_DEG, TOTAL_M, ALPHA_DEG = 10.0, 1.0, -6.0     # στροφή, μετατόπιση ανά σάρωση· επιτάχυνση (υπερβολικά, για το σχήμα)


def heading(ax, x, y, deg, color, label=None, L=0.30):
    a = np.radians(deg)
    ax.arrow(x, y, L * np.cos(a), L * np.sin(a), head_width=0.055, color=color, length_includes_head=True, zorder=3)
    if label:
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, -14), fontsize=8, color=color)


def main():
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    s = np.linspace(0, 1, 200)

    # ── (α) ομοιόμορφη κατανομή: στο s παίρνεις το s της στροφής και το s της μετατόπισης
    a0 = ax[0]
    SHOW = 4.0                                   # οι γωνίες σχεδιάζονται ×4 για να φαίνονται
    ang = TOTAL_DEG * s
    x, y = TOTAL_M * s, np.zeros_like(s)
    a0.plot(x, y, "-", color="0.7", lw=1.2, zorder=1)
    for si, col, lab, off in [(0.0, "tab:blue", "s = 0:  0°", -20), (0.3, "tab:red", "s = 0.3:  3°", 12),
                              (0.5, "tab:orange", "s = 0.5:  5°", -20), (1.0, "tab:green", "s = 1:  10°", 12)]:
        i = int(si * (len(s) - 1))
        a0.plot(x[i], y[i], "o", color=col, ms=6, zorder=3)
        heading(a0, x[i], y[i], SHOW * ang[i], col, None, L=0.42)
        a0.annotate(lab, (x[i], y[i]), textcoords="offset points", xytext=(-6, off), fontsize=9, color=col)
    a0.set_title("(α) Ομοιόμορφη κατανομή του delta (KISS)\nστο 30 % του χρόνου: 30 % της στροφής ΚΑΙ της μετατόπισης\n"
                 "(τα βέλη δείχνουν πού κοιτά· γωνίες ×4 για ορατότητα)", fontsize=9)
    a0.set_xlabel("μετατόπιση μέσα στη σάρωση (m)"); a0.set_ylabel("y (m)"); a0.set_aspect("equal"); a0.grid(alpha=0.3)
    a0.set_xlim(-0.35, 1.75); a0.set_ylim(-0.55, 0.75)

    # ── (β) γωνία ως προς τον χρόνο: ευθεία vs καμπύλη με επιτάχυνση
    a1 = ax[1]
    lin = TOTAL_DEG * s
    cur = (TOTAL_DEG - ALPHA_DEG / 2) * s + ALPHA_DEG / 2 * s ** 2      # ίδιο τέλος, άλλη πορεία
    a1.plot(s, lin, lw=2, color="0.35", label="ομοιόμορφα (KISS): 10°·s")
    a1.plot(s, cur, lw=2, color="tab:red", label="με επιτάχυνση: 13°·s − 3°·s²")
    for si in (0.3,):
        i = int(si * (len(s) - 1))
        a1.plot([si, si], [cur[i], lin[i]], color="tab:red", ls=":", lw=1.5)
        a1.plot(si, lin[i], "o", color="0.35"); a1.plot(si, cur[i], "o", color="tab:red")
        a1.annotate(f"{lin[i]:.1f}° (ομοιόμορφα)", (si, lin[i]), textcoords="offset points", xytext=(10, -16), fontsize=9, color="0.35")
        a1.annotate(f"{cur[i]:.1f}° (καμπύλη)", (si, cur[i]), textcoords="offset points", xytext=(10, 6), fontsize=9, color="tab:red")
    a1.axvline(0.3, color="0.8", lw=0.8, zorder=0)
    a1.set_title("(β) Πόσο είχε στρίψει τη στιγμή s\nίδια αρχή, ίδιο τέλος — άλλη πορεία", fontsize=10)
    a1.set_xlabel("s (κλάσμα της σάρωσης)"); a1.set_ylabel("στροφή (°)"); a1.grid(alpha=0.3); a1.legend(fontsize=8, loc="upper left")

    # ── (γ) τι σημαίνει για ένα σημείο στα 10 m
    a2 = ax[2]
    R, d = 10.0, np.radians(0.63)      # απόσταση σημείου· διαφορά γωνίας στο s = 0.3 (3.63° − 3.00°)
    a2.plot([0], [0], "s", color="k", ms=9); a2.annotate("σαρωτής", (0, 0), textcoords="offset points", xytext=(-14, -18), fontsize=9)
    a2.plot([0, R], [0, 0], color="0.85", lw=1)
    a2.plot(R, 0, "o", color="0.35", ms=9, label="όπου το βάζει η ομοιόμορφη κατανομή")
    a2.plot(R * np.cos(d), R * np.sin(d), "o", color="tab:red", ms=9, label="όπου το βάζει η καμπύλη")
    a2.annotate("", xy=(R * np.cos(d), R * np.sin(d) + 0.02), xytext=(R, -0.02),
                arrowprops=dict(arrowstyle="<->", color="tab:red", lw=1.6))
    a2.annotate("11 cm", (R, R * np.sin(d) / 2), textcoords="offset points", xytext=(12, -4), fontsize=11, color="tab:red")
    a2.set_title("(γ) Ένα σημείο στα 10 m που μετρήθηκε στο s = 0.3\nδιαφορά γωνίας 0.63° → 11 cm διαφορά θέσης\n"
                 "(με τα δικά μας μεγέθη: ~3 cm)", fontsize=9)
    a2.set_xlabel("x (m)"); a2.set_ylabel("y (m)"); a2.grid(alpha=0.3); a2.legend(fontsize=8, loc="lower left")
    a2.set_xlim(-1.5, 13.0); a2.set_ylim(-0.6, 0.6)

    fig.suptitle("Πού ήταν ο σαρωτής τη στιγμή που μετρήθηκε κάθε σημείο  —  οι στροφές είναι υπερβολικές για να φαίνονται",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    FIG.parent.mkdir(parents=True, exist_ok=True); fig.savefig(FIG, dpi=95)
    print(f"→ {FIG}")


if __name__ == "__main__":
    main()
