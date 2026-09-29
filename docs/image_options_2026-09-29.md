# Image options — summary and complete table (29/9/26)

*Official score of each dataset (APE, m; Hilti: challenge APE; NTU: prism ATE), mean over 4 seeds (KISS / no deskew: 1 run,
deterministic). **Bold** = best per sequence. "—" = not run. All variants use two starting points; defaults unchanged
(decision M.T. 28/9: options, not defaults). Entries #069–#076 of `docs/experiment_log.md`. Height problem: pending (#077).*

## Summary

| Option | What it does | Result | Verdict |
|---|---|---|---|
| **Range fallback** (`--range=fallback`) | depth-image motion where the intensity image gives none | tested on all 27: better 10, tie 13, worse 4; large gains NTU (−14…−42 %), Office_Mitte_1 (1.44 → 0.24 m), long experiment | **best general option** where the intensity image fails |
| **Range 3rd start** (`--range=candidate`) | depth-image motion as an extra ICP start | 6 sequences: fixes fast rotation (dynamic_spinning 0.46 → 0.30 m) and Office_Mitte_1; no help on NTU / UZH | for fast rotation / wrong motions |
| **Intensity ×1.0** (`--intensity-scale=1.0`) | brighter fixed scale (Hilti / NTU Ouster are dark at ×255/1024) | failures −82…−97 % (Hilti), −37 % (NTU); scores mixed (better 4, worse 3 of 9) | superseded by gain |
| **Per-scan gain** (`--normalise=gain`, branch `intensity_norm`) | each scan scaled so its p99 = 255 | Hilti better or equal on 5 / 6 (Office_Mitte_1 0.23 m); NCD unharmed; NTU mixed | **for Hilti-type (dark) data** |
| **Panorama 2048** (`--panorama-width=2048`) | the Hilti Ouster's own columns | fewer failures (UZH 64 → 31), no better scores | not worth it |
| `reflectivity` field | calibrated field instead of intensity | worse (NTU 1376 failures vs 358) | rejected |

vs two starts (±1 % = tie), better / tie / worse: range fallback 10 / 13 / 4 (27) · range 3rd start 4 / 1 / 1 (6) · ×1.0 4 / 2 / 3 (9) ·
×1.0 + fallback 5 / 1 / 3 (9) · ×1.0 + 3rd start 5 / 0 / 4 (9) · gain 6 / 1 / 5 (12) · gain + 2048 3 / 2 / 1 (6).

Still open: UZH (no image variant beats no deskew — the image motion there is wrong, not missing); the fast-rotation seed that still
fails on dynamic_spinning; per-sequence best settings differ on NTU.

## Complete table

| Dataset | Sequence | KISS | no deskew | two starts | + range fallback | + range 3rd start | ×1.0 | ×1.0 + fallback | ×1.0 + 3rd start | gain | gain + 2048 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Newer College 2020 | 01_short | 0.419 | 0.350 | **0.305** | 0.307 | — | — | — | — | — | — |
| Newer College 2021 | quad_easy | 0.104 | 0.083 | **0.079** | **0.079** | — | — | — | — | — | — |
| Newer College 2021 | stairs | 3.586 | 2.705 | **2.074** | **2.074** | — | — | — | — | — | — |
| Newer College 2021 | cloister | 0.396 | 0.480 | 0.188 | **0.188** | — | — | — | — | — | — |
| Newer College 2021 | math_easy | 0.160 | 0.104 | 0.108 | 0.108 | — | — | — | — | **0.104** | — |
| Newer College 2021 | underground_easy | 0.117 | 0.092 | **0.067** | **0.067** | — | — | — | — | — | — |
| Oxford Spires | christ-church-02 | 0.777 | 0.546 | 0.209 | **0.207** | — | — | — | — | — | — |
| Oxford Spires | christ-church-03 | 0.143 | 0.089 | **0.044** | **0.044** | — | — | — | — | — | — |
| Newer College 2021 | quad_hard | 0.329 | **0.209** | 0.222 | 0.218 | — | — | — | — | 0.216 | — |
| Newer College 2021 | math_medium | 0.253 | 0.164 | 0.154 | **0.152** | — | — | — | — | — | — |
| Newer College 2021 | underground_medium | 0.162 | 0.097 | **0.061** | **0.061** | — | — | — | — | — | — |
| Newer College 2021 | underground_hard | 12.780 | 12.341 | 0.086 | **0.086** | — | — | — | — | 0.087 | — |
| Oxford Spires | keble-college-03 | 9.757 | 11.370 | **0.094** | 0.094 | — | — | — | — | — | — |
| Oxford Spires | observatory-quarter-01 | 0.497 | 0.436 | **0.078** | 0.080 | — | — | — | — | — | — |
| Oxford Spires | blenheim-palace-02 | 0.293 | **0.205** | 0.273 | 0.268 | — | — | — | — | — | — |
| Oxford Spires | bodleian-library-02 | 1.911 | 1.363 | **0.513** | 0.545 | — | — | — | — | — | — |
| Newer College 2020 | 02_long_experiment | **1.275** | 3.498 | 2.113 | 1.619 | — | — | — | — | — | — |
| Newer College 2020 | dynamic_spinning | **0.159** | 20.751 | 0.460 | 0.504 | 0.302 | — | — | — | — | — |
| Hilti 2021 | Basement_1 | 0.055 | 0.078 | 0.058 | **0.050** | — | 0.058 | 0.058 | 0.060 | 0.052 | 0.052 |
| Hilti 2021 | IC_Office_1 | 6.344 | 1.655 | 0.072 | **0.071** | — | 0.076 | 0.076 | 0.074 | 0.074 | 0.072 |
| Hilti 2021 | Office_Mitte_1 | 4.286 | 0.575 | 1.437 | 0.241 | 0.238 | 0.282 | 0.282 | **0.202** | 0.226 | 0.315 |
| Hilti 2021 | Construction_Site_1 | 0.063 | 0.062 | 0.049 | 0.048 | — | 0.043 | 0.044 | **0.041** | 0.045 | 0.046 |
| Hilti 2021 | LAB_Survey_2 | 0.062 | 0.050 | 0.036 | 0.036 | — | 0.035 | 0.035 | **0.035** | 0.035 | 0.036 |
| Hilti 2021 | UZH_Tracking_Area_Run_2 | 0.585 | **0.204** | 0.503 | 0.551 | 0.502 | 0.576 | 0.579 | 0.576 | 0.576 | 0.579 |
| NTU VIRAL | eee_01 | 2.678 | 2.363 | 2.028 | 1.737 | 1.992 | 2.034 | **1.613** | 1.992 | 1.944 | — |
| NTU VIRAL | eee_02 | 1.486 | 1.490 | 0.820 | **0.679** | 0.830 | 1.101 | 0.927 | 1.101 | 0.877 | — |
| NTU VIRAL | eee_03 | 0.864 | 0.841 | 0.592 | **0.344** | 0.584 | 0.435 | 0.367 | 0.434 | 0.679 | — |
