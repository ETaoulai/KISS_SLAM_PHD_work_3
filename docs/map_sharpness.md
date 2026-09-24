# Map sharpness against the New College survey map (#048)

Point-to-plane distance of each run's map to the 5 cm survey map (New College: `new-college-combined-5cm-v2.ply`; christ-church-03 (#049): the Oxford Spires TLS map `merged-cloud-1cm.pcd`, 114 M points, thinned to 2 cm around the run), 1 run per arm (seed 0), KISS-SLAM paper config.  **GT poses**: the arm's deskewed scans placed at the ground-truth poses — only the deskew is measured.  **Own poses**: the map the SLAM builds.  Lower = sharper.  *Best per sequence in bold.*

## Deskew only (scans at the ground-truth poses)

| Sequence | Arm | median [cm] | mean [cm] | < 5 cm [%] | < 10 cm [%] | within 0.5 m [%] |
|---|---|---|---|---|---|---|
| quad_easy | KISS-SLAM, no deskew | **2.73** | 4.21 | **71.8** | 90.4 | 99.5 |
|  | KISS-SLAM | 3.14 | 5.18 | 65.7 | 85.1 | 99.3 |
|  | KISS-SLAM, indoor_detail | 3.23 | 5.33 | 64.5 | 84.4 | 99.2 |
|  | i3 + SIFT | 2.88 | 4.72 | 69.2 | 87.5 | 99.3 |
|  | i3 + SURF | 2.77 | 4.45 | 70.9 | 88.9 | 99.5 |
|  | i3 + SURF, rotation only | 2.84 | 4.50 | 70.1 | 88.9 | 99.5 |
| cloister | KISS-SLAM, no deskew | 2.22 | 3.06 | 83.2 | 96.6 | 99.5 |
|  | KISS-SLAM | 2.56 | 3.96 | 75.0 | 91.8 | 98.6 |
|  | KISS-SLAM, indoor_detail | 2.27 | 3.42 | 80.6 | 94.4 | 99.0 |
|  | i3 + SIFT | **1.85** | 2.53 | **89.6** | 97.8 | 99.5 |
|  | i3 + SURF | 1.88 | 2.57 | 89.0 | 97.8 | 99.5 |
|  | i3 + SURF, rotation only | 2.82 | 3.73 | 73.5 | 95.1 | 99.5 |
| christ-church-03 (Hesai) | KISS-SLAM, no deskew | 2.29 | 3.55 | 78.4 | 93.8 | 98.9 |
|  | KISS-SLAM | 4.21 | 6.67 | 55.9 | 78.9 | 98.2 |
|  | KISS-SLAM, indoor_detail | 3.86 | 6.28 | 58.8 | 80.8 | 98.3 |
|  | i3 + SIFT | 2.04 | 3.21 | 81.7 | 95.0 | 98.9 |
|  | i3 + SURF | **2.01** | 3.15 | **82.2** | 95.2 | 98.9 |
|  | i3 + SURF, rotation only | 2.35 | 3.67 | 75.9 | 93.2 | 98.9 |
| christ-church-02 (Hesai) | KISS-SLAM, no deskew | 2.31 | 3.83 | 78.2 | 92.4 | 96.4 |
|  | KISS-SLAM | 5.18 | 7.85 | 48.7 | 73.5 | 95.4 |
|  | KISS-SLAM, indoor_detail | 4.35 | 6.90 | 54.9 | 78.2 | 95.6 |
|  | i3 + SIFT | 2.20 | 3.55 | 80.3 | 93.6 | 96.5 |
|  | i3 + SURF | **2.13** | 3.42 | **81.4** | 94.1 | 96.5 |
|  | i3 + SURF, rotation only | 2.53 | 4.00 | 74.8 | 92.0 | 96.5 |

## The map as built (own poses)

| Sequence | Arm | median [cm] | mean [cm] | < 5 cm [%] | < 10 cm [%] | within 0.5 m [%] |
|---|---|---|---|---|---|---|
| quad_easy | KISS-SLAM, no deskew | 2.06 | 2.93 | 84.4 | 96.5 | 99.6 |
|  | KISS-SLAM | 2.17 | 3.15 | 82.4 | 95.5 | 99.5 |
|  | KISS-SLAM, indoor_detail | 2.14 | 3.13 | 82.6 | 95.4 | 99.5 |
|  | i3 + SIFT | 1.93 | 2.68 | 87.2 | 97.3 | 99.5 |
|  | i3 + SURF | **1.86** | 2.53 | **88.7** | 97.8 | 99.6 |
|  | i3 + SURF, rotation only | 1.94 | 2.66 | 87.1 | 97.6 | 99.6 |
| cloister | KISS-SLAM, no deskew | **6.10** | 9.56 | **44.9** | 63.8 | 34.9 |
|  | KISS-SLAM | 7.25 | 9.99 | 40.4 | 59.7 | 45.6 |
|  | KISS-SLAM, indoor_detail | 6.42 | 9.06 | 43.1 | 63.7 | 85.7 |
|  | i3 + SIFT | 6.84 | 9.06 | 40.9 | 63.3 | 73.2 |
|  | i3 + SURF | 6.93 | 9.37 | 40.6 | 62.2 | 70.5 |
|  | i3 + SURF, rotation only | 6.17 | 9.73 | 44.1 | 63.1 | 70.6 |
| christ-church-03 (Hesai) | KISS-SLAM, no deskew | 3.44 | 4.65 | 65.3 | 90.0 | 98.9 |
|  | KISS-SLAM | 3.60 | 4.92 | 63.2 | 88.4 | 98.9 |
|  | KISS-SLAM, indoor_detail | 3.70 | 4.97 | 62.6 | 88.4 | 98.9 |
|  | i3 + SIFT | **2.17** | 3.10 | **82.9** | 96.5 | 98.9 |
|  | i3 + SURF | 2.26 | 3.23 | 81.1 | 96.0 | 98.9 |
|  | i3 + SURF, rotation only | 2.68 | 3.61 | 76.4 | 95.4 | 98.9 |
| christ-church-02 (Hesai) | KISS-SLAM, no deskew | 7.26 | 10.40 | 37.4 | 61.8 | 90.7 |
|  | KISS-SLAM | 9.11 | 12.24 | 30.6 | 53.5 | 87.7 |
|  | KISS-SLAM, indoor_detail | 13.91 | 16.90 | 21.1 | 38.5 | 59.5 |
|  | i3 + SIFT | 4.70 | 7.25 | 52.4 | 78.3 | 89.3 |
|  | i3 + SURF | 5.00 | 7.59 | 50.0 | 75.3 | 95.2 |
|  | i3 + SURF, rotation only | **4.68** | 6.86 | **52.5** | 77.7 | 95.6 |

**Cloister, own poses: not valid.** Only 35–86 % of the map lies within 0.5 m of the survey and the rigid ICP moved the maps 1–5 m: the 429 m trajectory has drift and no loop closure, so one rigid alignment cannot fit the whole map.  **Stairs** is not in the table: the stairwell interior is not in the survey (25–36 % within 0.5 m even at GT poses).

**christ-church-02, own poses: partly valid.** The 642 m trajectory has drift and no loop closure: the rigid ICP moved the maps 15–208 cm
and 60–95 % of the points lie within 0.5 m (indoor_detail, ATE 3.1 m, is the worst).  **christ-church-03 at 100 % of each scan (#051)**
gives the same numbers as the 10 % sample above within 0.1 percentage point (e.g. SURF 82.2 % / 81.2 %), so 10 % is enough.
