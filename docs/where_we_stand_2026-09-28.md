# Where we stand 28/9/26

*Summary of problems and possible solutions, before the next decisions. Details in `docs/experiment_log.md` (entries
cited as #nnn); work on branch `vertical_drift`, `main` unchanged since #066. All conclusions ⏳ until validated by Λ.Γ.*

## The method today

Across 27 sequences (5 datasets, 4 sensors, each dataset's official evaluation), the method with two starting points beats
KISS without deskew on **22 of 27** (−29% APE, p = 0.0007), and upstream KISS on 24 of 27. Translation error per second is
38% lower. Rotation per second is a tie.

## Solved, no action needed

| Former problem | What it turned out to be |
|---|---|
| Per-run time offset (#045) | The pose-time convention. Every pose now carries its own time, with official evaluation (#061). |
| Path length overestimated by +13% (Β.2) | **Per-scan jitter**, not a scale error. Measured once per second the excess is +0.2%, and our method jitters least (#068). |
| Stairs failing (Β.3) | The coarse map. With the finer map (indoor_detail) the method solves them, 2.07 → 0.48 m (#063). |
| Switch margin (Β.6) | No measurable effect, so it isn't used (#062). |
| Different GT subsets per arm (Β.5) | No effect on any conclusion (#068). |
| dynamic_spinning time offset | It belongs to the cameras, not the LiDAR, so it isn't applied (#064). |

## Open problems

| # | Problem | Where it shows | Cause (what we know) | Possible solution | Evidence so far | Cost |
|---|---|---|---|---|---|---|
| **1** | **Height error from the pose graph** | NCD long experiment (APE 2.1 m, of which 2.1 m height). Any long route with loop closures. | Every graph edge has identity information, which makes rotation almost free, so the graph tilts nodes to make horizontal corrections (#067). | **Weight rotation ×100 in the pose graph** (a setting, default = upstream). | Offline replay: 2.11 → **0.39 m**, every arm improves, 01_short unchanged. | Small code change + ~3 h confirmation runs |
| **2** | **Image gives no motion** | NTU (16-beam sensor, 20–31% of scans), Office_Mitte_1 (a few scans at a critical moment) | Few rows or poor intensity texture; with no motion the scan isn't deskewed. | **Range image as fallback** (already implemented, #058). | NTU **−15 to −42%**; Office_Mitte_1 1.44 ± 1.21 → **0.24 ± 0.06 m** (#069). Not yet tested on the other 21 sequences. | Few hours of runs; about 2× feature cost per scan |
| **3** | **Height drift in the odometry itself** | Bodleian (690 m, no loop closures): z drift of 1–4 m | Builds up where the ground changes by a few decimetres. Likely the paper config's coarse 1 m map (#068). | (a) Finer map for indoor/urban routes. (b) A vertical constraint (ground plane / gravity from MapClosures). | Stairs showed the finer map helps (#063); not yet tested on Bodleian. | (a) 1 run ≈ 1 h. (b) New code, days |
| **4** | **Fast rotation (≳ 10° per scan)** | dynamic_spinning (last 20 s at ~100°/s): KISS wins there, 0.16 vs 0.46 m | The intensity image motion becomes unreliable at that speed (#068). | (a) Range image as a **third** starting point (implemented, #058). (b) Prefer constant velocity when the rotation is very fast. | Fallback didn't help here (#069); the third start is untested. | (a) Runs only. (b) Small code change |
| **5** | **Rotation not better than no deskew on easy sequences** | Blenheim, Bodleian, observatory (1 s / 10 s rotation error about 0.3–0.6° higher) | About 0.7° of per-pose rotation noise that doesn't accumulate: the image's per-scan rotation error passes through the deskew (#068, hypothesis of #045). | (a) Deskew a second time with the ICP's own motion and register again. (b) Smooth or weight the image rotation. | (b) partly tested in #047: better rotation, slightly worse APE. (a) untested. | (a) Code change + runs |
| **6** | **UZH: no deskew is best** | Hilti UZH (0.20 m vs 0.50 m) | The image fails on half the scans; the fallback removes the failures but doesn't improve the result (#069). | Probably none specific. Worth a look at what the scene is (a dark, low-texture room?). | — | Analysis only |
| **7** | **Blenheim's repeated façades** | Blenheim | Consistently wrong intensity matches (#053) | Already handled by the two starting points; nothing more needed unless it recurs elsewhere. | — | — |

## Older ideas, parked

- Intensity only in the ICP's weak directions (corridors, #022–#024).
- Joining maps across floors (needs a suitable dataset).
- A RANSAC threshold that scales with range.
- Two-way checking of matches.

## For publication

- Numbers for other methods under the same official protocols.
- Remaining component tests: the floor filter on/off, σ fixed vs adaptive.
- Map sharpness with 4 seeds.
- A real-time measurement on a laptop.

## Recommendation, in order

1. **Problem 1 (pose-graph weights):** the biggest measured gain and the smallest change, and it fixes the main limitation on long routes.
2. **Problem 2 (range fallback on all 27):** a proven gain where the image fails. The remaining question is whether it hurts anywhere else.
3. **Problem 3(a) (Bodleian with the finer map):** one run to test the height-drift explanation. Only if it's confirmed would the costlier vertical constraint (3b) be worth building.
4. **Problems 4 and 5:** method research, after 1–3.
5. **Problem 6:** a quick look at the scene only; probably a limit of the method, not something to fix.
