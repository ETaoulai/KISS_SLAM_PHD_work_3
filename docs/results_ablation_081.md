# Ablation of the method — #081 (30/9/26)

*Official score of each dataset (APE, m; Hilti: challenge APE; NTU: prism ATE), mean ± σ over 4 seeds. Each ablation changes ONE thing
in the method (two starts + range fallback, `surftworangefb`). **Bold** = best of the four per sequence. Runs: method in
`/home/photogrammetry/kiss_runs`, ablations on the external SSD (`~/kiss_runs_ssd`); full table `kiss_runs/results_official_allarms_081.md`,
paired tests `kiss_runs/comparisons_081.txt`. Entry #081 of `docs/experiment_log.md`. ⏳ until validated by Λ.Γ.*

## Paired comparison against the method (27 sequences; RPE: the 18 NCD / Spires sequences)

| ablation | APE: ablation wins – method wins | median APE change | p (Wilcoxon) | RPE 1 s translation: wins, change, p | verdict |
|---|---|---|---|---|---|
| − floor filter | 7 – 20 | +4.2 % | 0.003 | 3 – 15, +2.7 %, 0.005 | the filter helps (small, consistent) |
| adaptive σ (KISS default) | 15 – 12 | −1.8 % | 0.90 | 11 – 7, −0.2 %, 0.77 | tie on average; fixed σ protects NTU / long experiment / Office_Mitte_1 |
| image for deskew only | 6 – 21 | +8.4 % | 0.002 | 2 – 16, +9.0 %, 0.0001 | the image start is the largest part; failures up to 14 m |

## Per sequence (APE, m)

| dataset | sequence | method (two starts + range fallback) | − floor filter | adaptive σ (KISS default) | image for deskew only (ICP from constant velocity) |
|---|---|---|---|---|---|
| Newer College 2020 | 01_short | 0.307 ± 0.018 | 0.335 ± 0.034 | **0.299** ± 0.007 | 0.329 ± 0.050 |
| Newer College 2021 | quad_easy | **0.079** ± 0.001 | 0.082 ± 0.000 | 0.079 ± 0.000 | 0.080 ± 0.000 |
| Newer College 2021 | stairs | **2.074** ± 0.225 | 2.332 ± 0.609 | 2.200 ± 0.642 | 3.399 ± 0.352 |
| Newer College 2021 | cloister | **0.188** ± 0.013 | 0.190 ± 0.021 | 0.189 ± 0.015 | 0.267 ± 0.041 |
| Newer College 2021 | math_easy | 0.108 ± 0.001 | 0.112 ± 0.002 | **0.106** ± 0.003 | 0.114 ± 0.002 |
| Newer College 2021 | underground_easy | 0.067 ± 0.002 | 0.068 ± 0.001 | **0.058** ± 0.000 | 0.071 ± 0.004 |
| Oxford Spires | christ-church-02 | 0.207 ± 0.044 | 0.248 ± 0.089 | **0.185** ± 0.030 | 0.453 ± 0.116 |
| Oxford Spires | christ-church-03 | 0.044 ± 0.001 | 0.045 ± 0.004 | **0.042** ± 0.002 | 0.065 ± 0.004 |
| Newer College 2021 | quad_hard | 0.218 ± 0.012 | 0.241 ± 0.007 | **0.210** ± 0.019 | 0.212 ± 0.036 |
| Newer College 2021 | math_medium | **0.152** ± 0.001 | 0.163 ± 0.002 | 0.153 ± 0.002 | 0.155 ± 0.003 |
| Newer College 2021 | underground_medium | 0.061 ± 0.002 | 0.061 ± 0.003 | **0.058** ± 0.001 | 0.080 ± 0.014 |
| Newer College 2021 | underground_hard | 0.086 ± 0.003 | 0.089 ± 0.002 | **0.083** ± 0.003 | 14.215 ± 1.697 |
| Oxford Spires | keble-college-03 | 0.094 ± 0.001 | 0.101 ± 0.004 | **0.093** ± 0.002 | 4.426 ± 3.524 |
| Oxford Spires | observatory-quarter-01 | 0.080 ± 0.020 | **0.077** ± 0.005 | 0.078 ± 0.022 | 0.262 ± 0.115 |
| Oxford Spires | blenheim-palace-02 | 0.268 ± 0.015 | 0.385 ± 0.017 | 0.262 ± 0.020 | **0.253** ± 0.020 |
| Oxford Spires | bodleian-library-02 | 0.545 ± 0.030 | **0.487** ± 0.070 | 0.528 ± 0.145 | 0.955 ± 0.073 |
| Newer College 2020 | 02_long_experiment | **1.619** ± 0.761 | 1.695 ± 0.838 | 2.302 ± 0.204 | 2.244 ± 0.233 |
| Newer College 2020 | dynamic_spinning | 0.504 ± 0.263 | 0.547 ± 0.378 | **0.451** ± 0.082 | 6.103 ± 3.533 |
| Hilti 2021 | Basement_1 | 0.050 ± 0.008 | 0.043 ± 0.017 | 0.061 ± 0.011 | **0.043** ± 0.016 |
| Hilti 2021 | IC_Office_1 | 0.071 ± 0.006 | 0.117 ± 0.091 | **0.068** ± 0.005 | 7.248 ± 3.610 |
| Hilti 2021 | Office_Mitte_1 | 0.241 ± 0.060 | **0.210** ± 0.054 | 0.303 ± 0.034 | 1.490 ± 2.185 |
| Hilti 2021 | Construction_Site_1 | 0.048 ± 0.004 | 0.047 ± 0.004 | 0.045 ± 0.002 | **0.042** ± 0.006 |
| Hilti 2021 | LAB_Survey_2 | 0.036 ± 0.001 | **0.035** ± 0.000 | 0.036 ± 0.000 | 0.037 ± 0.001 |
| Hilti 2021 | UZH_Tracking_Area_Run_2 | **0.551** ± 0.000 | 0.565 ± 0.015 | 0.564 ± 0.015 | 0.577 ± 0.001 |
| NTU VIRAL | eee_01 | 1.737 ± 0.190 | 1.789 ± 0.118 | 1.983 ± 0.089 | **0.923** ± 0.201 |
| NTU VIRAL | eee_02 | 0.679 ± 0.064 | 0.710 ± 0.042 | 1.174 ± 0.104 | **0.640** ± 0.059 |
| NTU VIRAL | eee_03 | **0.344** ± 0.079 | 0.400 ± 0.041 | 0.495 ± 0.075 | 0.373 ± 0.092 |
