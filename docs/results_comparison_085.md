# Comparison with other methods — all sequences (#083–#085)

*Each dataset's official protocol (#061). APE in m; **bold** = best per sequence (among non-failures); ✗ = failure (APE > 5 m); — = not run (method cannot read the data: Traj-LO / FAST-LIO2 / DLO no .pcd for 01_short; COIN-LIO Ouster OS0-128 only; Traj-LO long experiment > 46 GB memory; MAD-ICP segfault on the long experiment). Ours: SLAM mean of 4 seeds; all other methods one run (deterministic). The other methods are odometry only — compare them with “Ours odo”. FAST-LIO2 and COIN-LIO use an IMU (reference, not competitors).*

### APE [m]

| dataset | sequence | Ours SLAM | Ours odo | KISS-SLAM | KISS no deskew | GenZ-ICP | MAD-ICP | DLO | CT-ICP | Traj-LO | FAST-LIO2 (IMU) | COIN-LIO (IMU) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Newer College 2020 | 01_short | 0.307 | 0.675 | 0.419 | 0.350 | 0.512 | 1.741 | — | **0.302** | — | — | — |
| Newer College 2021 | quad_easy | 0.079 | 0.079 | 0.104 | 0.083 | 0.077 | 0.090 | 0.082 | 0.073 | 0.070 | **0.067** | 0.068 |
| Newer College 2021 | stairs | 2.074 | 2.074 | 3.586 | 2.705 | 2.003 | **0.136** | 0.175 | 4.129 | 0.191 | *733* ✗ | 0.218 |
| Newer College 2021 | cloister | 0.188 | 0.188 | 0.396 | 0.480 | 0.152 | 0.936 | 0.186 | 0.393 | 0.061 | 0.105 | **0.053** |
| Newer College 2021 | math_easy | 0.108 | 0.108 | 0.160 | 0.104 | 0.098 | 0.087 | 0.166 | 0.097 | **0.083** | 0.094 | 0.093 |
| Newer College 2021 | underground_easy | 0.067 | 0.067 | 0.117 | 0.092 | 0.056 | 0.077 | 0.243 | 0.046 | **0.026** | 0.036 | 0.038 |
| Oxford Spires | christ-church-02 | 0.207 | 0.207 | 0.777 | 0.546 | **0.178** | 0.830 | 0.467 | *21.838* ✗ | 0.290 | 0.342 | — |
| Oxford Spires | christ-church-03 | 0.044 | 0.044 | 0.143 | 0.089 | 0.064 | 0.124 | 0.054 | 0.056 | **0.017** | 0.018 | — |
| Newer College 2021 | quad_hard | 0.218 | 0.218 | 0.329 | 0.209 | 0.120 | *5.407* ✗ | 0.134 | 0.054 | 0.053 | 0.066 | **0.050** |
| Newer College 2021 | math_medium | 0.152 | 0.152 | 0.253 | 0.164 | 0.146 | 0.178 | 0.816 | 0.139 | 0.116 | **0.104** | 0.115 |
| Newer College 2021 | underground_medium | 0.061 | 0.061 | 0.162 | 0.097 | 0.078 | 0.113 | 0.058 | 0.044 | **0.028** | 0.036 | 0.039 |
| Newer College 2021 | underground_hard | 0.086 | 0.086 | *12.780* ✗ | *12.341* ✗ | 0.105 | *8.910* ✗ | 0.568 | *9.594* ✗ | **0.050** | 0.053 | 0.054 |
| Oxford Spires | keble-college-03 | 0.094 | 0.094 | *9.757* ✗ | *11.370* ✗ | 0.333 | 0.358 | 0.338 | 0.090 | **0.053** | 0.065 | — |
| Oxford Spires | observatory-quarter-01 | 0.080 | 0.080 | 0.497 | 0.436 | 0.104 | 0.545 | 0.214 | 0.105 | **0.053** | 0.058 | — |
| Oxford Spires | blenheim-palace-02 | 0.268 | 0.268 | 0.293 | 0.205 | 0.317 | 0.539 | 0.485 | 0.303 | 0.228 | **0.151** | — |
| Oxford Spires | bodleian-library-02 | 0.545 | 0.545 | 1.911 | 1.363 | 0.657 | 2.077 | 1.694 | 0.520 | 0.877 | **0.247** | — |
| Newer College 2020 | 02_long_experiment | 1.619 | 1.378 | 1.275 | 3.498 | 1.893 | — | **0.339** | 0.483 | — | 0.342 | — |
| Newer College 2020 | dynamic_spinning | 0.504 | 0.504 | 0.159 | *20.751* ✗ | *15.596* ✗ | *26.169* ✗ | 4.450 | *10.761* ✗ | **0.080** | 0.085 | — |
| Hilti 2021 | Basement_1 | 0.050 | 0.050 | 0.055 | 0.078 | 0.069 | 0.106 | 0.083 | 0.066 | 0.040 | **0.030** | — |
| Hilti 2021 | IC_Office_1 | 0.071 | 0.071 | *6.344* ✗ | 1.655 | 0.069 | 0.941 | 0.208 | **0.061** | 0.062 | 0.074 | — |
| Hilti 2021 | Office_Mitte_1 | 0.241 | 0.241 | 4.286 | 0.575 | **0.117** | 0.176 | 0.120 | 0.126 | *1632* ✗ | 0.121 | — |
| Hilti 2021 | Construction_Site_1 | 0.048 | 0.048 | 0.063 | 0.062 | 0.032 | 0.168 | 0.120 | 0.034 | 0.027 | **0.023** | — |
| Hilti 2021 | LAB_Survey_2 | 0.036 | 0.036 | 0.062 | 0.050 | 0.036 | 0.035 | 0.076 | 0.038 | **0.026** | **0.026** | — |
| Hilti 2021 | UZH_Tracking_Area_Run_2 | 0.551 | 0.551 | 0.585 | 0.204 | 0.198 | 0.188 | 0.196 | 0.498 | 0.270 | **0.188** | — |
| NTU VIRAL | eee_01 | 1.737 | 1.737 | 2.678 | 2.363 | 1.597 | 1.503 | 0.220 | 0.234 | **0.082** | 0.087 | — |
| NTU VIRAL | eee_02 | 0.679 | 0.679 | 1.486 | 1.490 | 0.222 | 1.271 | 0.149 | 0.096 | 0.075 | **0.072** | — |
| NTU VIRAL | eee_03 | 0.344 | 0.344 | 0.864 | 0.841 | 0.739 | 2.477 | 0.226 | 0.287 | 0.111 | **0.111** | — |

### RTE [%] (KITTI 100–800 m; NCD / Spires with ≥ 100 m of path)

| dataset | sequence | Ours SLAM | Ours odo | KISS-SLAM | KISS no deskew | GenZ-ICP | MAD-ICP | DLO | CT-ICP | Traj-LO | FAST-LIO2 (IMU) | COIN-LIO (IMU) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Newer College 2020 | 01_short | 0.74 | 0.78 | 0.92 | 0.62 | **0.60** | 0.80 | — | 0.72 | — | — | — |
| Newer College 2021 | quad_easy | 0.27 | 0.27 | 0.28 | 0.28 | 0.28 | 0.26 | 0.28 | 0.24 | 0.25 | 0.23 | **0.23** |
| Newer College 2021 | cloister | 0.30 | 0.30 | 0.77 | 0.63 | 0.34 | 0.69 | 0.38 | 0.56 | 0.20 | 0.22 | **0.16** |
| Newer College 2021 | math_easy | 0.53 | 0.53 | 0.92 | 0.42 | 0.42 | 0.38 | 0.49 | 0.48 | **0.37** | 0.41 | 0.41 |
| Newer College 2021 | underground_easy | 0.27 | 0.27 | 0.48 | 0.30 | 0.29 | 0.36 | 0.59 | 0.26 | **0.22** | 0.23 | 0.23 |
| Oxford Spires | christ-church-02 | 0.32 | 0.32 | 1.08 | 0.54 | 0.26 | 0.55 | 0.48 | 9.90 | **0.19** | 0.24 | — |
| Oxford Spires | christ-church-03 | 0.13 | 0.13 | 0.50 | 0.15 | 0.11 | 0.21 | 0.10 | 0.15 | 0.07 | **0.05** | — |
| Newer College 2021 | quad_hard | 0.89 | 0.89 | 1.35 | 1.10 | 0.69 | 5.76 | 0.83 | 0.66 | 0.62 | 0.68 | **0.57** |
| Newer College 2021 | math_medium | 0.98 | 0.98 | 1.44 | 0.76 | 0.72 | 0.71 | 1.42 | 0.81 | 0.69 | 0.73 | **0.67** |
| Newer College 2021 | underground_medium | 0.22 | 0.22 | 0.41 | 0.28 | 0.25 | 0.34 | 0.23 | 0.21 | **0.18** | 0.20 | 0.19 |
| Newer College 2021 | underground_hard | 0.25 | 0.25 | 14.83 | 24.29 | 0.30 | 7.34 | 1.15 | 8.15 | **0.16** | 0.17 | 0.16 |
| Oxford Spires | keble-college-03 | 0.69 | 0.69 | 26.33 | 32.48 | 0.70 | 0.72 | 0.78 | 0.42 | 0.20 | **0.19** | — |
| Oxford Spires | observatory-quarter-01 | 0.28 | 0.28 | 0.85 | 0.62 | 0.19 | 0.42 | 0.37 | 0.24 | 0.10 | **0.10** | — |
| Oxford Spires | blenheim-palace-02 | 0.69 | 0.69 | 0.83 | 0.35 | 0.42 | 0.48 | 0.79 | 0.33 | **0.20** | 0.27 | — |
| Oxford Spires | bodleian-library-02 | 0.99 | 0.99 | 2.48 | 0.89 | 0.47 | 1.36 | 1.03 | 1.10 | 0.56 | **0.27** | — |
| Newer College 2020 | 02_long_experiment | 0.99 | 0.82 | 1.07 | 1.25 | 0.69 | — | **0.67** | 0.77 | — | 0.73 | — |

### RPE 1 m translation [cm] (NCD / Spires)

| dataset | sequence | Ours SLAM | Ours odo | KISS-SLAM | KISS no deskew | GenZ-ICP | MAD-ICP | DLO | CT-ICP | Traj-LO | FAST-LIO2 (IMU) | COIN-LIO (IMU) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Newer College 2020 | 01_short | 10.7 | 10.7 | 23.6 | 10.6 | 8.1 | **5.6** | — | 8.1 | — | — | — |
| Newer College 2021 | quad_easy | 5.8 | 5.8 | 12.1 | 6.5 | 5.3 | 5.0 | 5.0 | 3.2 | 2.5 | 2.1 | **2.0** |
| Newer College 2021 | stairs | 34.7 | 34.7 | 147 | 104 | 33.5 | **3.7** | 5.3 | 266 | 13.9 | 5402 | 5.5 |
| Newer College 2021 | cloister | 6.4 | 6.4 | 19.5 | 12.9 | 6.9 | 4.0 | 5.7 | 6.5 | 2.7 | 2.7 | **2.1** |
| Newer College 2021 | math_easy | 8.1 | 8.1 | 14.1 | 5.5 | 4.6 | **3.6** | 4.2 | 6.6 | 4.4 | 4.3 | 5.2 |
| Newer College 2021 | underground_easy | 4.8 | 4.8 | 11.4 | 6.5 | 4.2 | 3.3 | 3.8 | 3.7 | 2.2 | **2.1** | 2.1 |
| Oxford Spires | christ-church-02 | 5.8 | 5.8 | 27.5 | 13.0 | 7.9 | 5.2 | 6.9 | 14.8 | 1.9 | **0.8** | — |
| Oxford Spires | christ-church-03 | 3.8 | 3.8 | 16.3 | 8.0 | 5.8 | 4.3 | 5.2 | 2.9 | 1.4 | **1.0** | — |
| Newer College 2021 | quad_hard | 8.2 | 8.2 | 20.9 | 14.6 | 10.7 | 15.6 | 11.2 | 5.3 | 4.2 | **3.4** | 3.6 |
| Newer College 2021 | math_medium | 12.9 | 12.9 | 21.5 | 10.0 | 8.6 | **6.8** | 8.4 | 10.8 | 7.8 | 7.5 | 8.9 |
| Newer College 2021 | underground_medium | 4.8 | 4.8 | 15.5 | 9.1 | 6.4 | 4.8 | 5.7 | 3.9 | 2.6 | 2.4 | **2.4** |
| Newer College 2021 | underground_hard | 5.7 | 5.7 | 242 | 110 | 8.2 | 8.4 | 7.7 | 81.1 | 3.4 | 3.0 | **2.7** |
| Oxford Spires | keble-college-03 | 8.5 | 8.5 | 60.5 | 25.2 | 12.1 | 6.5 | 7.0 | 3.8 | 1.5 | **0.9** | — |
| Oxford Spires | observatory-quarter-01 | 5.9 | 5.9 | 23.2 | 10.1 | 7.9 | 5.6 | 6.6 | 3.0 | 1.1 | **0.7** | — |
| Oxford Spires | blenheim-palace-02 | 8.4 | 8.4 | 21.3 | 8.7 | 5.6 | 4.2 | 4.7 | 3.0 | 1.3 | **0.6** | — |
| Oxford Spires | bodleian-library-02 | 9.7 | 9.7 | 63.7 | 16.5 | 11.6 | 7.4 | 9.8 | 4.2 | 1.5 | **0.7** | — |
| Newer College 2020 | 02_long_experiment | 11.6 | 11.6 | 27.7 | 11.6 | 9.8 | — | **7.5** | 9.8 | — | 7.9 | — |
| Newer College 2020 | dynamic_spinning | 20.4 | 20.4 | 19.2 | 1023 | 297 | 217 | 99.7 | 345 | **11.2** | 12.1 | — |

### Summary per method

| method | sequences | median APE [m] | worst APE [m] | failures (APE > 5 m) | median RPE 1 m [cm] | median RTE [%] | median RRE [°/100 m] |
|---|---|---|---|---|---|---|---|
| Ours SLAM | 27 | 0.188 | 2.074 | 0 | 8.1 | 0.43 | 1.07 |
| Ours odo | 27 | 0.188 | 2.074 | 0 | 8.1 | 0.43 | 1.03 |
| KISS-SLAM | 27 | 0.419 | 12.780 ✗ | 3 | 21.4 | 0.92 | 2.38 |
| KISS no deskew | 27 | 0.436 | 20.751 ✗ | 3 | 11.1 | 0.62 | 1.36 |
| GenZ-ICP | 27 | 0.146 | 15.596 ✗ | 1 | 8.0 | 0.38 | 0.91 |
| MAD-ICP | 26 | 0.449 | 26.169 ✗ | 3 | 5.2 | 0.55 | 1.02 |
| DLO | 26 | 0.202 | 4.450 | 0 | 6.6 | 0.59 | 1.01 |
| CT-ICP | 27 | 0.126 | 21.838 ✗ | 3 | 5.9 | 0.52 | 1.08 |
| Traj-LO | 25 | 0.070 | 1631.670 ✗ | 1 | 2.5 | 0.20 | 0.74 |
| FAST-LIO2 (IMU) | 26 | 0.080 | 732.504 ✗ | 1 | 2.4 | 0.23 | 0.88 |
| COIN-LIO (IMU) | 9 | 0.054 | 0.218 | 0 | 2.7 | 0.23 | 1.02 |
