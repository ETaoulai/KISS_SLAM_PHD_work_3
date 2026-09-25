# Request for a theoretical review: intensity-image motion estimation, deskew and point-to-point ICP in a KISS-SLAM derivative

You are asked to review, as a senior SLAM / estimation-theory expert, the mathematical formulation of a LiDAR-only SLAM
pipeline for a **hand-held** spinning LiDAR, and to analyse why it drifts **vertically** in featureless staircases and
narrow corridors. Everything below describes the code **as currently implemented** (KISS-SLAM + KISS-ICP 1.3.0 +
our additions). Where the implementation differs from the textbook formulation, the implementation is what is written.

> **Scope note (important).** Intensity is **not** part of the registration cost. There is no joint
> photometric–geometric objective and no weighting matrix that trades intensity against geometry in odometry. Intensity
> enters only through a **separate, decoupled** estimator (§2) whose output — the motion during the sweep — is used
> (a) to deskew the scan and (b) as the initial guess of a purely geometric point-to-point ICP (§4). An intensity–geometry
> blend exists in the code (Open3D ColoredICP, $\lambda_{geo}=0.9$) only in loop-closure verification of a legacy,
> disabled configuration; in the configuration under review, loop closures are verified with geometric point-to-plane ICP.
> The ICP and deskew are in C++ (KISS-ICP 1.3.0, unmodified); the intensity motion estimator is Python/SciPy; the pose
> graph is g2o (C++).

---

## 0. Sensors, frames, conventions

- Spinning LiDAR, 10 Hz ($T = 0.1$ s per sweep), 16–128 rings, 1024–2048 columns (Ouster OS0/OS1, Hesai QT64).
  Every point carries its own acquisition time $t_j$. The message stamp of scan $k$ is the time of its **first** point.
- Hand-held: per-scan rotation typically 1–1.5°, and the rotation *changes* between consecutive scans by nearly as much.
- $\mathrm{Exp}, \mathrm{Log}$: exponential/logarithm of $SO(3)$ or $SE(3)$ as stated; $[\cdot]_\times$ the skew operator.
- Poses $T\in SE(3)$ act on points as $T x = R x + t$.
- **No IMU, no gravity, no vertical prior** is used anywhere.

## 1. State and overall objective

Per scan $k$ the pipeline estimates, **sequentially and separately**:

1. $\hat\Delta_k \in SE(3)$ — the motion of the sensor during sweep $k$, as the pose of the end-of-sweep frame of
   scan $k$ in the end-of-sweep frame of scan $k-1$. It comes from intensity-image features (§2) and is **not** refined
   by the ICP.
2. $T_k \in SE(3)$ — the pose of the end-of-sweep sensor frame of scan $k$ in the frame of the current local map
   (node), from point-to-point ICP of the deskewed scan against a voxelised local map (§4).
3. Node poses $\{X_n\}\subset SE(3)$ from a pose graph over local maps with loop closures (§5), and then per-scan poses
   from a smoothing graph (§5).

There is **no joint objective** over $(\Delta_k, T_k)$. In effect:

$$
\hat x_k = \arg\min_x \sum_i \rho_{\text{soft-}\ell_1}\!\big(r^{img}_i(x)\big) \quad(\S 2), \qquad
\hat\Delta_k = P_{\hat x_k}(1),
$$
$$
\hat T_k = \arg\min_{T\in SE(3)} \sum_{j\in\mathcal C(T)} \rho_{GM}\!\big(\| T\,\tilde p_j - m_{c(j)}\|\big),\qquad
\tilde p_j = \mathcal D(p_j; \hat\Delta_k) \quad(\S 3, \S 4),
$$
with $\mathcal D$ the deskew operator and $m_{c(j)}$ the nearest map point.

---

## 2. Motion during the sweep from the intensity panorama (Python, `kiss_slam/intensity_deskew.py`)

### 2.1 Measurements
Each raw scan is rendered as a ring × azimuth panorama ($H$ = number of rings sorted by elevation, $W = 1024$ azimuth
bins; with 2048-column sensors two returns share a bin and the last one wins). Each pixel stores intensity (scaled to
0–255), the raw 3D point $x$ in the sensor frame at its own time, and its time $t$. SURF (or SIFT) keypoints are
detected on the 8× vertically up-sampled 8-bit image; matches between scans $k-1$ and $k$ use a Lowe ratio test of 0.75.
Each keypoint is lifted to $(x, t)$ by bilinear interpolation over the 4 surrounding pixels when all four are valid
and their ranges agree within 5 %, otherwise from the nearest valid pixel.

A match $i$ gives $p_i$ (point in scan $k-1$, at time $t_{p,i}$) and $q_i$ (in scan $k$, at time $t_{q,i}$). Times are
normalised to sweep periods from the start of scan $k$:
$$
\tau = \frac{t - t_{\text{start},k}}{T},\qquad \tau_{p}\in[-1,0),\ \ \tau_{q}\in[0,1).
$$

**Floor filter:** a match is dropped if $\|p_i - q_i\| < 0.05$ m **and** $q_i$ lies below −10° elevation and closer than
5 m (intensity patterns that travel with the sensor on the near floor).

### 2.2 Motion model (default "car": constant angular acceleration, constant linear velocity)
The parameters are $x = (\omega, v, \alpha)\in\mathbb R^9$ (units: per sweep period). The pose of the sensor at time
$\tau$ in the reference frame (sensor at $\tau=0$) is, on $SO(3)\times\mathbb R^3$ (**not** an $SE(3)$ exponential; rotation
and translation are decoupled):
$$
\phi(\tau) = \tau\,\omega + \tfrac{\tau^2}{2}\,\alpha,\qquad s(\tau) = \tau\, v,\qquad
P_x(\tau)\,y = \mathrm{Exp}\big(\phi(\tau)\big)\,y + s(\tau).
$$
(Alternatives in code: "cv", with $\alpha=0$, 6 parameters; "ca", with an additional $\tfrac{\tau^2}{2}a$ in $s$,
12 parameters.)

### 2.3 Residual
$$
r_i(x) = \mathrm{Exp}\big(\phi(\tau_{p,i})\big)\,p_i + s(\tau_{p,i}) \;-\; \mathrm{Exp}\big(\phi(\tau_{q,i})\big)\,q_i - s(\tau_{q,i}) \;\in\mathbb R^3 .
$$

### 2.4 Analytical Jacobian (as implemented)
With $y = \mathrm{Exp}(\phi)\,z$ and the left Jacobian of $SO(3)$
$$
J_l(\phi) = I + \frac{1-\cos\theta}{\theta^2}[\phi]_\times + \frac{\theta-\sin\theta}{\theta^3}[\phi]_\times^2,\qquad \theta=\|\phi\|
$$
(Taylor coefficients $\tfrac12-\tfrac{\theta^2}{24}$, $\tfrac16-\tfrac{\theta^2}{120}$ below $\theta = 10^{-4}$), the code uses
$$
\frac{\partial\,\mathrm{Exp}(\phi)z}{\partial\phi} = -[\,\mathrm{Exp}(\phi)z\,]_\times\, J_l(\phi),
$$
and, with $\sigma = +1$ for the $p$ term and $\sigma=-1$ for the $q$ term, summing both terms:
$$
\frac{\partial r_i}{\partial \omega} = \sum_{\sigma}\sigma\,\tau\,\big(-[y]_\times J_l(\phi(\tau))\big),\qquad
\frac{\partial r_i}{\partial \alpha} = \sum_{\sigma}\sigma\,\tfrac{\tau^2}{2}\,\big(-[y]_\times J_l(\phi(\tau))\big),\qquad
\frac{\partial r_i}{\partial v} = \sum_{\sigma}\sigma\,\tau\, I_3 .
$$

### 2.5 Robust cost, weighting and solver
- **No explicit weight matrix.** All matches have equal a-priori weight, regardless of range, incidence, ring, or
  feature scale. (The metric noise of a feature grows with range, since the angular noise is about one pixel,
  0.35°–1°.)
- SciPy `least_squares` (trust-region reflective), `loss="soft_l1"`, `f_scale = 0.05` m. **The loss is applied to each
  scalar component** of the stacked $3N$ residual vector, not to $\|r_i\|$:
  $\ \sum_i\sum_{c\in\{x,y,z\}} 2 f^2\big(\sqrt{1 + r_{i,c}^2/f^2} - 1\big)$. So the robustification is per-axis and not
  rotation invariant.
- **Initialisation and staging:** 3-point RANSAC with a rigid Kabsch fit that ignores the times (inlier:
  $\|p_i - (R q_i + t)\| < 0.30$ m, adaptive iteration count at confidence 0.999, at most 400, at least 10 inliers).
  $(\omega, v) \leftarrow (\mathrm{Log}\,R, t)$. Then 3 rounds of the "cv" fit with re-selection of inliers at
  $\|r_i\| < 0.10$ m, then 3 rounds of the "car" fit with $\alpha$ initialised at 0.
- Output: $\hat\Delta_k = P_{\hat x}(1)$, the motion over the later sweep. The curve $\phi(\tau)$ itself is **not** used
  by the deskew (see §3). If fewer than 10 inliers remain, the scan gets no motion (fallback, §3).
- Known empirical facts: matches saturate at 80–160 per scan pair; residual rotation error is about 0.5° per scan and
  systematic. The translation from near-field matches (< 5 m) is biased short (about 74 % of the truth, vs 100 % beyond
  12 m); a correction exists but is off.

## 3. Deskew and initial guess (KISS-ICP 1.3.0, C++ `Preprocessing.cpp`, unmodified)

For every raw point $x_j$ with time $t_j$:
$$
s_j = \frac{t_j - \min_\ell t_\ell}{\max_\ell t_\ell - \min_\ell t_\ell}\in[0,1],\qquad
\tilde x_j = \mathrm{Exp}_{SE(3)}\!\big((s_j - 1)\,\mathrm{Log}_{SE(3)}(\Delta)\big)\,x_j ,
$$
i.e. a **constant twist over the sweep**, expressed in the **end-of-sweep** frame ($s = 1$). Points outside
$(r_{\min}, r_{\max})$ are removed **after** deskewing. With our method, $\Delta = \hat\Delta_k$; upstream KISS uses the
constant-velocity guess $\Delta = T_{k-2}^{-1}T_{k-1}$.

Consequences worth noting:
- The estimator (§2) models angular **acceleration** on $SO(3)\times\mathbb R^3$, but the deskew uses only its endpoint,
  re-interpolated as a constant $SE(3)$ twist. The curvature is discarded, and so is the difference between the
  $SO(3)\times\mathbb R^3$ and $SE(3)$ geodesics. A per-point deskew along the fitted curve exists but is off; it gave no
  measurable gain.
- The two time normalisations differ: §2 uses $t_{\text{start}}$ of the valid points and the nominal period $T$; §3
  uses the min/max point times of the whole message.

**Initial guess of the ICP:** $T_k^{(0)} = T_{k-1}\hat\Delta_k$ (deskew and initial guess **must** agree; using the
image motion for one and constant velocity for the other diverges).

**Fallback:** when §2 fails, $\Delta = I$ (no deskew) and $T_k^{(0)} = T_{k-1}$.

**Two starting points (default):** if the rotation angle between $\hat\Delta_k$ and $\Delta_{cv}=T_{k-2}^{-1}T_{k-1}$
exceeds 5°, the ICP is run a second time from $(\Delta = I,\ T^{(0)} = T_{k-1}\Delta_{cv})$. The result kept is the one
with the smaller truncated mean nearest-neighbour distance to the local map,
$$
f(T) = \frac{1}{N}\sum_j \min\big(\| T\,\tilde p_j - \mathrm{NN}(T\,\tilde p_j)\|,\ 3\sigma\big)
$$
(a KD-tree over the local map points).

## 4. Registration: point-to-point ICP (KISS-ICP 1.3.0, C++ `Registration.cpp`, unmodified)

- **Source:** the deskewed scan, voxel-downsampled at $0.5v$, then at $1.5v$ for the ICP ($v$ = map voxel size:
  1.0 m in the paper configuration, i.e. $r_{\max}/100$ with $r_{\max} = 100$ m; 0.25 m in the "indoor detail"
  configuration).
- **Map:** voxel hash of size $v$, at most 20 points per voxel, a minimum spacing of $\sqrt{v^2/20}$ between points of
  a voxel, and points farther than $r_{\max}$ from the current sensor position removed. The local map is re-anchored
  at each new node.
- **Data association** (re-done every iteration): for the transformed source point $s_j = T\tilde p_j$, the nearest map
  point $m_j$ among the 27 neighbouring voxels, accepted if $\|s_j - m_j\| < 3\sigma$.
- **Residual and Jacobian** (left perturbation $T \leftarrow \mathrm{Exp}(\delta)\,T$, $\delta = (\rho,\theta)$,
  translation first):
$$
r_j = s_j - m_j\in\mathbb R^3,\qquad J_j = \big[\, I_3 \;\;\; -[s_j]_\times \,\big]\in\mathbb R^{3\times 6}.
$$
- **Weighting matrix:** isotropic, per correspondence $W_j = w_j I_3$ with the Geman–McClure-type weight
$$
w_j = \frac{\kappa^2}{\big(\kappa + \|r_j\|^2\big)^2},\qquad \kappa = \sigma .
$$
  Note that $\kappa$ (metres) is added to $\|r\|^2$ (square metres). No normals, no point-to-plane information, no
  intensity.
- **Solver:** Gauss–Newton, $H = \sum_j w_j J_j^\top J_j$, $g = \sum_j w_j J_j^\top r_j$,
  $\delta = -H^{-1}g$ via LDLT, **no damping, no eigenvalue check, no degeneracy handling**. Update
  $T\leftarrow \mathrm{Exp}_{SE(3)}(\delta)\,T$; at most 500 iterations, stop when $\|\delta\| < 10^{-4}$.
- **Threshold $\sigma$:** our method **fixes** $\sigma = 2.0$ m for the whole run (so $\kappa = 2$ m and the gate is
  6 m; $w = 1, 0.79, 0.44$ at $\|r\| = 0, 0.5, 1$ m, i.e. close to plain least squares). Upstream KISS adapts $\sigma$ from
  the deviation between prediction and result; with a good initial guess that collapses to about 0.6 m and the ICP
  then corrects too little.
- **Degeneracy:** only a diagnostic is logged, not used: eigenvalues $\lambda_1\le\lambda_2\le\lambda_3$ of the
  **covariance of the source points** (linearity, planarity, $\lambda_3/\lambda_1$). The Hessian $H$ is never inspected.

## 5. Back end (g2o, C++, and MapClosures)

- **Local maps (nodes):** a new node every 100 m of travel (paper configuration) or every 15 m (indoor detail). The
  node-to-node odometry edge has **information matrix $\Omega = I_6$**, with no covariance from the ICP.
- **Loop closures:** candidates from MapClosures (a bird's-eye-view density image of each ground-aligned local map,
  with feature matching and 2D RANSAC in the library). Each candidate is verified by Open3D point-to-plane ICP between
  local maps (correspondence threshold $\sqrt3\cdot 0.5$ m) and accepted if the voxel overlap exceeds 0.4. Loop edge
  $\Omega = I_6$. Optimiser: g2o Dogleg + Cholmod.
- **Fine-grained smoothing:** every per-scan pose is a vertex, with consecutive odometry edges ($\Omega = I_6$) and the
  last pose of each node fixed.
- **Vertical information:** none. Optional guards exist but are **off** by default: rejecting closures whose height
  disagrees with odometry (using the up axis from MapClosures' ground alignment), and splitting nodes by height.

---

## 6. The failure mode

Hand-held sensor, non-linear motion (walking with pitch/roll oscillation, turning, climbing stairs). Observed, with the
official protocols and 4 seeds where stated:

1. **Staircase (Newer College 2021 "stairs", 57 m).** Paper configuration ($v = 1$ m, $r_{\max} = 100$ m):
   - image motion + two starts: APE 2.07 ± 0.23 m, RPE 21.3 cm and 5.0° per second, path length overestimated by +64 %;
   - upstream KISS: APE 3.59 m, path +318 %; no deskew: 2.71 m.

   With $v = 0.25$ m, $r_{\max} = 50$ m, 15 m nodes: image motion + two starts APE 0.48 ± 0.05 m, RPE 6.4 cm and
   2.5° per second, path +10.5 %. No deskew, same configuration: 0.54 m, 9.9 cm and 2.9°, path +20 %.
2. **Long route (Oxford Spires Bodleian, 690 m).** The APE (0.5–0.8 m) is almost entirely vertical: z RMSE 0.55–0.94 m
   vs xy 0.12–0.17 m. The final height drift varies from −1.8 m to +9.8 m across seeds with otherwise identical
   inputs. Both arms (with and without the image motion) show 5–8 m of local drift in the first 30 s. The heading drift
   start-to-end is only −0.5° to +0.1°.
3. **Narrow corridor (Oxford Spires Christ Church ground floor).** Even with a perfect deskew (from the ground truth),
   the ICP error there is 7–10× larger than elsewhere, while the image motion measures the motion 2–4× better than the
   ICP.
4. **Systematic path-length overestimation:** +3 % to +24 % on most sequences (mean +13 %), not explained by the
   near-field scale bias of §2.
5. Over 16 sequences, the image motion lowers translation RPE by 38–42 % vs the no-deskew baseline (p ≤ 0.003) but
   **rotation RPE is a tie (8 vs 8 sequences)**.

Our working hypothesis: in treads/risers and corridors, the point-to-point Hessian is nearly rank-deficient along the
corridor axis and in pitch/vertical translation. The fixed, almost-L2 kernel ($\kappa = 2$ m) with coarse voxels and
isotropic weights lets the vertical/pitch component be driven by the initial guess, i.e. by $\hat\Delta_k$ and its
systematic rotation error. Nothing (gravity, $\Omega$, closures) then pulls the height back.

---

## 7. Questions for the reviewer

**A. Observability and degeneracy of the ICP (§4)**
1. For point-to-point ICP with $J_j = [I, -[s_j]_\times]$ and isotropic $W_j = w_j I$: characterise the eigenstructure
   of $H = \sum w_j J_j^\top J_j$ for (i) a staircase (alternating horizontal/vertical planes, periodic along one axis),
   and (ii) a long corridor of two parallel walls plus floor/ceiling. Which directions in $\mathfrak{se}(3)$ are weakly
   observable? How strongly do translation along the stair axis, vertical translation and pitch couple?
2. Does point-to-point with the nearest map point in 27 voxels (with $v = 1$ m and 20 points per voxel) *manufacture*
   constraints along the degenerate directions? That is, is the Fisher information $H$ over-confident relative to the
   true information of the surfaces, and how does that depend on $v$ and on the downsampling at $1.5v$?
3. Given that the solver never inspects $H$: which principled treatment would you recommend, and what are its
   failure modes for hand-held data? Candidates: solution remapping or eigenvalue truncation (Zhang et al.),
   Levenberg–Marquardt damping only in the weak subspace, or keeping the initial guess (§3) along the weak directions.
4. Is the Geman–McClure form $w = \kappa^2/(\kappa + e^2)^2$ with $\kappa$ in metres a sensible M-estimator? How does
   fixing $\kappa = 2$ m (nearly L2 over a 6 m gate) interact with degeneracy and with wrong associations on stair
   nosings?

**B. The intensity motion estimator (§2) and its coupling to the ICP**
5. For the model $\phi(\tau) = \tau\omega + \tfrac{\tau^2}{2}\alpha$, $s(\tau) = \tau v$ on $SO(3)\times\mathbb R^3$: derive
   the Fisher information of $x = (\omega, v, \alpha)$ given matches with times spread over $\tau_p\in[-1,0)$ and
   $\tau_q\in[0,1)$. Under which geometric and temporal distributions of matches are $\alpha$ and $\omega$ (or $v$ and
   $\omega$) confounded? What is the effect of 16-ring sensors (few rows), where features cluster in a few elevations?
6. The robust loss is applied per scalar component of $r_i$, not to $\|r_i\|$, and all matches get equal weight
   regardless of range. What bias and what loss of efficiency does this introduce? Derive a range- and bearing-aware
   information matrix $\Sigma_i^{-1}$ for $r_i$, given angular pixel noise and range noise.
7. The motion is **fixed** after §2: the ICP optimises only $T_k$ from $T_{k-1}\hat\Delta_k$ and never re-estimates the
   deskew. How do errors in $\hat\Delta_k$ (≈0.5° per scan of systematic rotation) propagate through (a) the deskew and
   (b) the initial guess into the weakly observable directions of §4? Is there a first-order expression for the induced
   vertical drift per metre of travel?
8. Is it consistent to estimate an angular-acceleration curve on $SO(3)\times\mathbb R^3$ and then deskew with a
   constant $SE(3)$ twist through the endpoint? What error does this mismatch induce per point, as a function of
   $\alpha$, range, and the screw component of the motion?

**C. Joint formulation and alternative parameterisations**
9. Formulate a joint estimator of $(\Delta_k, T_k)$ — or of a continuous-time trajectory — that combines the §2
   feature residuals and the §4 geometric residuals. How should $W$ be built so that the two residual types are
   commensurate, e.g. by normalising each by its own covariance so that $\chi^2$ is dimensionless?
   Which is preferable here and why:
   - a single $SE(3)$ twist per sweep,
   - $SO(3)\times\mathbb R^3$ with separate rates,
   - a cubic B-spline on $SE(3)$ or on $SO(3)\times\mathbb R^3$,
   - a Gaussian-process (white-noise-on-jerk) prior?
10. With a joint estimator, would the intensity residuals make the degenerate ICP directions of question 1 observable?
    Show when they would and when they cannot, e.g. a feature-poor intensity image in an unlit stairwell, or few rings.
11. The two-start rule selects between two ICP results by a truncated mean nearest-neighbour distance. Is this a
    consistent model-selection criterion when one hypothesis is deskewed and the other is not? What would a
    likelihood-ratio or information-based alternative look like?

**D. Vertical drift in the back end**
12. With all edges at $\Omega = I_6$ (no ICP covariance), no gravity, and 100 m nodes: how does the unobserved
    roll/pitch/z of a single odometry edge propagate into height error after $L$ metres? Would a per-edge
    $\Omega = H_{ICP}$ (or its conditioned version from question 3) materially change vertical consistency after loop
    closure?
13. Without an IMU, what is the most principled LiDAR-only vertical constraint? Candidates:
    - a ground-plane / gravity-direction factor from MapClosures' ground alignment per local map;
    - a planar-floor prior per storey;
    - a unary factor that penalises roll/pitch change between nodes.

    What are the risks on stairs, ramps and multi-storey buildings?
14. The path length is systematically overestimated (mean +13 %) while per-second translation errors are small. Which
    mechanisms in §§2–5 can produce a positive bias in path length without a bias in displacement? Candidates: jitter
    of a zero-mean error, association bias on stair treads, or the interpolation of §3. How would you test each
    analytically?

**Deliverables requested:** for each question, a short derivation or a pointer to the governing result, the predicted
signature in the data (so that we can test it), and a concrete recommendation ranked by expected benefit versus
implementation risk. Point out any errors in the formulations above.
