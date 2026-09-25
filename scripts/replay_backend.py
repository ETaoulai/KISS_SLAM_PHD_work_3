#!/usr/bin/env python3
"""Replay the KISS-SLAM back end of a finished run offline, with other loop-closure settings (open_tasks B.1, #067).

    python scripts/replay_backend.py <run dir> <variant> [<variant> ...] [--out=/home/photogrammetry/kiss_runs/backend_replay]

A run saves everything the back end used: the per-scan odometry (edges of <run>/*/trajectory.g2o) and the node graph with
its odometry and loop-closure edges (<run>/*/local_maps/local_map_graph.g2o).  This rebuilds the trajectory for a
variant of the node graph without running the odometry again:
  1. per-scan odometry chained from the first scan, P_i;
  2. the first scan of every node: node k+1 starts where the chained odometry from the start of node k reaches the
     node-to-node odometry edge k -> k+1 (the local trajectory of a node starts at the identity);
  3. the node graph rebuilt INCREMENTALLY as KissSLAM builds it (a node, its odometry edge, the closures of the node
     just finished, a short optimisation — PoseGraphOptimizerConfig.max_iterations Dogleg iterations — then the next node
     from the corrected pose), from its odometry edges plus the loop-closure edges kept by the variant, node 0 fixed;
  4. every scan: K_k · P_{s_k}^{-1} · P_i with K_k the optimised pose of its node.
  5. KISS-SLAM's final per-scan smoothing (fine_grained_optimization), repeated exactly: after the node graph moved,
     consecutive nodes no longer join up, and this spreads the gap along the per-scan odometry edges (~1 mm each, metres
     in total).  Variant "all" must reproduce the run (checked, printed).

Variants:
  all            every accepted closure, information I6 (the run as it was)
  none           no closures (pure odometry)
  height=<m>     drop closures whose height difference disagrees with the odometry's by more than <m>, measured along the
                 z of the query node (the sensor is held roughly level; LoopCloser.max_height_disagreement uses the
                 MapClosures ground normal instead)
  planar=<w>     keep every closure but trust it only in x, y, yaw: information diag(1, 1, w, w, w, 1) in the query node
                 frame (g2o EdgeSE3 order x y z qx qy qz); a bird's-eye density match measures no height, roll or pitch
  iters=<n>      every closure, exactly as the run but with pose_graph_optimizer.max_iterations = n (default 10)
  rotw=<w>       every closure, incremental g2o exactly as the run, but every node-graph edge (odometry and closure) with
                 information diag(1, 1, 1, w, w, w): rotation weighted w times (g2o's rotation error is the quaternion vector,
                 ~theta/2, so identity information makes rotation errors ~4x cheaper than in radians)
  batch6         every closure, the same 6-DoF graph but optimised once to convergence (control for "4dof")
  4dof           every closure, 4-DoF pose graph as in VINS-Mono: per node x, y, z and yaw are optimised, roll and pitch stay
                 those of the odometry (in the frame of node 0); residuals Log(Z^-1 K_i^-1 K_j) of all edges, unit weights;
                 solved to convergence (SciPy), then the same per-scan smoothing

Output: <out>/<variant>/<sequence>/<run>/replay/<name>_poses_posetime_tum.txt, with the run's pose times, so that
scripts/evaluate_official.py scores it like any run.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_official import pose_times, write_tum  # noqa: E402

from kiss_slam.pose_graph_optimizer import PoseGraphOptimizer  # noqa: E402
from kiss_slam.config.config import PoseGraphOptimizerConfig  # noqa: E402

OUT = Path("/home/photogrammetry/kiss_runs/backend_replay")


def se3(v):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(v[3:7]).as_matrix()
    T[:3, 3] = v[:3]
    return T


def read_g2o(path):
    """(vertices {id: T}, edges [(v0, v1, T, info 6x6)], fixed ids).  EDGE_SE3:QUAT v0 v1: measurement = pose of v1 in v0."""
    V, E, fixed = {}, [], []
    for line in open(path):
        f = line.split()
        if not f:
            continue
        if f[0] == "VERTEX_SE3:QUAT":
            V[int(f[1])] = se3(np.array(f[2:9], float))
        elif f[0] == "EDGE_SE3:QUAT":
            u = np.array(f[10:31], float)
            info = np.zeros((6, 6))
            info[np.triu_indices(6)] = u
            info = info + info.T - np.diag(np.diag(info))
            E.append((int(f[1]), int(f[2]), se3(np.array(f[3:10], float)), info))
        elif f[0] == "FIX":
            fixed += [int(x) for x in f[1:]]
    return V, E, fixed


def chained_odometry(traj_g2o):
    _, E, _ = read_g2o(traj_g2o)
    step = {(i, j): T for i, j, T, _ in E}
    n = max(j for _, j in step) + 1
    P = [np.eye(4)]
    for k in range(1, n):
        P.append(P[-1] @ step[(k - 1, k)])
    return np.array(P)


def node_starts(P, node_odometry):
    """First scan of every node: where the chained odometry from the previous node start matches its node edge."""
    starts = [0]
    for T in node_odometry:
        s = starts[-1]
        rel = np.einsum("ij,njk->nik", np.linalg.inv(P[s]), P[s:])
        d = np.linalg.norm(rel[:, :3, 3] - T[:3, 3], axis=1)
        k = int(np.argmin(d))
        if d[k] > 0.05:
            raise RuntimeError(f"node edge not found in the scan odometry (closest {d[k]:.3f} m)")
        starts.append(s + k)
    return starts


def replay(run, variant):
    run = Path(run)
    Vn, En, _ = read_g2o(sorted(run.glob("*/local_maps/local_map_graph.g2o"))[-1])
    P = chained_odometry(sorted(run.glob("*/trajectory.g2o"))[-1])
    n_nodes = max(Vn) + 1
    odo = {min(i, j): (i, j, T) for i, j, T, _ in En if abs(i - j) == 1}
    node_odo = [odo[k][2] if odo[k][:2] == (k, k + 1) else np.linalg.inv(odo[k][2]) for k in range(n_nodes - 1)]
    closures = [(i, j, T, I) for i, j, T, I in En if abs(i - j) != 1]
    starts = node_starts(P, node_odo)
    K0 = [np.eye(4)]
    for T in node_odo:
        K0.append(K0[-1] @ T)                                      # node poses from odometry only: initial values

    kind, _, val = variant.partition("=")
    kept, report = [], []
    for i, j, T, I in closures:                                   # edge i -> j: pose of j in the frame of i (i = query)
        dz_closure, dz_odo = T[2, 3], (np.linalg.inv(K0[i]) @ K0[j])[2, 3]
        report.append((i, j, dz_closure - dz_odo))
        if kind == "none":
            continue
        if kind == "height" and abs(dz_closure - dz_odo) > float(val):
            continue
        if kind == "planar":
            I = np.diag([1.0, 1.0, float(val), float(val), float(val), 1.0])
        kept.append((i, j, T, I))

    if kind in ("batch6", "4dof"):
        K = batch_graph(K0, node_odo, kept, four_dof=kind == "4dof")
        return fine_grained(K, P, starts, n_nodes) + (report, len(kept))
    # Incremental, as KissSLAM.generate_new_node: node k+1 is added at (current pose of node k) · odometry edge, then the
    # closures whose query is node k, then (if any) a short optimisation that moves every node so far.
    pgo = PoseGraphOptimizer(PoseGraphOptimizerConfig(max_iterations=int(val)) if kind == "iters" else PoseGraphOptimizerConfig())
    pgo.add_variable(0, np.eye(4))
    pgo.fix_variable(0)
    K = {0: np.eye(4)}
    W = np.diag([1.0, 1.0, 1.0] + [float(val)] * 3) if kind == "rotw" else np.eye(6)
    for k, T in enumerate(node_odo):
        pgo.add_variable(k + 1, K[k] @ T)
        pgo.add_factor(k + 1, k, T, W)
        K[k + 1] = K[k] @ T
        new = [(i, j, C, I) for i, j, C, I in kept if i == k]
        for i, j, C, I in new:
            pgo.add_factor(j, i, C, W if kind == "rotw" else I)
        if new:
            pgo.optimize()
            K = dict(pgo.estimates())
    return fine_grained(K, P, starts, n_nodes) + (report, len(kept))


def se3_log(T):
    """(rho, phi) of T, 6-vector: translation part approximated by T's translation (small relative errors)."""
    return np.r_[T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_rotvec()]


def batch_graph(K0, node_odo, closures, four_dof):
    """Node poses minimising sum ||Log(Z^-1 K_i^-1 K_j)||^2 over odometry and closure edges, node 0 fixed.  4-DoF: node k
    is K_k = [Rz(yaw_k) · Rz(-yaw0_k) · R0_k, t_k], i.e. its roll and pitch (w.r.t. node 0) stay those of the odometry."""
    from scipy.optimize import least_squares
    n = len(K0)
    R0 = [K[:3, :3] for K in K0]
    yaw0 = [np.arctan2(R[1, 0], R[0, 0]) for R in R0]
    edges = [(k, k + 1, T) for k, T in enumerate(node_odo)] + [(i, j, C) for i, j, C, _ in closures]

    def poses(x):
        K = [np.eye(4)]
        for k in range(1, n):
            T = np.eye(4)
            if four_dof:
                t, yaw = x[4 * (k - 1):4 * (k - 1) + 3], x[4 * (k - 1) + 3]
                T[:3, :3] = Rotation.from_euler("z", yaw - yaw0[k]).as_matrix() @ R0[k]
            else:
                t, phi = x[6 * (k - 1):6 * (k - 1) + 3], x[6 * (k - 1) + 3:6 * k]
                T[:3, :3] = Rotation.from_rotvec(phi).as_matrix()
            T[:3, 3] = t
            K.append(T)
        return K

    def res(x):
        K = poses(x)
        return np.concatenate([se3_log(np.linalg.inv(Z) @ np.linalg.inv(K[i]) @ K[j]) for i, j, Z in edges])

    if four_dof:
        x0 = np.concatenate([np.r_[K0[k][:3, 3], yaw0[k]] for k in range(1, n)])
    else:
        x0 = np.concatenate([np.r_[K0[k][:3, 3], Rotation.from_matrix(R0[k]).as_rotvec()] for k in range(1, n)])
    return dict(enumerate(poses(least_squares(res, x0, xtol=1e-10, ftol=1e-10).x)))


def fine_grained(K, P, starts, n_nodes):
    # KissSLAM.fine_grained_optimization, repeated exactly: one vertex per scan at K_k · local, the per-scan odometry
    # edges (information I6, across node boundaries too), vertex 0 fixed and, per node, the vertex before its last one.
    # After the node graph moved, consecutive nodes no longer join up and this spreads the gap along the edges.
    bounds = starts + [len(P) - 1]
    fg = PoseGraphOptimizer(PoseGraphOptimizerConfig())
    fg.add_variable(0, K[0])
    fg.fix_variable(0)
    fixed = [0]
    for k in range(n_nodes):
        a, b = bounds[k], bounds[k + 1]
        for scan in range(a + 1, b + 1):
            fg.add_variable(scan, K[k] @ np.linalg.inv(P[a]) @ P[scan])
            fg.add_factor(scan, scan - 1, np.linalg.inv(P[scan - 1]) @ P[scan], np.eye(6))
        fg.fix_variable(b - 1)
        fixed.append(b - 1)
    fg.optimize()
    est = fg.estimates()
    poses = np.array([est[i] for i in range(len(P))])
    return poses, fixed


def main():
    out = Path(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")), OUT))
    args = [a for a in sys.argv[1:] if not a.startswith("--out=")]
    run, variants = Path(args[0]), args[1:]
    t, T_run, _ = pose_times(run, "none")
    for v in variants:
        poses, fixed, report, n_kept = replay(run, v)
        if v == "all":
            _, _, run_fixed = read_g2o(sorted(run.glob("*/trajectory.g2o"))[-1])
            print(f"{run.name}: fixed scans as in the run: {sorted(set(fixed)) == sorted(set(run_fixed))}; "
                  f"replay of the run itself, max |position - run| = "
                  f"{np.abs(poses[:, :3, 3] - T_run[:, :3, 3]).max():.3f} m")
        dest = out / v.replace("=", "") / run.parent.name / run.name / "replay"
        dest.mkdir(parents=True, exist_ok=True)
        write_tum(dest / f"{run.name}_poses_posetime_tum.txt", t, poses)
        print(f"{run.name} {v}: {n_kept}/{len(report)} closures kept; height disagreement per closure (query<-ref: m) "
              + ", ".join(f"{i}<-{j}: {d:+.2f}" for i, j, d in report))


if __name__ == "__main__":
    main()
