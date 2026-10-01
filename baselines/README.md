# Other methods for the comparison of the paper (#083)

Everything needed to rebuild and re-run the comparison methods of `docs/experiment_log.md` #083. Runs are driven by
`scripts/run_baseline.py` (GenZ-ICP, MAD-ICP, Traj-LO, CT-ICP) and `baselines/lio_docker` (FAST-LIO2, COIN-LIO); trajectories
go to `~/kiss_runs_ssd/<sequence>/<arm>_s0` and are scored by `scripts/results_table.py --all-arms --runs=...` like our arms.

| method | source (pinned) | how it runs | our changes |
|---|---|---|---|
| GenZ-ICP | pip `genz-icp` 0.3.2 | `run_baseline.py genz`, our readers | none (pretuned configs of the authors) |
| MAD-ICP | pip `mad-icp` 0.0.10 (built with `scikit-build-core<0.10`) | `run_baseline.py mad`, our readers | a scan with < 100 points keeps the last pose |
| Traj-LO | github kevin2431/Traj-LO `ba273d3` | `run_baseline.py trajlo`, its own bag reader | `patches/traj-lo_ba273d3.patch`: headless runner; a folder of bags = one sequence; Ouster point time = header + t (as our reader; upstream header - 0.1 s + t) |
| CT-ICP | github jedeschaud/ct_icp `d467813` (superbuild, `CXXFLAGS=-include cstdint`) | `run_baseline.py cticp`, our readers streamed through a pipe | `patches/ct_icp_d467813.patch`: `stream_odometry` runner (profile robust_low_inertia / driving); SIGSTKSZ fix for glibc >= 2.34 |
| FAST-LIO2 | github hku-mars/FAST_LIO `7cc4175` | `lio_docker`, ROS Noetic, bags at real time | `patches/fast_lio_7cc4175.patch`: Hesai reader (`lidar_type: 5`, through the Velodyne path) |
| DLO | github vectr-ucla/direct_lidar_odometry `11528c0` | `lio_docker` (image `kiss-lio:3`), bags at real time, `scripts/lio_to_tum.py --raw-scan` | `patches/dlo_11528c0.patch`: `imu: false` (LiDAR only, #085); RViz optional |
| COIN-LIO | github ethz-asl/COIN-LIO `76729cc` | `lio_docker`, its `mapping_newer_college.launch` | none (Ouster only: NCD 2021) |

`lio_docker/`: `Dockerfile` (image `kiss-lio:2`: `ros:noetic-perception` + Livox SDK / livox_ros_driver (HEAD, message types only) +
COIN-LIO + FAST-LIO2, copied from local clones next to it), `run_in_container.sh` (starts the method, plays the bags with `--clock`,
records `/Odometry`), `cfg/` (FAST-LIO2 per dataset: the authors' `ouster64.yaml` / `velodyne.yaml` and launch parameters with the
dataset's own LiDAR-IMU calibration and `pcd_save_en: false`; `fastlio_ncd2021_blind1.yaml` = blind 1 m instead of 4 m, stairs only).
`scripts/lio_to_tum.py` turns `/Odometry` (IMU pose at the scan end) into the LiDAR pose with the config's extrinsic.
bz2-compressed sequences (Oxford Spires, Hilti 2021) are first reduced to their LiDAR + IMU topics with `scripts/extract_lio_topics.py`
(rosbag play cannot decompress them at real time once a node subscribes).

Clone the sources at the pinned commits into `/home/photogrammetry/baselines/` and apply the patch of each with `git apply` (FAST-LIO2:
`patch -p1` inside `lio_docker/FAST_LIO` — the patch is a `diff -ruN` of `src/`).
