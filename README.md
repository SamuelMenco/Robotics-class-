# Robotics-class-
Repository created to upload the assignments for ME676 at UK

# Assignment 1

## Project Overview
This repository implements scan‑matching SLAM using ICP, loop‑closure selection, and Pose Graph Optimization (PGO). The pipeline produces estimated poses, evaluation metrics, and map visualizations.

## Scripts

**`loader.py`**  
Loads scans/poses from the simulator `.npz` files and visualizes them.

**`ICP.py`**  
Runs ICP between consecutive scans, computes relative transforms, composes them into world poses, and saves the initial pose trajectory used for evaluation.

**`loop_closure.py`**  
Selects loop‑closure candidates, validates them with ICP, and **writes all pose‑graph data** to `pose_graph_data.npz`.  
This file includes:
- `world_poses` (initial pose estimates)
- consecutive edges (`edge_i`, `edge_j`, `edge_R`, `edge_t`)
- loop‑closure edges (`lc_i`, `lc_j`, `lc_R`, `lc_t`)

**`pgo.py`**  
Loads `pose_graph_data.npz`, builds residuals/Jacobians, runs Gauss‑Newton PGO, and saves optimized poses to `sim_25_slam.npz` (or `sim_30_slam.npz`).  
It can also plot the environment map before/after PGO.

**`evaluator.py`**  
Evaluates the estimated poses against ground truth using successive relative pose errors and final pose error.

**`test.py`**  
Quick visualization of the estimated trajectory only; saves `estimated_trajectory.png`.

## Output Files

### Pose Files
- `sim_25_slam.npz` (or `sim_30_slam.npz`)  
  Estimated poses after PGO.  
  Saved as keys `"0"`, `"1"`, … with `[x, y, theta]`.

### Pose‑Graph Data
- `pose_graph_data.npz`  
  Pose estimates and edges used by PGO.

### Images
- `robot_environment_map_sim_25.png` / `robot_environment_map_sim_30.png`  
  Map before/after PGO (from `pgo.py`).
- `estimated_trajectory.png`  
  Estimated trajectory only (from `test.py`).
