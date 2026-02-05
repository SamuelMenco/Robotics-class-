# Robotics-class-
Repository created to upload the assignments for ME676 at UK

# Assignment 1
Please find the required files and the scripts used for the assignment 1 solution in the "assignment1" directory

## Project Overview
This repository implements scan‑matching SLAM using ICP, loop‑closure selection, and Pose Graph Optimization (PGO). The pipeline produces estimated poses and map visualizations.

## Scripts


**`ICP.py`**  
Implements ICP between consecutive scans, composes transforms into world poses, and provides helpers for plotting.

**`loop_closure.py`**  
Selects loop‑closure candidates, validates them with ICP, and **writes all pose‑graph data** to `pose_graph_data.npz`.  
This file includes:
- `world_poses` (initial pose estimates)
- consecutive edges (`edge_i`, `edge_j`, `edge_R`, `edge_t`)
- loop‑closure edges (`lc_i`, `lc_j`, `lc_R`, `lc_t`)

**`pgo.py`**  
Loads `pose_graph_data.npz`, builds residuals/Jacobians, runs Gauss‑Newton PGO, and saves optimized poses to `sim_25_slam.npz` (or `sim_30_slam.npz`).  
It can also plot the environment map before/after PGO.

**`test.py`**  
Quick visualization of the estimated trajectory only; saves `estimated_trajectory.png`.

## Output Files

### Pose Files
- `sim_25_slam.npz` (or `sim_30_slam.npz`)  
  Estimated poses after PGO.  

### Pose‑Graph Data
- `pose_graph_data.npz`  
  Pose estimates from Scan matching and edges later used by PGO.

### Images
- `robot_environment_map_sim_25.png` / `robot_environment_map_sim_30.png`  
  Map before/after PGO (from `pgo.py`).
- `estimated_trajectory.png`  
  Estimated trajectory only (from `test.py`).
