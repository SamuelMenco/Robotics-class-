# Assignment 2 Planning Scripts

This folder contains implementations for:
- 2D occupancy-grid planning (`A*` and `RRT`)
- 7-DOF Franka Panda joint-space planning (`RRT` with MuJoCo collision checks)

The scripts are configured with module-level variables, so the normal workflow is:
1. Open the target script.
2. Edit the parameters at the top (for example `MAP_NUMBER` / `PROBLEM_NUMBER`).
3. Run the script.

---

## Repository Files

### 2D planning
- `astar.py`: A* planner on occupancy grids (8-connected).
- `RRT_2D.py`: RRT planner in continuous `(x,y)` space with occupancy-grid collision checks.

### Franka planning
- `RRT_franka.py`: 7D joint-space RRT planner. It can optionally animate the trajectory by setting the "Animate_result" variable = True

---

## Output Files (Required Format)

### 2D A*
- File: `map_N_astar.npz`
- Key: `path`
- Shape: `M x 2`
- Coordinates: world `(x, y)` in meters

### 2D RRT
- File: `map_N_rrt.npz`
- Key: `path`
- Shape: `M x 2`
- Coordinates: world `(x, y)` in meters

### Saved plots
- The scripts also save plot images directly in this same folder.
- Example files: `map_1_astar.png`, `map_1_rrt.png`, `map_2_astar.png`, `map_2_rrt.png`, `map_3_astar.png`, `map_3_rrt.png`.

### Franka RRT
- File: `franka_N_path.npz`
- Key: `path`
- Shape: `K x 7`
- Coordinates: joint angles (radians)
- This folder also includes a recorded video of the Franka arm moving along a planned trajectory.
- The trajectory shown in that video was generated using the negative of the start and goal configurations from `generate_franka_problems.py`.

---

## Problem Numbering (Important)

- In **2D scripts** (`astar.py`, `RRT_2D.py`):  
  `MAP_NUMBER` is expected to be **1 to 3**.

- In **Franka script** (`RRT_franka.py`):  
  `PROBLEM_NUMBER` is **0 to 1** (0-based index into `franka_problems.npz` keys `start_0/goal_0`, `start_1/goal_1`).

Franka output filename is automatically converted to evaluator format:
- `PROBLEM_NUMBER = 0` -> `franka_1_path.npz`
- `PROBLEM_NUMBER = 1` -> `franka_2_path.npz`

---

## Setup and Dependencies

Install dependencies (using the local requirements file):

```bash
pip install -r REQUIREMENTS.txt
```

MuJoCo + an OpenGL-capable environment are required for visualization/animation.

---

## How to Run

### 1) Generate data (optional if files already exist)

```bash
python3 generate_maps.py
python3 generate_franka_problems.py
```

### 2) Visualize maps / problems

```bash
python3 loader.py 1
python3 franka_utils.py 1
```

### 3) Run scripts. Expample for 2D A*:

Edit in `astar.py`:
- `MAP_NUMBER` (1..3)
- `SAVE_PLOT` (`True/False`)

Then:

```bash
python3 astar.py
```
---

## Script Parameters You Can Modify

### `astar.py`
- `MAP_NUMBER`: map ID (1..3)
- `MAP_PREFIX`: file prefix (default `map_`)
- `SAVE_PLOT`: save `map_N_astar.png`

### `RRT_2D.py`
- `MAP_NUMBER`: map ID (1..3)
- `MAP_PREFIX`: file prefix
- `SAVE_PLOT`: save `map_N_rrt.png`
- `MAX_ITERS`: max expansions
- `STEP_SIZE`: growth step in meters
- `GOAL_TOL`: distance threshold to connect goal
- `RNG_SEED`: random seed. Used in this case mainly for repeatability and debugging
- `CLEARANCE_CELLS`: obstacle inflation radius (0 = none)
- `EDGE_SAMPLE_FRACTION`: edge collision sampling density

### `RRT_franka.py`
- `PROBLEM_NUMBER`: problem index (0..1)
- `MAX_ITERS`: max expansions
- `STEP_SIZE`: growth step in joint space (radians)
- `GOAL_TOL`: joint-space threshold to connect goal
- `RNG_SEED`: random seed
- `EDGE_CHECKS`: interpolation checks per edge
- `GOAL_BIAS`: probability of sampling goal directly
- `ANIMATE_RESULT`: animate planned path in MuJoCo viewer

---


