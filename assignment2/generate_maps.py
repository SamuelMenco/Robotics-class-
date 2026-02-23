"""
Generate occupancy grid maps for Assignment 2: Path Planning.

Usage:
    python3 generate_maps.py

Produces map_1.npz, map_2.npz, map_3.npz in the current directory.
"""

import numpy as np


def make_map(rows, cols, resolution, origin, start, goal, obstacles):
    """
    Create an occupancy grid with rectangular obstacles.

    Parameters:
        rows, cols: grid dimensions
        resolution: meters per cell
        origin: (x, y) world position of cell (0, 0)
        start: (x, y) world start position
        goal: (x, y) world goal position
        obstacles: list of (r_min, r_max, c_min, c_max) in cell indices
    Returns:
        dict suitable for np.savez
    """
    grid = np.zeros((rows, cols), dtype=np.int8)
    for r_min, r_max, c_min, c_max in obstacles:
        grid[r_min:r_max, c_min:c_max] = 1
    return dict(
        grid=grid,
        resolution=np.array(resolution),
        origin=np.array(origin, dtype=np.float64),
        start=np.array(start, dtype=np.float64),
        goal=np.array(goal, dtype=np.float64),
    )


def main():
    np.random.seed(42)

    # --- Map 1: Simple wall with a gap ---
    rows, cols = 50, 50
    res = 0.1  # 5m x 5m world
    origin = (0.0, 0.0)
    obstacles = [
        (20, 30, 0, 20),   # wall from left side
        (20, 30, 25, 50),  # wall continues after gap
    ]
    m = make_map(rows, cols, res, origin,
                 start=(0.5, 0.5), goal=(4.5, 4.5), obstacles=obstacles)
    np.savez("map_1.npz", **m)
    print("Created map_1.npz: simple wall with gap (50x50, 0.1 m/cell)")

    # --- Map 2: Maze-like environment ---
    rows, cols = 80, 80
    res = 0.1  # 8m x 8m
    origin = (0.0, 0.0)
    obstacles = [
        (15, 20, 0, 55),
        (30, 35, 25, 80),
        (45, 50, 0, 55),
        (60, 65, 25, 80),
        (0, 80, 0, 2),     # left wall
        (0, 80, 78, 80),   # right wall
        (0, 2, 0, 80),     # top wall
        (78, 80, 0, 80),   # bottom wall
    ]
    m = make_map(rows, cols, res, origin,
                 start=(0.5, 0.5), goal=(7.5, 7.5), obstacles=obstacles)
    np.savez("map_2.npz", **m)
    print("Created map_2.npz: maze-like (80x80, 0.1 m/cell)")

    # --- Map 3: Cluttered random obstacles ---
    rows, cols = 100, 100
    res = 0.1  # 10m x 10m
    origin = (0.0, 0.0)
    grid = np.zeros((rows, cols), dtype=np.int8)
    # Add random rectangular obstacles
    for _ in range(30):
        w = np.random.randint(3, 10)
        h = np.random.randint(3, 10)
        r = np.random.randint(0, rows - h)
        c = np.random.randint(0, cols - w)
        grid[r:r+h, c:c+w] = 1
    # Clear start and goal regions
    grid[0:8, 0:8] = 0
    grid[92:100, 92:100] = 0
    np.savez("map_3.npz",
             grid=grid,
             resolution=np.array(res),
             origin=np.array(origin, dtype=np.float64),
             start=np.array([0.3, 0.3], dtype=np.float64),
             goal=np.array([9.7, 9.7], dtype=np.float64))
    print("Created map_3.npz: cluttered random (100x100, 0.1 m/cell)")


if __name__ == "__main__":
    main()
