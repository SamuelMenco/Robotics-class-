"""
Evaluate a submitted path for Assignment 2: Path Planning.

Usage:
    python3 evaluator.py 1 astar
    python3 evaluator.py 1 rrt

Loads map_1.npz and map_1_astar.npz (or map_1_rrt.npz) and checks:
  1. Path connects start to goal (within tolerance)
  2. Path is collision-free
  3. Total path length
"""

import numpy as np
import argparse


def load_map(prefix, num):
    fname = f"{prefix}{num}.npz"
    data = np.load(fname)
    return {
        "grid": data["grid"],
        "resolution": float(data["resolution"]),
        "origin": data["origin"],
        "start": data["start"],
        "goal": data["goal"],
    }


def load_path(prefix, num, method):
    fname = f"{prefix}{num}_{method}.npz"
    data = np.load(fname)
    return data["path"]


def world_to_grid(point, origin, resolution):
    col = int(round((point[0] - origin[0]) / resolution))
    row = int(round((point[1] - origin[1]) / resolution))
    return row, col


def check_segment_collision(p1, p2, grid, origin, resolution):
    """
    Check whether the line segment from p1 to p2 passes through
    any occupied cell by sampling at sub-cell resolution.
    Returns True if collision detected, False if clear.
    """
    dist = np.linalg.norm(p2 - p1)
    if dist < 1e-12:
        r, c = world_to_grid(p1, origin, resolution)
        rows, cols = grid.shape
        if 0 <= r < rows and 0 <= c < cols:
            return grid[r, c] == 1
        return True  # out of bounds

    n_samples = max(int(np.ceil(dist / (resolution * 0.5))), 2)
    for t in np.linspace(0, 1, n_samples):
        pt = p1 + t * (p2 - p1)
        r, c = world_to_grid(pt, origin, resolution)
        rows, cols = grid.shape
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return True
        if grid[r, c] == 1:
            return True
    return False


def evaluate(m, path, tol=0.3):
    """
    Evaluate a path against a map.

    Returns a dict with:
      - connects_start: bool
      - connects_goal: bool
      - collision_free: bool
      - n_collisions: int (number of segments in collision)
      - path_length: float (meters)
      - n_waypoints: int
    """
    grid = m["grid"]
    res = m["resolution"]
    origin = m["origin"]

    # Check start/goal connectivity
    start_dist = np.linalg.norm(path[0] - m["start"])
    goal_dist = np.linalg.norm(path[-1] - m["goal"])
    connects_start = start_dist <= tol
    connects_goal = goal_dist <= tol

    # Check collisions
    n_collisions = 0
    for i in range(len(path) - 1):
        if check_segment_collision(path[i], path[i+1], grid, origin, res):
            n_collisions += 1

    # Path length
    diffs = np.diff(path, axis=0)
    lengths = np.linalg.norm(diffs, axis=1)
    path_length = np.sum(lengths)

    return {
        "connects_start": connects_start,
        "connects_goal": connects_goal,
        "collision_free": n_collisions == 0,
        "n_collisions": n_collisions,
        "path_length": path_length,
        "n_waypoints": len(path),
        "start_distance": start_dist,
        "goal_distance": goal_dist,
    }


def plot_result(m, path, title="Path"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    grid = m["grid"]
    res = m["resolution"]
    origin = m["origin"]
    rows, cols = grid.shape

    extent = [
        origin[0] - res / 2,
        origin[0] + (cols - 0.5) * res,
        origin[1] - res / 2,
        origin[1] + (rows - 0.5) * res,
    ]
    ax.imshow(grid, cmap="Greys", origin="lower", extent=extent, vmin=0, vmax=1)
    ax.plot(path[:, 0], path[:, 1], "b-", linewidth=1.5, label="Path")
    ax.plot(*m["start"], "go", markersize=12, label="Start")
    ax.plot(*m["goal"], "r*", markersize=15, label="Goal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal")
    return fig


def main():
    parser = argparse.ArgumentParser(description="Evaluate a path planning solution")
    parser.add_argument("num", type=int, help="Map number N")
    parser.add_argument("method", type=str, choices=["astar", "rrt"],
                        help="Method name (astar or rrt)")
    parser.add_argument("--prefix", type=str, default="map_")
    parser.add_argument("--plot", action="store_true", help="Show plot")
    args = parser.parse_args()

    m = load_map(args.prefix, args.num)
    path = load_path(args.prefix, args.num, args.method)

    print(f"Evaluating: map_{args.num}_{args.method}.npz")
    print(f"  Path shape: {path.shape}")

    results = evaluate(m, path)

    print(f"\n  Connects start: {results['connects_start']} "
          f"(distance: {results['start_distance']:.4f} m)")
    print(f"  Connects goal:  {results['connects_goal']} "
          f"(distance: {results['goal_distance']:.4f} m)")
    print(f"  Collision free: {results['collision_free']} "
          f"({results['n_collisions']} collisions)")
    print(f"  Path length:    {results['path_length']:.4f} m")
    print(f"  Waypoints:      {results['n_waypoints']}")

    if results["connects_start"] and results["connects_goal"] and results["collision_free"]:
        print("\n  PASS: Valid path found.")
    else:
        print("\n  FAIL: Path does not meet requirements.")

    if args.plot:
        import matplotlib.pyplot as plt
        plot_result(m, path, title=f"Map {args.num} - {args.method.upper()}")
        plt.show()


if __name__ == "__main__":
    main()
