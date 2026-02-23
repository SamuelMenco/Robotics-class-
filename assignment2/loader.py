"""
Load and visualize an occupancy grid map for Assignment 2.

Usage:
    python3 loader.py 1

Loads map_1.npz and displays the occupancy grid with start/goal markers.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse


def load_map(prefix, num, suffix=".npz"):
    """Load an occupancy grid map from file."""
    fname = f"{prefix}{num}{suffix}"
    data = np.load(fname)
    return {
        "grid": data["grid"],
        "resolution": float(data["resolution"]),
        "origin": data["origin"],
        "start": data["start"],
        "goal": data["goal"],
    }


def world_to_grid(point, origin, resolution):
    """Convert world coordinates (x, y) to grid indices (row, col)."""
    col = int(round((point[0] - origin[0]) / resolution))
    row = int(round((point[1] - origin[1]) / resolution))
    return row, col


def grid_to_world(row, col, origin, resolution):
    """Convert grid indices (row, col) to world coordinates (x, y)."""
    x = origin[0] + col * resolution
    y = origin[1] + row * resolution
    return x, y


def plot_map(m, ax=None, title="Occupancy Grid"):
    """Plot the occupancy grid with start and goal."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    grid = m["grid"]
    res = m["resolution"]
    origin = m["origin"]
    rows, cols = grid.shape

    # Compute extent in world coordinates
    extent = [
        origin[0] - res / 2,
        origin[0] + (cols - 0.5) * res,
        origin[1] - res / 2,
        origin[1] + (rows - 0.5) * res,
    ]
    ax.imshow(grid, cmap="Greys", origin="lower", extent=extent, vmin=0, vmax=1)

    ax.plot(*m["start"], "go", markersize=12, label="Start")
    ax.plot(*m["goal"], "r*", markersize=15, label="Goal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal")
    return ax


def main():
    parser = argparse.ArgumentParser(description="Load and visualize a map")
    parser.add_argument("num", type=int, help="Map number N (loads map_N.npz)")
    parser.add_argument("--prefix", type=str, default="map_", help="File prefix")
    args = parser.parse_args()

    m = load_map(args.prefix, args.num)
    print(f"Map {args.num}: grid shape = {m['grid'].shape}, "
          f"resolution = {m['resolution']} m/cell")
    print(f"  Start: {m['start']}")
    print(f"  Goal:  {m['goal']}")
    print(f"  Occupied cells: {m['grid'].sum()} / {m['grid'].size}")

    plot_map(m, title=f"Map {args.num}")
    plt.show()


if __name__ == "__main__":
    main()
