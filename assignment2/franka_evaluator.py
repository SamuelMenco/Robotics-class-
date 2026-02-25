"""
Evaluate a submitted Franka arm path for Assignment 2, Part 2.

Usage:
    python3 franka_evaluator.py 1
    python3 franka_evaluator.py 1 --animate

Loads franka_problems.npz and franka_1_path.npz, then checks:
  1. Path connects start to goal (within joint-space tolerance)
  2. All waypoints and interpolated edges are collision-free
  3. Total joint-space path length
"""

import numpy as np
import argparse
import os
from franka_utils import (
    load_scene, load_problems, check_collision, check_edge,
    set_config, animate_path, N_JOINTS,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_path(num):
    fname = os.path.join(SCRIPT_DIR, f"franka_{num}_path.npz")
    data = np.load(fname)
    return data["path"]


def evaluate(model, data, q_start, q_goal, path, joint_tol=0.1, n_edge_checks=20):
    """
    Evaluate a Franka arm path.

    Returns dict with:
      - connects_start: bool
      - connects_goal: bool
      - collision_free: bool
      - n_waypoint_collisions: int
      - n_edge_collisions: int
      - path_length: float (joint-space Euclidean)
      - n_waypoints: int
    """
    # Check start/goal connectivity
    start_err = np.max(np.abs(path[0] - q_start))
    goal_err = np.max(np.abs(path[-1] - q_goal))
    connects_start = start_err <= joint_tol
    connects_goal = goal_err <= joint_tol

    # Check waypoint collisions
    n_waypoint_collisions = 0
    for q in path:
        if check_collision(model, data, q):
            n_waypoint_collisions += 1

    # Check edge collisions
    n_edge_collisions = 0
    for i in range(len(path) - 1):
        if not check_edge(model, data, path[i], path[i+1], n_checks=n_edge_checks):
            n_edge_collisions += 1

    # Path length in joint space
    diffs = np.diff(path, axis=0)
    lengths = np.linalg.norm(diffs, axis=1)
    path_length = np.sum(lengths)

    collision_free = (n_waypoint_collisions == 0) and (n_edge_collisions == 0)

    return {
        "connects_start": connects_start,
        "connects_goal": connects_goal,
        "collision_free": collision_free,
        "n_waypoint_collisions": n_waypoint_collisions,
        "n_edge_collisions": n_edge_collisions,
        "path_length": path_length,
        "n_waypoints": len(path),
        "start_error_max": start_err,
        "goal_error_max": goal_err,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a Franka arm path planning solution"
    )
    parser.add_argument("num", type=int, help="Problem number (1-indexed)")
    parser.add_argument("--animate", action="store_true",
                        help="Animate the path in the viewer")
    args = parser.parse_args()

    model, data = load_scene()
    problems = load_problems()

    idx = args.num - 1
    if idx < 0 or idx >= len(problems):
        print(f"Problem {args.num} not found. Available: 1-{len(problems)}")
        return

    q_start, q_goal = problems[idx]
    path = load_path(args.num)

    print(f"Evaluating: franka_{args.num}_path.npz")
    print(f"  Path shape: {path.shape}")

    results = evaluate(model, data, q_start, q_goal, path)

    print(f"\n  Connects start: {results['connects_start']} "
          f"(max joint error: {results['start_error_max']:.4f} rad)")
    print(f"  Connects goal:  {results['connects_goal']} "
          f"(max joint error: {results['goal_error_max']:.4f} rad)")
    print(f"  Collision free: {results['collision_free']} "
          f"({results['n_waypoint_collisions']} waypoint collisions, "
          f"{results['n_edge_collisions']} edge collisions)")
    print(f"  Path length:    {results['path_length']:.4f} rad")
    print(f"  Waypoints:      {results['n_waypoints']}")

    if (results["connects_start"] and results["connects_goal"]
            and results["collision_free"]):
        print("\n  PASS: Valid path found.")
    else:
        print("\n  FAIL: Path does not meet requirements.")

    if args.animate:
        print("\nAnimating path...")
        animate_path(model, data, path)


if __name__ == "__main__":
    main()
