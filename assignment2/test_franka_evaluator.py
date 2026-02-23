"""
Test that franka_evaluator.py works correctly.

Generates two test paths for problem 1:
  1. Linear interpolation from start to goal (may collide with obstacle)
  2. Stay-at-start path (should be collision-free but not connect to goal)

Requires: mujoco, franka_scene.xml, franka_problems.npz

Usage:
    python3 test_franka_evaluator.py
"""

import numpy as np
import os
from franka_utils import load_scene, load_problems, check_collision
from franka_evaluator import evaluate

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

model, data = load_scene()
problems = load_problems()
q_start, q_goal = problems[0]

print(f"Start: {np.array2string(q_start, precision=3)}")
print(f"Goal:  {np.array2string(q_goal, precision=3)}")
print(f"Start in collision: {check_collision(model, data, q_start)}")
print(f"Goal in collision:  {check_collision(model, data, q_goal)}")
print()

# Test 1: linear interpolation from start to goal (50 waypoints)
path_linear = np.linspace(q_start, q_goal, 50)
np.savez(os.path.join(SCRIPT_DIR, "franka_1_path.npz"), path=path_linear)

results = evaluate(model, data, q_start, q_goal, path_linear)
print("Test 1: Linear interpolation start -> goal")
print(f"  Connects start: {results['connects_start']}")
print(f"  Connects goal:  {results['connects_goal']}")
print(f"  Collision free: {results['collision_free']} "
      f"({results['n_waypoint_collisions']} waypoint, "
      f"{results['n_edge_collisions']} edge collisions)")
print(f"  Path length:    {results['path_length']:.4f} rad")
assert results["connects_start"], "Should connect to start!"
assert results["connects_goal"], "Should connect to goal!"
print("  OK\n")

# Test 2: path that stays at start (collision-free but doesn't reach goal)
path_stationary = np.tile(q_start, (10, 1))
results = evaluate(model, data, q_start, q_goal, path_stationary)
print("Test 2: Stationary at start (expect: connects start, not goal)")
print(f"  Connects start: {results['connects_start']}")
print(f"  Connects goal:  {results['connects_goal']}")
print(f"  Collision free: {results['collision_free']}")
print(f"  Path length:    {results['path_length']:.4f} rad")
assert results["connects_start"], "Should connect to start!"
assert not results["connects_goal"], "Should NOT connect to goal!"
assert results["collision_free"], "Start config should be collision-free!"
print("  OK\n")

# Clean up
os.remove(os.path.join(SCRIPT_DIR, "franka_1_path.npz"))

print("All tests passed.")
