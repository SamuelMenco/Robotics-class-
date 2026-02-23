"""
Test that evaluator.py works correctly on map_1.

Generates two test paths:
  1. Straight line from start to goal (should FAIL - crosses wall)
  2. Path through the gap in the wall (should PASS)

Usage:
    python3 test_evaluator.py
"""

import numpy as np
from evaluator import load_map, load_path, evaluate

# Load map 1
m = load_map("map_", 1)
start = m["start"]
goal = m["goal"]

# Test 1: straight line (should collide with the wall)
path_straight = np.linspace(start, goal, 50)
np.savez("map_1_astar.npz", path=path_straight)

results = evaluate(m, path_straight)
print("Test 1: Straight line (expect FAIL)")
print(f"  Collision free: {results['collision_free']} ({results['n_collisions']} collisions)")
print(f"  Path length:    {results['path_length']:.4f} m")
assert not results["collision_free"], "Should have collisions!"
print("  OK\n")

# Test 2: path that goes through the gap
# Wall is rows 20-30, gap at cols 20-25, resolution=0.1
# Gap center in world coords: x=2.25, y=2.5
path_gap = np.array([
    start,
    [2.25, 1.5],   # approach gap
    [2.25, 2.5],   # through gap
    [2.25, 3.5],   # past gap
    goal,
])
np.savez("map_1_rrt.npz", path=path_gap)

results = evaluate(m, path_gap)
print("Test 2: Through the gap (expect PASS)")
print(f"  Collision free: {results['collision_free']} ({results['n_collisions']} collisions)")
print(f"  Path length:    {results['path_length']:.4f} m")
assert results["collision_free"], "Should be collision-free!"
assert results["connects_start"], "Should connect to start!"
assert results["connects_goal"], "Should connect to goal!"
print("  OK\n")

# Clean up
import os
os.remove("map_1_astar.npz")
os.remove("map_1_rrt.npz")

print("All tests passed.")
