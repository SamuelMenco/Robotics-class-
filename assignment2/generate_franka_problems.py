"""
Generate start/goal configuration pairs for Franka arm planning problems.

Usage:
    python3 generate_franka_problems.py

Produces franka_problems.npz with collision-free start/goal pairs.
"""

import numpy as np
import mujoco
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_PATH = os.path.join(SCRIPT_DIR, "franka_scene.xml")


def check_collision(model, data, q):
    """Check if configuration q is in collision (simplified)."""
    data.qpos[:7] = q
    mujoco.mj_forward(model, data)
    for i in range(data.ncon):
        contact = data.contact[i]
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
        if {g1, g2} == {"floor", "link0_geom"}:
            continue
        return True
    return False


def sample_free_config(model, data, rng):
    """Sample a collision-free random configuration."""
    lower = model.jnt_range[:7, 0]
    upper = model.jnt_range[:7, 1]
    for _ in range(10000):
        q = rng.uniform(lower, upper)
        if not check_collision(model, data, q):
            return q
    raise RuntimeError("Could not find collision-free config")


def main():
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(42)

    # Problem 1: arm to one side of obstacle -> other side
    # Use hand-picked configs near the obstacle for an interesting problem
    q_home = np.array([0, -0.3, 0, -2.0, 0, 1.9, 0.78])

    # Verify home is collision-free
    assert not check_collision(model, data, q_home), "Home config in collision!"

    problems = {}

    # Problem 1: home -> reach to the right side
    q_goal1 = np.array([0.8, -0.5, 0.3, -1.8, 0.2, 1.5, 0.4])
    if check_collision(model, data, q_goal1):
        print("Warning: goal1 in collision, searching for alternative...")
        q_goal1 = sample_free_config(model, data, rng)
    problems["start_0"] = q_home
    problems["goal_0"] = q_goal1
    print(f"Problem 1: start={q_home}, goal={q_goal1}")
    print(f"  Start collision: {check_collision(model, data, q_home)}")
    print(f"  Goal collision:  {check_collision(model, data, q_goal1)}")

    # Problem 2: wider motion, arm sweeps across workspace
    q_start2 = np.array([-0.5, -0.4, 0.5, -2.2, 0.3, 2.0, 0.5])
    q_goal2 = np.array([0.5, 0.2, -0.5, -1.5, -0.3, 2.5, -0.3])
    if check_collision(model, data, q_start2):
        print("Warning: start2 in collision, searching...")
        q_start2 = sample_free_config(model, data, rng)
    if check_collision(model, data, q_goal2):
        print("Warning: goal2 in collision, searching...")
        q_goal2 = sample_free_config(model, data, rng)
    problems["start_1"] = q_start2
    problems["goal_1"] = q_goal2
    print(f"Problem 2: start={q_start2}, goal={q_goal2}")
    print(f"  Start collision: {check_collision(model, data, q_start2)}")
    print(f"  Goal collision:  {check_collision(model, data, q_goal2)}")

    np.savez(os.path.join(SCRIPT_DIR, "franka_problems.npz"), **problems)
    print("\nSaved franka_problems.npz")


if __name__ == "__main__":
    main()
