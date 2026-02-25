import numpy as np

from franka_utils import (
    load_scene,
    load_problems,
    get_joint_limits,
    check_collision,
    check_edge,
    animate_path,
)

PROBLEM_NUMBER = 0
# RRT tuning
MAX_ITERS = 20000
STEP_SIZE = 0.05     # rad
GOAL_TOL = 0.1       # rad
RNG_SEED = 32        # Random seed, mainly added for debbuging.
EDGE_CHECKS = 20     # Samples for helper edge collision check.
GOAL_BIAS = 0.10     # Probability of sampling q_goal directly.
ANIMATE_RESULT = True

def sample_state_franka(lower, upper, q_goal, rng):
    if rng.random() < GOAL_BIAS:
        return q_goal.copy()
    return rng.uniform(lower, upper)

def distance(q1, q2):
    return float(np.linalg.norm(q2 - q1))

def nearest_neighbor(vertices, q_rand):
    v_np = np.array(vertices)
    return np.argmin(np.linalg.norm(v_np - q_rand, axis=1))

def steer(q_near, q_rand, step_size):
    d = distance(q_near, q_rand)
    if d <= step_size:
        return q_rand.copy()
    direction = (q_rand - q_near) / d
    return q_near + step_size * direction

def point_in_collision(q, lower, upper, model, data):
    """
    True if q is in geometric collision (MuJoCo contact check).
    """
    q = np.asarray(q, dtype=np.float64)
    return bool(check_collision(model, data, q))


def edge_in_collision(q0, q1, lower, upper, model, data, n_checks=EDGE_CHECKS):
    """
    True if edge q0 -> q1 is invalid.
    Uses helper check_edge for geometric collision and a joint-limits gate.
    """
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)

    # Helper returns True if collision-free; convert to invalid-edge boolean.
    return not check_edge(model, data, q0, q1, n_checks=n_checks)

def build_rrt_tree(
    q_start,
    q_goal,
    lower,
    upper,
    model,
    data,
    n_iters=MAX_ITERS,
):
    """Build an RRT tree and return vertices, parents, rejected-count, and goal index."""
    rng = np.random.default_rng(RNG_SEED)
    vertices = [q_start.copy()]
    parents = [-1]
    n_rejected = 0
    goal_idx = None

    print("\nGrowing tree, please wait...")
    for k in range(n_iters):
        q_rand = sample_state_franka(lower, upper, q_goal, rng)
        i_near = nearest_neighbor(vertices, q_rand)
        q_near = vertices[i_near]
        q_new = steer(q_near, q_rand, STEP_SIZE)

        # Safety gate: never add a node that is itself in collision.
        if point_in_collision(q_new,  lower, upper, model, data):
            n_rejected += 1
            continue

        if edge_in_collision(q_near, q_new,  lower, upper, model, data):
            n_rejected += 1
            continue

        vertices.append(q_new)
        parents.append(i_near)
        i_new = len(vertices) - 1

        # Goal connection check: close enough + collision-free edge to goal.
        if distance(q_new, q_goal) <= GOAL_TOL:
            if not edge_in_collision(q_new, q_goal,  lower, upper, model, data):
                vertices.append(q_goal.copy())
                parents.append(i_new)
                goal_idx = len(vertices) - 1
                print(f"Goal connected at iter {k:02d} (goal_idx={goal_idx}).")
                break
        if (k + 1) % 2000 == 0:
            print(f"  Iteration {k+1}: vertices={len(vertices)}, rejected={n_rejected}")

    return vertices, parents, n_rejected, goal_idx

def reconstruct_path(vertices, parents, goal_idx):
    """Backtrack from goal_idx to root and return Kx7 array."""
    if goal_idx is None:
        return None
    idx_path = [goal_idx]
    cur = goal_idx
    while parents[cur] != -1:
        cur = parents[cur]
        idx_path.append(cur)
    idx_path.reverse()
    return np.array([vertices[i] for i in idx_path], dtype=np.float64)

def main():
    model, data = load_scene()
    problems = load_problems()

    idx = PROBLEM_NUMBER
    if idx < 0 or idx >= len(problems):
        print(f"Problem index {idx} is out of range. Available: 0..{len(problems)-1}")
        return
    
    q_start, q_goal = problems[idx]
    #q_start = -q_start
    #q_goal = -q_goal
    
    lower, upper = get_joint_limits(model)

    print(f"Loaded Franka problem {PROBLEM_NUMBER}")
    print(f"  Start q: {np.array2string(q_start, precision=3)}")
    print(f"  Goal  q: {np.array2string(q_goal, precision=3)}")
    print(f"  Joint lower limits: {np.array2string(lower, precision=3)}")
    print(f"  Joint upper limits: {np.array2string(upper, precision=3)}")

    start_collision = check_collision(model, data, q_start)
    goal_collision = check_collision(model, data, q_goal)
    print(f"  Start in collision: {start_collision}")
    print(f"  Goal in collision:  {goal_collision}")

    vertices, parents, n_rejected, goal_idx = build_rrt_tree(
        q_start, q_goal, lower, upper, model, data
    )

    print(f"  Tree vertices: {len(vertices)}")
    print(f"  Rejected expansions: {n_rejected}")
    print(f"  Goal index: {goal_idx}")

    path = reconstruct_path(vertices, parents, goal_idx)
    if path is None:
        print("No path found within MAX_ITERS.")
        return

    # Evaluator expects files named franka_{N}_path.npz, where N is 1-indexed.
    out_num = idx + 1
    out_file = f"franka_{out_num}_path.npz"
    np.savez(out_file, path=path)
    print(f"Saved path file: {out_file}")
    print(f"Path waypoints: {len(path)}")

    if ANIMATE_RESULT:
        animate_path(model, data, path)

if __name__ == "__main__":
    main()
