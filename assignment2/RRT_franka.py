import numpy as np

from franka_utils import load_scene, load_problems, get_joint_limits, check_collision

from RRT_2D import distance, nearest_neighbor, steer

PROBLEM_NUMBER = 0
# RRT tuning
MAX_ITERS = 20000
STEP_SIZE = 0.1      # rad
GOAL_TOL = 0.20      # rad
RNG_SEED = 35        # Random seed, mainly added for debbuging.  
CLEARANCE_CELLS = 0  # Set to 1 for a visually clearer obstacle avoidance

def sample_state_franka(lower, upper, rng):
    return rng.uniform(lower, upper)

def point_in_collision(q, lower, upper, model, data):
    """
    True if q violates joint limits OR robot is in collision.
    """
    # Ensure shapes match
    q = np.asarray(q, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)

    # Joint-limits gate (fast)
    if np.any(q < lower) or np.any(q > upper):
        return True

    # Geometry collision (slow)
    return bool(check_collision(model, data, q))


EDGE_Q_STEP = 0.05  # radians between samples along an edge (tune)

def edge_in_collision(q0, q1, lower, upper, model, data, q_step=EDGE_Q_STEP):
    """
    True if the joint-space segment q0 -> q1 passes through:
      - out-of-limits configurations, or
      - any collision configuration.

    q_step controls sampling resolution in joint space (radians).
    """
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)

    seg_len = float(np.linalg.norm(q1 - q0))
    if seg_len < 1e-12:
        return point_in_collision(q0, lower, upper, model, data)

    # Number of samples so that consecutive samples are about q_step apart
    n_samples = max(int(np.ceil(seg_len / q_step)) + 1, 2)

    for t in np.linspace(0.0, 1.0, n_samples):
        q = q0 + t * (q1 - q0)
        if point_in_collision(q, lower, upper, model, data):
            return True

    return False

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
        q_rand = sample_state_franka(lower, upper, rng)
        i_near = nearest_neighbor(vertices, q_rand)
        q_near = vertices[i_near]
        q_new = steer(q_near, q_rand, STEP_SIZE)
        q_new = np.clip(q_new, lower, upper)

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

    return vertices, parents, n_rejected, goal_idx


def main():
    model, data = load_scene()
    problems = load_problems()

    idx = PROBLEM_NUMBER
    q_start, q_goal = problems[idx]
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

    vertices, parents, n_rejected, goal_idx = build_rrt_tree(q_start,q_goal,lower,upper,model,data)

if __name__ == "__main__":
    main()
