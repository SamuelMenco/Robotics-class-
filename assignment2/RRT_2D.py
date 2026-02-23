import numpy as np

from loader import load_map, world_to_grid

MAP_NUMBER = 2
MAP_PREFIX = "map_"
SHOW_PLOT = True

# RRT tuning
MAX_ITERS = 20000
STEP_SIZE = 0.1      # meters
GOAL_TOL = 0.20      # meters
RNG_SEED = 35        # Random seed, mainly added for debbuging.  
CLEARANCE_CELLS = 0  # Set to 1 for a visually clearer obstacle avoidance
EDGE_SAMPLE_FRACTION = 0.25  # Fraction of a cell between edge samples.


def distance(q1, q2):
    """Euclidean distance in state space."""
    return float(np.linalg.norm(q2 - q1))


def sample_state(x_min, x_max, y_min, y_max, rng):
    """
    Sample a random (x, y) state in map bounds.
    """
    x = rng.uniform(x_min, x_max)
    y = rng.uniform(y_min, y_max)
    return np.array([x, y], dtype=np.float64)


def nearest_neighbor(vertices, q_rand):
    """Index of tree vertex closest to q_rand in Euclidean distance."""
    dists = [distance(v, q_rand) for v in vertices]
    return int(np.argmin(dists))


def steer(q_near, q_rand, step_size):
    """
    Move from q_near toward q_rand by at most step_size.
    If q_rand is closer than step_size, return q_rand.
    """
    d = distance(q_near, q_rand)
    if d <= step_size:
        return q_rand.copy()
    direction = (q_rand - q_near) / d
    return q_near + step_size * direction


def inflate_obstacles(grid, radius_cells):
    """Binary obstacle inflation using pure NumPy."""
    if radius_cells <= 0:
        return grid

    rows, cols = grid.shape
    inflated = grid.copy()
    occ = np.argwhere(grid == 1)
    for r, c in occ:
        r0 = max(0, r - radius_cells)
        r1 = min(rows, r + radius_cells + 1)
        c0 = max(0, c - radius_cells)
        c1 = min(cols, c + radius_cells + 1)
        inflated[r0:r1, c0:c1] = 1
    return inflated


def point_in_collision(q, collision_grid, origin, resolution):
    """True if q=(x,y) is out of bounds or in an occupied cell."""
    r, c = world_to_grid(q, origin, resolution)
    rows, cols = collision_grid.shape
    if r < 0 or r >= rows or c < 0 or c >= cols:
        return True
    return collision_grid[r, c] == 1


def edge_in_collision(q_0, q_1, collision_grid, origin, resolution):
    """True if segment q_0 -> q_1 intersects obstacles/out-of-bounds."""
    seg_len = distance(q_0, q_1)
    if seg_len < 1e-12:
        return point_in_collision(q_0, collision_grid, origin, resolution)

    """Here, several point over the line between q0 and q1 are evaluated looking for collisions"""
    n_samples = max(int(np.ceil(seg_len / (resolution * EDGE_SAMPLE_FRACTION))), 2)
    for t in np.linspace(0.0, 1.0, n_samples):
        q = q_0 + t * (q_1 - q_0)
        if point_in_collision(q, collision_grid, origin, resolution):
            return True
    return False


def build_rrt_tree(
    collision_grid,
    origin,
    resolution,
    start_m,
    goal_m,
    x_min,
    x_max,
    y_min,
    y_max,
    n_iters=MAX_ITERS,
):
    """Build an RRT tree and return vertices, parents, rejected-count, and goal index."""
    rng = np.random.default_rng(RNG_SEED)
    vertices = [start_m.copy()]
    parents = [-1]
    n_rejected = 0
    goal_idx = None

    print("\nGrowing tree, please wait...")
    for k in range(n_iters):
        q_rand = sample_state(x_min, x_max, y_min, y_max, rng)
        i_near = nearest_neighbor(vertices, q_rand)
        q_near = vertices[i_near]
        q_new = steer(q_near, q_rand, STEP_SIZE)

        # Safety gate: never add a node that is itself in collision.
        if point_in_collision(q_new, collision_grid, origin, resolution):
            n_rejected += 1
            continue

        if edge_in_collision(q_near, q_new, collision_grid, origin, resolution):
            n_rejected += 1
            continue

        vertices.append(q_new)
        parents.append(i_near)
        i_new = len(vertices) - 1

        # Goal connection check: close enough + collision-free edge to goal.
        if distance(q_new, goal_m) <= GOAL_TOL:
            if not edge_in_collision(q_new, goal_m, collision_grid, origin, resolution):
                vertices.append(goal_m.copy())
                parents.append(i_new)
                goal_idx = len(vertices) - 1
                print(f"Goal connected at iter {k:02d} (goal_idx={goal_idx}).")
                break

    return vertices, parents, n_rejected, goal_idx


def reconstruct_path(vertices, parents, goal_idx):
    """Backtrack parent pointers from goal node to start node."""
    if goal_idx is None:
        return None

    idx_path = [goal_idx]
    cur = goal_idx
    while parents[cur] != -1:
        cur = parents[cur]
        idx_path.append(cur)
    idx_path.reverse()

    return np.array([vertices[i] for i in idx_path], dtype=np.float64)


def plot_tree(m, vertices, parents, title, ax=None):
    from loader import plot_map

    if ax is None:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots(1, 1, figsize=(8, 8))

    plot_map(m, ax=ax, title=title)

    for i in range(1, len(vertices)):
        p = parents[i]
        if p < 0:
            continue
        q_parent = vertices[p]
        q_child = vertices[i]
        ax.plot(
            [q_parent[0], q_child[0]],
            [q_parent[1], q_child[1]],
            "b-",
            linewidth=0.8,
            alpha=0.7,
        )

    # Draw tree nodes.
    pts = np.array(vertices)
    ax.scatter(pts[:, 0], pts[:, 1], s=28, c="deepskyblue", edgecolors="k", linewidths=0.3)

    return ax


def plot_path_only(m, path, title, ax=None):
    from loader import plot_map

    if ax is None:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots(1, 1, figsize=(8, 8))

    plot_map(m, ax=ax, title=title)
    ax.plot(path[:, 0], path[:, 1], "b-", linewidth=1.2, label="RRT Path")
    ax.legend()
    return ax


def main() ->None:
    m = load_map(MAP_PREFIX, MAP_NUMBER)
    grid = m["grid"]
    collision_grid = inflate_obstacles(grid, CLEARANCE_CELLS)
    start_m = m["start"].astype(np.float64)
    goal_m = m["goal"].astype(np.float64)
    print(start_m)
    print(goal_m)

    #Compute samplig boundaries
    l_x, l_y = m["origin"]
    rows, cols = grid.shape
    res = float(m["resolution"])
    u_x = l_x + (cols - 1) * res
    u_y = l_y + (rows - 1) * res
    print(f"bounds (m): x:[{l_x}, {u_x}], y:[{l_y}, {u_y}]")

    vertices, parents, n_rejected, goal_idx = build_rrt_tree(
        collision_grid=collision_grid,
        origin=m["origin"],
        resolution=res,
        start_m=start_m,
        goal_m=goal_m,
        x_min=l_x,
        x_max=u_x,
        y_min=l_y,
        y_max=u_y,
        n_iters=MAX_ITERS,
    )

    print(f"\nTree size after demo growth: {len(vertices)} nodes")
    print(f"Rejected expansions (collision): {n_rejected}")
    print(f"Goal index: {goal_idx}")
    path = reconstruct_path(vertices, parents, goal_idx)
    if path is None:
        print("No path found yet.")
    else:
        print(f"Path waypoints (start->goal): {len(path)}")
        out_file = f"{MAP_PREFIX}{MAP_NUMBER}_rrt.npz"
        np.savez(out_file, path=path)
        print(f"Saved path file: {out_file}")

    if SHOW_PLOT:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        plot_tree(m, vertices, parents, title="Exploration structure", ax=axes[0])
        if path is not None:
            plot_path_only(m, path, title="Obtained path", ax=axes[1])
        else:
            plot_tree(m, vertices, parents, title="Obtained path", ax=axes[1])
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
