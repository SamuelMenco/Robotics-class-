import heapq
import numpy as np
import matplotlib.pyplot as plt
from evaluator import plot_result
from loader import load_map, world_to_grid, grid_to_world

MAP_NUMBER = 3
MAP_PREFIX = "map_"
SAVE_PLOT = True


def in_bounds(r, c, rows, cols):
    return 0 <= r < rows and 0 <= c < cols


def is_free(grid, r, c):
    return grid[r, c] == 0


def neighbors_8(grid, r, c):
    """
    Return valid 8-connected neighbors as (row, col, move_cost).
    Cardinal moves cost 1.0, diagonal moves cost sqrt(2).
    """
    rows, cols = grid.shape
    out = []

    moves = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, 2 ** 0.5),
        (-1, 1, 2 ** 0.5),
        (1, -1, 2 ** 0.5),
        (1, 1, 2 ** 0.5),
    ]

    for dr, dc, cost in moves:
        nr = r + dr
        nc = c + dc
        if not in_bounds(nr, nc, rows, cols):
            continue
        if not is_free(grid, nr, nc):
            continue

        # Prevent diagonal corner-cutting between obstacle cells.
        if dr != 0 and dc != 0:
            if not is_free(grid, r + dr, c):
                continue
            if not is_free(grid, r, c + dc):
                continue

        out.append((nr, nc, cost))

    return out


def heuristic(cell, goal):
    """Octile-distance heuristic for 8-connected grid motion."""
    dr = abs(cell[0] - goal[0])
    dc = abs(cell[1] - goal[1])
    return (dr + dc) + ((2 ** 0.5) - 2.0) * min(dr, dc)


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def grid_path_to_world(path_rc, origin, resolution, start_world, goal_world):
    """Convert a (row, col) path to an (x, y) waypoint array."""
    waypoints = np.array(
        [grid_to_world(r, c, origin, resolution) for r, c in path_rc],
        dtype=np.float64,
    )
    waypoints[0] = start_world
    waypoints[-1] = goal_world
    return waypoints


def astar_grid(grid, start_rc, goal_rc):
    """
    Run A* on the occupancy grid.
    Returns a list of (row, col) cells from start to goal, or None if no path.
    """
    open_heap = []
    g_score = {start_rc: 0.0}
    came_from = {}
    closed = set()

    start_f = heuristic(start_rc, goal_rc)
    heapq.heappush(open_heap, (start_f, start_rc))

    while open_heap:
        current_f, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)

        if current == goal_rc:
            return reconstruct_path(came_from, current)

        cr, cc = current
        for nr, nc, move_cost in neighbors_8(grid, cr, cc):
            neighbor = (nr, nc)
            tentative_g = g_score[current] + move_cost

            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                neighbor_f = tentative_g + heuristic(neighbor, goal_rc)
                heapq.heappush(open_heap, (neighbor_f, neighbor))

    return None


def main() -> None:
    m = load_map(MAP_PREFIX, MAP_NUMBER)
    grid = m["grid"]
    start_rc = world_to_grid(m["start"], m["origin"], m["resolution"])
    goal_rc = world_to_grid(m["goal"], m["origin"], m["resolution"])
    sr, sc = start_rc

    print(f"Loaded map {MAP_NUMBER}")
    print(f"Grid shape: {grid.shape}")
    print(f"Start (world): {m['start']} -> (row, col): {start_rc}")
    print(f"Goal  (world): {m['goal']} -> (row, col): {goal_rc}")
    print("Debug neighbors from start:")
    for nbr in neighbors_8(grid, sr, sc):
        print(f"  {nbr}")

    path_rc = astar_grid(grid, start_rc, goal_rc)
    if path_rc is None:
        print("No path found.")
    else:
        print(f"A* found a path with {len(path_rc)} grid cells.")

        # Convert to world coordinates and save in evaluator format.
        waypoints = grid_path_to_world(
            path_rc,
            m["origin"],
            m["resolution"],
            m["start"],
            m["goal"],
        )
        out_npz = f"{MAP_PREFIX}{MAP_NUMBER}_astar.npz"
        np.savez(out_npz, path=waypoints)
        print(f"Saved path file: {out_npz}")

        if SAVE_PLOT:

            fig = plot_result(m, waypoints, title=f"Map {MAP_NUMBER} - A*")
            out_png = f"{MAP_PREFIX}{MAP_NUMBER}_astar.png"
            fig.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved plot: {out_png}")


if __name__ == "__main__":
    main()
