import numpy as np
import matplotlib.pyplot as plt

scans_file = "sim_25_scans.npz"

def load_scans(file):
    data = np.load(file)
    scans = [data[k] for k in data]
    return scans

def load_pose_graph_data(path):
    data = np.load(path)
    world_poses = data["world_poses"]
    edge_i = data["edge_i"]
    edge_j = data["edge_j"]
    edge_R = data["edge_R"]
    edge_t = data["edge_t"]
    lc_i = data["lc_i"]
    lc_j = data["lc_j"]
    lc_R = data["lc_R"]
    lc_t = data["lc_t"]

    return {
        "world_poses": world_poses,
        "edge_i": edge_i,
        "edge_j": edge_j,
        "edge_R": edge_R,
        "edge_t": edge_t,
        "lc_i": lc_i,
        "lc_j": lc_j,
        "lc_R": lc_R,
        "lc_t": lc_t,
    }

def rt_to_pose(R, t):
    theta = np.arctan2(R[1, 0], R[0, 0])
    return np.array([t[0], t[1], theta])


def pose_to_rt(pose):
    x, y, theta = pose
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    t = np.array([x, y])
    return R, t


def transform_scan(scan, pose):
    R, t = pose_to_rt(pose)
    return (R @ scan.T).T + t


def plot_all_scans_world(scans, poses, title, ax):
    for scan, pose in zip(scans, poses):
        scan_world = transform_scan(scan, pose)
        ax.scatter(scan_world[:, 0], scan_world[:, 1], s=1, alpha=0.3)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")

def compute_residual(xi, xj, z_ij):

    x_i, y_i, theta_i = xi[0], xi[1], xi[2]
    x_j, y_j, theta_j = xj[0], xj[1], xj[2]
    dx_meas, dy_meas, dtheta_meas = z_ij[0], z_ij[1], z_ij[2]

    c, s = np.cos(theta_i), np.sin(theta_i)
    dx_world = x_j - x_i
    dy_world = y_j - y_i
    dx_pred = c * dx_world + s * dy_world
    dy_pred = -s * dx_world + c * dy_world
    dtheta_pred = theta_j - theta_i

    e = np.array(
        [
            dx_pred - dx_meas,
            dy_pred - dy_meas,
            np.arctan2(
                np.sin(dtheta_pred - dtheta_meas),
                np.cos(dtheta_pred - dtheta_meas),
            ),
        ]
    )
    return e


def predict_measurement(xi, xj):
    x_i, y_i, theta_i = xi[0], xi[1], xi[2]
    x_j, y_j, theta_j = xj[0], xj[1], xj[2]
    c, s = np.cos(theta_i), np.sin(theta_i)
    dx_world = x_j - x_i
    dy_world = y_j - y_i
    dx_pred = c * dx_world + s * dy_world
    dy_pred = -s * dx_world + c * dy_world
    dtheta_pred = theta_j - theta_i
    dtheta_pred = np.arctan2(np.sin(dtheta_pred), np.cos(dtheta_pred))
    return np.array([dx_pred, dy_pred, dtheta_pred])


def compute_jacobians(xi, xj, z_ij):
    """
    Compute Jacobians of residual w.r.t. poses xi and xj.
    Returns: A (de/dxi), B (de/dxj), both 3x3
    """
    theta_i = xi[2]
    c, s = np.cos(theta_i), np.sin(theta_i)
    dx = xj[0] - xi[0]
    dy = xj[1] - xi[1]

    A = np.array(
        [
            [-c, -s, -s * dx + c * dy],
            [s, -c, -c * dx - s * dy],
            [0, 0, -1],
        ]
    )

    B = np.array(
        [
            [c, s, 0],
            [-s, c, 0],
            [0, 0, 1],
        ]
    )

    return A, B


def build_linear_system(poses, edges, measurements):
    """
    Build H and b for Gauss-Newton.
    poses: array of (x, y, theta)
    edges: array of (i, j) pairs
    measurements: array of (dx, dy, dtheta) for each edge
    """
    n = len(poses)
    H = np.zeros((3 * n, 3 * n))
    b = np.zeros(3 * n)

    for (i, j), z_ij in zip(edges, measurements):
        xi, xj = poses[i], poses[j]
        e_ij = compute_residual(xi, xj, z_ij)
        A, B = compute_jacobians(xi, xj, z_ij)

        H[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] += A.T @ A
        H[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] += A.T @ B
        H[3 * j : 3 * j + 3, 3 * i : 3 * i + 3] += B.T @ A
        H[3 * j : 3 * j + 3, 3 * j : 3 * j + 3] += B.T @ B

        b[3 * i : 3 * i + 3] += A.T @ e_ij
        b[3 * j : 3 * j + 3] += B.T @ e_ij

    return H, b


def optimize_pose_graph(initial_poses, edges, measurements, n_iter=10, fix_first=True):
    """
    Optimize pose graph using Gauss-Newton.
    """
    poses = [np.array(p) for p in initial_poses]
    n = len(poses)

    for _ in range(n_iter):
        H, b = build_linear_system(poses, edges, measurements)

        # Fix first pose (add large weight)
        if fix_first:
            H[0:3, 0:3] += 1e6 * np.eye(3)

        dx = np.linalg.solve(H, -b)

        for i in range(n):
            poses[i] = poses[i] + dx[3 * i : 3 * i + 3]
            poses[i][2] = np.arctan2(np.sin(poses[i][2]), np.cos(poses[i][2]))

    return poses


def save_slam_poses(poses, fnum, prefix="sim_", suffix=".npz"):
    data = {str(i): np.array(p) for i, p in enumerate(poses)}
    np.savez(f"{prefix}{fnum}_slam{suffix}", **data)

def main() -> None:
    scans = load_scans(scans_file)
    pose_graph = load_pose_graph_data("pose_graph_data.npz")
    poses = pose_graph["world_poses"]

    #1. Create a list of indices
    cons_edges = np.column_stack(
        [
            pose_graph["edge_i"],
            pose_graph["edge_j"],
        ]
    )
    loop_edges = np.column_stack(
        [
            pose_graph["lc_i"],
            pose_graph["lc_j"],
        ]
    )

    edges = np.vstack(
        [
            cons_edges,
            loop_edges,
        ]

    )

    #2. Create a list of transformations
    edge_z = np.stack(
        [rt_to_pose(R, t) for R, t in zip(pose_graph["edge_R"], pose_graph["edge_t"])],
        axis=0,
    )
    lc_z = (
        np.stack(
            [rt_to_pose(R, t) for R, t in zip(pose_graph["lc_R"], pose_graph["lc_t"])],
            axis=0,
        )
        if loop_edges.shape[0] > 0
        else np.zeros((0, 3))
    )
    measurements = np.vstack([edge_z, lc_z]) if lc_z.shape[0] > 0 else edge_z
    corrected_poses = optimize_pose_graph(poses, edges, measurements, n_iter=100)

    save_slam_poses(corrected_poses, fnum=30)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    plot_all_scans_world(scans, poses, "Scan Matching", ax1)
    plot_all_scans_world(scans, corrected_poses, "PGO", ax2)
    fig.suptitle("Robot's environment map (sim 25)")
    plt.tight_layout()
    plt.savefig("robot_environment_map_sim_25.png", dpi=300)
    plt.show()
    


if __name__ == "__main__":
    main()
