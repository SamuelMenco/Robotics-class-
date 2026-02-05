import numpy as np
import matplotlib.pyplot as plt

file_name = "sim_25_scans.npz"

def load_scans(file):
    data = np.load(file)
    scans = [data[k] for k in data]
    return scans

def plot_two_scans(scan_a, scan_b) -> None:
    x_a = scan_a[:, 0]
    y_a = scan_a[:, 1]
    x_b = scan_b[:, 0]
    y_b = scan_b[:, 1]

    plt.figure()
    plt.scatter(x_a, y_a, s=5, label="scan 1")
    plt.scatter(x_b, y_b, s=5, label="scan 2")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Two scans (robot frame)")
    plt.legend()
    plt.axis("equal")
    plt.show()


def plot_before_after(target_scan, source_before, source_after) -> None:
    x_t = target_scan[:, 0]
    y_t = target_scan[:, 1]
    x_b = source_before[:, 0]
    y_b = source_before[:, 1]
    x_a = source_after[:, 0]
    y_a = source_after[:, 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.scatter(x_t, y_t, s=5, label="target (scan i)")
    ax1.scatter(x_b, y_b, s=5, label="source (scan i+1) before")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_title("Before ICP Transform")
    ax1.legend()
    ax1.set_aspect("equal", adjustable="box")

    ax2.scatter(x_t, y_t, s=5, label="target (scan i)")
    ax2.scatter(x_a, y_a, s=5, label="source (scan i+1) after")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.set_title("After ICP Transform")
    ax2.legend()
    ax2.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.show()

def n_n(source, target):
    indices = []
    for p in source:
        dists = np.linalg.norm(target - p, axis=1)
        indices.append(np.argmin(dists))
    return np.array(indices)

def compute_transformation(source, target_matched):
    # Centroids
    centroid_s = source.mean(axis=0)
    centroid_t = target_matched.mean(axis=0)
    # Center points
    s_centered = source - centroid_s
    t_centered = target_matched - centroid_t
    # SVD
    H = s_centered.T @ t_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # Ensure proper rotation (det = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = centroid_t - R @ centroid_s
    return R, t.flatten()

def icp(source,target,max_iters=100, tol=1e-6):

    R_total = np.eye(2)
    t_total = np.zeros(2)
    errors = []
    current = source.copy()

    for i in range(max_iters):

        #Step 1. Find the nearest neighbors
        nn_indices = n_n(current,target)
        target_matched = target[nn_indices]

        #Step 2. Compute error (current alignment)
        error = np.mean(np.linalg.norm(current - target_matched, axis=1))

        #Step 3. Compute transformation
        R, t = compute_transformation(current, target_matched)

        #Step 4. Apply transformation (candidate)
        current_candidate = (R @ current.T).T + t
        nn_indices_candidate = n_n(current_candidate, target)
        target_matched_candidate = target[nn_indices_candidate]
        error_candidate = np.mean(
            np.linalg.norm(current_candidate - target_matched_candidate, axis=1)
        )

        # Accept only if it improves (or stays equal)
        if error_candidate <= error:
            current = current_candidate
            errors.append(error_candidate)
            R_total = R @ R_total
            t_total = R @ t_total + t
        else:
            errors.append(error)
            break

        # Check convergence: stop if improvement is small
        if len(errors) > 1 and abs(errors[-2] - errors[-1]) < tol:
            break

    return R_total, t_total, errors[-1]


def forward_pass(scans):
    results = []
    for i in range(len(scans) - 1):
        source = scans[i+1]
        target = scans[i]
        R_total, t_total, error = icp(source, target)
        results.append(
            {
                "pair": (i, i + 1),
                "R": R_total,
                "t": t_total,
                "last_error": error,
            }
        )
    return results


def plot_pair_from_results(scans, results, i) -> None:
    if i < 0 or i > len(scans) - 2:
        raise ValueError("i must be between 0 and len(scans)-2")
    target = scans[i]
    source = scans[i+1]
    result = results[i]
    R_total = result["R"]
    t_total = result["t"]
    source_trans = (R_total @ source.T).T + t_total
    plot_before_after(target, source, source_trans)


def compose_to_world(results):
    R_world = [np.eye(2)]
    t_world = [np.zeros(2)]
    pose_world = np.zeros((len(results) + 1, 3))
    pose_world[0] = np.array([0.0, 0.0, 0.0])
    for i, result in enumerate(results):
        R_k1_k = result["R"]
        t_k1_k = result["t"]
        R_next = R_world[-1] @ R_k1_k
        t_next = R_world[-1] @ t_k1_k + t_world[-1]
        R_world.append(R_next)
        t_world.append(t_next)
        pose_world[i + 1, 0] = t_next[0]
        pose_world[i + 1, 1] = t_next[1]
        pose_world[i + 1, 2] = np.arctan2(R_next[1, 0], R_next[0, 0])
    return R_world, t_world, pose_world


def plot_all_scans_world(scans, R_world, t_world) -> None:
    plt.figure()
    for scan, R, t in zip(scans, R_world, t_world):
        scan_world = (R @ scan.T).T + t
        plt.scatter(scan_world[:, 0], scan_world[:, 1], s=1, alpha=0.3)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("All scans in world frame")
    plt.axis("equal")
    plt.show()

def compute_results():
    scans = load_scans(file_name)
    print(f"Loaded {len(scans)} scans from {file_name}")
    results = forward_pass(scans)
    _, _, world_transforms = compose_to_world(results)
    return results, world_transforms

    

def main() -> None:
    """
    scans = load_scans(file_name)
    print(f"Loaded {len(scans)} scans from {file_name}")
    if len(scans) > 1:

       
        first = scans[207]
        second = scans[237]
        R_total, t_total, errors = icp(second,first)
        second_mod = (R_total @ second.T).T + t_total
        plot_before_after(first,second,second_mod)
        print(R_total)
        print(f"angle: {np.arctan2(R_total[1, 0], R_total[0, 0])}")
        print(t_total)

 
        results = forward_pass(scans)
        plot_pair_from_results(scans,results,0)
        R_world, t_world, world_transforms = compose_to_world(results)
        plot_all_scans_world(scans, R_world, t_world)
        #print(f"{len(world_transforms)} transforms computed in total")
        #_,_,error_list = icp(scans[1],scans[0])
        #print(error_list)
    """



if __name__ == "__main__":
    main()
