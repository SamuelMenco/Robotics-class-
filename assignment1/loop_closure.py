import numpy as np

from ICP import compute_results, icp, load_scans

file_name = "sim_25_scans.npz"

def main()-> None:
    """Resuts= Transfroms from one scan to another starting in transform T1 as seen in 0, also includes the last error from ICP
       world_transfomrs = Pose of the robot at each scan with respecct to the scan 0
    """
    #1. Compute consecutive transforms and estimate world pose for each scan
    results, world_poses = compute_results()
    scans = load_scans(file_name)

    #2. Get error information from ICP
    errors = [r["last_error"] for r in results]
    min_error = min(errors)
    max_error = max(errors)
    max_idx = max(range(len(errors)), key=errors.__getitem__)
    min_idx = min(range(len(errors)), key=errors.__getitem__)
    avg_error = sum(errors) / len(errors)
    print(f"min error: {min_error} with index: {min_idx}")
    print(f"max error: {max_error} with index: {max_idx}")
    print("average error:", avg_error)

    #3. Select loop closure candidates (no ICP yet)
    loop_candidates = initial_loop_candidates(
        poses=world_poses,
        min_sep=10,
        max_dist=2,
        max_angle_deg=90.0,
        max_loops_per_edge=5,
    )
    
    #4. Run ICP on candidates and keep those with error < 0.5
    loop_closures = loop_closures_by_icp(scans, loop_candidates, 0.5)
    print(f"Selected {len(loop_closures)} loop closures (error < 0.5)")
    print(loop_closures)
    # Save everything to disk for later use
    save_pose_graph_data(world_poses, results, loop_closures, "pose_graph_data.npz")
    print("Saved pose_graph_data.npz")


def initial_loop_candidates(
    poses,
    min_sep,
    max_dist,
    max_angle_deg,
    max_loops_per_edge,
):
    max_angle = np.deg2rad(max_angle_deg)
    num = poses.shape[0]
    candidates = []
    per_i = np.zeros(num, dtype=int)
    per_j = np.zeros(num, dtype=int)

    for i in range(num):
        for j in range(i + min_sep, num):
            if per_i[i] >= max_loops_per_edge:
                break
            if per_j[j] >= max_loops_per_edge:
                break

            dx = poses[i, 0] - poses[j, 0]
            dy = poses[i, 1] - poses[j, 1]
            dist = np.hypot(dx, dy)
            if dist > max_dist:
                continue

            dtheta = np.arctan2(
                np.sin(poses[i, 2] - poses[j, 2]),
                np.cos(poses[i, 2] - poses[j, 2]),
            )
            if abs(dtheta) > max_angle:
                continue

            candidates.append(
                {
                    "i": i,
                    "j": j,
                    "dist": dist,
                    "dtheta": dtheta,
                }
            )
            per_i[i] += 1
            per_j[j] += 1

    return candidates


def loop_closures_by_icp(scans, candidates, max_error):
    closures = []
    for c in candidates:
        i = c["i"]
        j = c["j"]
        R, t, err = icp(scans[j], scans[i])
        if err < max_error:
            closures.append(
                {
                    "i": i,
                    "j": j,
                    "error": err,
                    "R": R,
                    "t": t,
                    "dist": c["dist"],
                    "dtheta": c["dtheta"],
                }
            )
    return closures


def save_pose_graph_data(world_poses, results, loop_closures, out_path):
    # Consecutive edges
    edge_i = np.array([r["pair"][0] for r in results], dtype=int)
    edge_j = np.array([r["pair"][1] for r in results], dtype=int)
    edge_R = np.stack([r["R"] for r in results], axis=0)
    edge_t = np.stack([r["t"] for r in results], axis=0)

    # Loop closure edges
    if loop_closures:
        lc_i = np.array([lc["i"] for lc in loop_closures], dtype=int)
        lc_j = np.array([lc["j"] for lc in loop_closures], dtype=int)
        lc_R = np.stack([lc["R"] for lc in loop_closures], axis=0)
        lc_t = np.stack([lc["t"] for lc in loop_closures], axis=0)
    else:
        lc_i = np.array([], dtype=int)
        lc_j = np.array([], dtype=int)
        lc_R = np.zeros((0, 2, 2))
        lc_t = np.zeros((0, 2))

    np.savez(
        out_path,
        world_poses=world_poses,
        edge_i=edge_i,
        edge_j=edge_j,
        edge_R=edge_R,
        edge_t=edge_t,
        lc_i=lc_i,
        lc_j=lc_j,
        lc_R=lc_R,
        lc_t=lc_t,
    )






if __name__ == "__main__":
    main()
