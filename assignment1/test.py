import numpy as np
import matplotlib.pyplot as plt


def load_npz_as_array(path):
    data = np.load(path)
    poses = np.asarray([data[k] for k in data])
    return poses


def plot_estimated(est, title, out_path):
    plt.figure()
    plt.plot(est[:, 0], est[:, 1], label="Estimated")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(title)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()


if __name__ == "__main__":
    est = load_npz_as_array("sim_25_slam.npz")
    plot_estimated(est, "Estimated robot trajectory", "estimated_trajectory.png")
