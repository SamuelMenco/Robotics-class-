import numpy as np

data = np.load("sim_25_slam.npz")
print(data["0"])
print(data["1"])
print(len(data.files))