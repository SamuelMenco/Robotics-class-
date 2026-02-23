## Assignment 2

You will submit code that reads in problem information and produces corresponding output files (see below). **Include instructions for how to run your code to generate the outputs.** Your code should not be hard-coded to solve specific examples. 

### Part 1: 2D Grid Planning

You can visualize a map by running:

```
python3 loader.py 1
```
where the file being loaded is `map_1.npz`.

I will use the `evaluator.py` file to evaluate the output of your code. It expects files like `map_1_astar.npz` and `map_1_rrt.npz` to exist. Run:
```
python3 evaluator.py 1 astar
python3 evaluator.py 1 rrt
```
to see the output. Use this code to ensure that your path planning code outputs files that can be suitably evaluated.

Generate A* output files with:
```
python3 astar.py 1
python3 astar.py 2
python3 astar.py 3
```
This creates `map_1_astar.npz`, `map_2_astar.npz`, and `map_3_astar.npz`.

You can store a path as:
```python
np.savez(f"map_{N}_astar.npz", path=waypoints)
```
where `waypoints` is an Mx2 numpy array of (x, y) positions in world coordinates.

### Part 2: Franka Arm Planning

Visualize a planning problem and test the collision checker:
```
python3 franka_utils.py 1
```

Evaluate your solution:
```
python3 franka_evaluator.py 1
python3 franka_evaluator.py 1 --animate
```

Save your path as:
```python
np.savez(f"franka_{N}_path.npz", path=joint_waypoints)
```
where `joint_waypoints` is a Kx7 numpy array of joint angles.

### Generating test data

```
python3 generate_maps.py
python3 generate_franka_problems.py
```
