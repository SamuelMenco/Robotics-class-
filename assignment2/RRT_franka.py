import numpy as np

from franka_utils import load_scene, load_problems, get_joint_limits, check_collision

PROBLEM_NUMBER = 0


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


if __name__ == "__main__":
    main()
