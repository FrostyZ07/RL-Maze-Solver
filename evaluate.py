import argparse
import glob
import heapq
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from envs.maze_env import MazeEnv

SEED0 = 1000  # held-out seeds: 1000, 1001, ...

def astar_action(env):
    """Cheating baseline: knows the map and goal, ignores moving obstacles. Replans every step."""
    pos, orn = env.get_pose()
    start = (int(round(pos[1])) - env.offset, int(round(pos[0])) - env.offset)
    goal = env.goal_cell
    g = {start: 0}
    prev = {}
    pq = [(abs(start[0] - goal[0]) + abs(start[1] - goal[1]), start)]
    while pq:
        _, cur = heapq.heappop(pq)
        if cur == goal:
            break
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (cur[0] + dr, cur[1] + dc)
            if env.layout[n[0]][n[1]] == 0 and g[cur] + 1 < g.get(n, 1e9):
                g[n] = g[cur] + 1
                prev[n] = cur
                heapq.heappush(pq, (g[n] + abs(n[0] - goal[0]) + abs(n[1] - goal[1]), n))
    nxt = goal
    while nxt in prev and prev[nxt] != start:
        nxt = prev[nxt]
    if start == goal:
        target = env.goal_pos
    else:
        target = np.array([nxt[1] + env.offset, nxt[0] + env.offset], dtype=float)
    d = target - pos[:2]
    err = np.arctan2(d[1], d[0]) - orn
    err = (err + np.pi) % (2 * np.pi) - np.pi
    return 1 if err > 0.3 else 2 if err < -0.3 else 0

def run(name, env, act, episodes, vec=None):
    wins, steps_ok = 0, []
    for i in range(episodes):
        if vec is None:
            env.reset(seed=SEED0 + i)
        else:
            vec.seed(SEED0 + i)
            obs = vec.reset()
        steps = 0
        while True:
            steps += 1
            if vec is None:
                _, _, term, trunc, _ = env.step(act(env))
                done = term or trunc
                goal = term
            else:
                obs, _, d, info = vec.step(act(obs))
                done, goal = d[0], info[0]["goal"]
            if done:
                break
        if goal:
            wins += 1
            steps_ok.append(steps)
    return name, wins / episodes, (np.mean(steps_ok) if steps_ok else float("nan"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--fixed", action="store_true")
    a = ap.parse_args()

    results = []
    env = MazeEnv(render_mode=None, fixed=a.fixed)
    # pose helper for the A* controller
    import pybullet as p
    def get_pose():
        pos, orn = p.getBasePositionAndOrientation(env.agent_id)
        return np.array(pos[:2]), p.getEulerFromQuaternion(orn)[2]
    env.get_pose = get_pose

    # PPO: same env, wrapped in the training-time frame stack; stochastic predict
    if os.path.exists("ppo_maze_final.zip"):
        path = "ppo_maze_final"
    else:
        files = glob.glob("logs/*.zip")
        path = max(files, key=os.path.getctime) if files else None
    if path:
        print(f"Loading {path}")
        model = PPO.load(path)
        vec = VecFrameStack(DummyVecEnv([lambda: env]), 4)
        results.append(run("PPO", env, lambda o: model.predict(o, deterministic=False)[0], a.episodes, vec))
    else:
        print("No PPO model found (ppo_maze_final.zip or logs/*.zip); skipping PPO.")

    results.append(run("Random", env, lambda e: e.action_space.sample(), a.episodes))
    results.append(run("A* (cheats: knows map+goal, ignores obstacles; upper reference)", env, astar_action, a.episodes))
    env.close()

    print(f"\n{a.episodes} episodes, seeds {SEED0}.., {'fixed test maze' if a.fixed else 'random mazes'}")
    print(f"{'agent':<70}{'success':>9}{'mean steps (success)':>24}")
    for name, rate, steps in results:
        print(f"{name:<70}{rate:>8.0%}{steps:>24.1f}")

if __name__ == "__main__":
    main()
