"""Fixed evaluation set for MazeNav: held-out maze seeds x maze sizes, several agents, confidence intervals.

Every episode is pinned by one seed in [0, EVAL_SEED_RANGE) (never used for training layouts), so the same
command always scores the same mazes, starts, goals, obstacles and sensor noise.

Agents
  Random        uniform actions
  NodeCount     no map: odometry + lidar + goal only when in line of sight (the information PPO gets)
  Frontier      no ground truth: follows the route over the map it built itself (lidar + odometry)
  Oracle        cheats: knows the layout and the goal (upper reference)
  PPO[...]      one row per --models zip (default: ppo_maze_final / newest logs/*.zip)
"""
import argparse
import csv
import glob
import json
import os
from datetime import datetime
from importlib import metadata

import numpy as np
from stable_baselines3 import PPO

from envs.maze_env import MazeEnv, EVAL_SEED_RANGE
from train import SIZES

DEFAULT_SIZES = SIZES
CARDINALS = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # world E, N, W, S


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def steer(err):
    """Heading error (rad) -> discrete action: 1 = turn left, 2 = turn right, 0 = forward."""
    err = wrap(err)
    return 1 if err > 0.3 else 2 if err < -0.3 else 0


def toward(pos, yaw, target):
    d = np.asarray(target, dtype=float) - pos
    return steer(np.arctan2(d[1], d[0]) - yaw)


def episode_seed(base_seed, size, i):
    seed = base_seed + size * 10_000 + i
    assert seed < EVAL_SEED_RANGE, "evaluation seed would collide with the training layout range"
    return seed


def wilson_ci(k, n, z=1.96):
    """95% Wilson score interval for k successes in n trials."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


# ---------------------------------------------------------------- agents
# reset(env, obs, seed) starts an episode; act(env, obs) returns an action 0/1/2.

class RandomAgent:
    name = "Random"

    def reset(self, env, obs, seed):
        self.rng = np.random.default_rng(seed)

    def act(self, env, obs):
        return int(self.rng.integers(3))


class OracleAgent:
    name = "Oracle (cheats: knows map + goal)"

    def reset(self, env, obs, seed):
        pass

    def act(self, env, obs):
        pos, yaw = env.get_pose()
        start = (int(round(pos[1])) - env.offset, int(round(pos[0])) - env.offset)
        if start == env.goal_cell:
            return toward(pos, yaw, env.goal_pos)
        nxt = env.shortest_path(start, env.goal_cell)[1]
        return toward(pos, yaw, [nxt[1] + env.offset, nxt[0] + env.offset])


class NodeCountAgent:
    """Map-free explorer. Sensors: pose (odometry), the 8 lidar readings and the goal observation, as in obs.
    At each tile centre it steps to the open neighbour it has entered least often (node counting, which
    eventually covers any connected maze); once the goal is in line of sight it drives straight at it."""
    name = "NodeCount (map-free, same sensors)"
    OPEN = 0.9          # lidar reading (m) above which the neighbouring tile counts as open
    AT_CENTRE = 0.2     # m

    def reset(self, env, obs, seed):
        self.rng = np.random.default_rng(seed)
        self.visits = {}
        self.phase = "approach"   # approach: reach the tile centre; go: drive to the chosen neighbour
        self.target = None

    def act(self, env, obs):
        pos, yaw = env.get_pose()
        if obs[1] > 0.5:  # goal visible: obs = [time_left, visible, dist, sin, cos, ...]
            self.phase, self.target = "approach", None
            return steer(np.arctan2(obs[3], obs[4]))

        cur = (int(round(pos[0])), int(round(pos[1])))
        if self.phase == "go":
            if cur != self.target and abs(cur[0] - self.target[0]) + abs(cur[1] - self.target[1]) == 1:
                return toward(pos, yaw, self.target)
            self.phase = "approach"
        if np.linalg.norm(pos - np.array(cur, dtype=float)) > self.AT_CENTRE:
            return toward(pos, yaw, cur)

        # at a tile centre: line up with the nearest axis so the lidar rays point down the corridors
        k = int(round(yaw / (np.pi / 2)))
        misalign = wrap(yaw - k * np.pi / 2)
        if abs(misalign) > 0.2:
            return 1 if misalign < 0 else 2

        lidar = obs[9:17]
        best, best_key = None, None
        for j, (dx, dy) in enumerate(CARDINALS):
            rel = wrap(j * np.pi / 2 - yaw)
            if lidar[(int(round(rel / (np.pi / 4))) + 4) % 8] <= self.OPEN:
                continue
            nxt = (cur[0] + dx, cur[1] + dy)
            key = (self.visits.get(nxt, 0), abs(rel), self.rng.random())  # least visited, then straightest
            if best_key is None or key < best_key:
                best, best_key = nxt, key
        if best is None:
            return 1  # boxed in (e.g. by a patrolling block): spin and look again
        self.visits[cur] = self.visits.get(cur, 0) + 1
        self.phase, self.target = "go", best
        return toward(pos, yaw, best)


class FrontierAgent:
    """Uses only the agent's own map (lidar + odometry, as in the observation): drives straight at the goal once
    it is in line of sight, otherwise follows the BFS route to the remembered goal / nearest unexplored tile.
    Shows the map memory carries enough information to solve every maze; PPO has to learn to use it."""
    name = "Frontier (own map, same sensors)"
    AT_CENTRE = 0.2     # m

    def reset(self, env, obs, seed):
        pass

    def act(self, env, obs):
        pos, yaw = env.get_pose()
        if obs[1] > 0.5:  # goal visible
            return steer(np.arctan2(obs[3], obs[4]))
        wp = env.waypoint()
        if wp is None:
            return 1  # nothing known to head for (e.g. boxed in by a patrolling block): spin and rescan
        cur = pos.round()
        # corner cutting would scrape the walls: when the route turns, first centre on the current tile
        if np.linalg.norm(pos - cur) > self.AT_CENTRE and abs(wrap(np.arctan2(wp[1] - pos[1], wp[0] - pos[0]) - yaw)) > np.pi / 3:
            return toward(pos, yaw, cur)
        return toward(pos, yaw, wp)


class PPOAgent:
    """Wraps a trained model with the same frame stacking as VecFrameStack (oldest frame first, zero-filled)."""

    def __init__(self, path, deterministic=False):
        self.path = path
        self.model = PPO.load(path, device="cpu")
        self.deterministic = deterministic
        self.level = getattr(self.model, "curriculum_level", None)
        self.name = f"PPO[{os.path.splitext(os.path.basename(path))[0]}]"

    def reset(self, env, obs, seed):
        import torch
        torch.manual_seed(seed)
        self.dim = obs.shape[0]
        self.stack = np.zeros(self.model.observation_space.shape[0], dtype=np.float32)
        self.fresh = True

    def act(self, env, obs):
        if self.fresh:
            self.fresh = False
        else:
            self.stack = np.roll(self.stack, -self.dim)
        self.stack[-self.dim:] = obs
        return int(self.model.predict(self.stack, deterministic=self.deterministic)[0])


# ---------------------------------------------------------------- running
def run_episode(env, agent, size, seed):
    obs, _ = env.reset(seed=seed, options={"size": size, "layout_seed": seed})
    agent.reset(env, obs, seed)
    shortest = len(env.shortest_path(env.start_cell, env.goal_cell)) - 1  # tiles are 1 m apart
    prev = env.get_pose()[0]
    dist, hits, steps, success = 0.0, 0, 0, False
    while True:
        obs, _, term, trunc, info = env.step(agent.act(env, obs))
        steps += 1
        pos = env.get_pose()[0]
        dist += float(np.linalg.norm(pos - prev))
        prev = pos
        hits += int(info["hit_obstacle"])
        if term or trunc:
            success = bool(info["goal"])
            break
    spl = shortest / max(shortest, dist) if success else 0.0
    return dict(size=size, seed=seed, success=int(success), steps=steps, spl=spl, shortest=shortest,
                path_len=dist, obstacle_hits=hits)


def summarise(rows):
    n = len(rows)
    k = sum(r["success"] for r in rows)
    lo, hi = wilson_ci(k, n)
    ok = [r["steps"] for r in rows if r["success"]]
    return dict(n=n, successes=k, success=k / n, ci_lo=lo, ci_hi=hi,
                steps_ok=float(np.mean(ok)) if ok else float("nan"),
                spl=float(np.mean([r["spl"] for r in rows])),
                hits=float(np.mean([r["obstacle_hits"] for r in rows])))


def bootstrap_mean_ci(values, n_boot=10_000, seed=0):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, len(values))).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def find_models(paths):
    if paths:
        return paths
    if os.path.exists("ppo_maze_final.zip"):
        return ["ppo_maze_final.zip"]
    files = glob.glob("logs/*.zip")
    return [max(files, key=os.path.getctime)] if files else []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=200, help="episodes per maze size")
    ap.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--models", nargs="*", help="PPO zips, e.g. one per training seed")
    ap.add_argument("--deterministic", action="store_true", help="argmax actions for PPO (default: sample)")
    ap.add_argument("--agents", nargs="+", default=["random", "nodecount", "frontier", "oracle", "ppo"],
                    choices=["random", "nodecount", "frontier", "oracle", "ppo"])
    ap.add_argument("--fixed", action="store_true", help="seed-0 test maze instead of seeded random mazes")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    agents = []
    if "random" in a.agents:
        agents.append(RandomAgent())
    if "nodecount" in a.agents:
        agents.append(NodeCountAgent())
    if "frontier" in a.agents:
        agents.append(FrontierAgent())
    if "oracle" in a.agents:
        agents.append(OracleAgent())
    ppo_agents = []
    if "ppo" in a.agents:
        ppo_agents = [PPOAgent(m, a.deterministic) for m in find_models(a.models)]
        if not ppo_agents:
            print("No PPO model found (ppo_maze_final.zip, logs/*.zip or --models); skipping PPO.")
        for g in ppo_agents:
            lvl = f"{SIZES[g.level]}x{SIZES[g.level]}" if g.level is not None else "unknown (pre-checkpointed level)"
            print(f"Loaded {g.path}: curriculum reached maze size {lvl}")
        agents += ppo_agents

    env = MazeEnv(render_mode=None, fixed=a.fixed)
    rows = []
    try:
        for agent in agents:
            for size in a.sizes:
                for i in range(a.episodes):
                    r = run_episode(env, agent, size, episode_seed(a.base_seed, size, i))
                    rows.append(dict(agent=agent.name, **r))
                s = summarise([r for r in rows if r["agent"] == agent.name and r["size"] == size])
                print(f"  {agent.name:<40} size {size:>2}: {s['success']:.0%} ({s['successes']}/{s['n']})", flush=True)
    finally:
        env.close()

    # ---- tables
    names = [g.name for g in agents]
    table = []  # (agent, size or "all", summary)
    for name in names:
        for size in a.sizes + ["all"]:
            sel = [r for r in rows if r["agent"] == name and (size == "all" or r["size"] == size)]
            table.append((name, size, summarise(sel)))
    if len(ppo_agents) > 1:
        for size in a.sizes + ["all"]:
            per_model = [s["success"] for n_, sz, s in table if n_ in {g.name for g in ppo_agents} and sz == size]
            lo, hi = bootstrap_mean_ci(per_model)
            table.append((f"PPO mean of {len(ppo_agents)} models", size,
                          dict(n=a.episodes * len(ppo_agents), successes=float("nan"), success=float(np.mean(per_model)),
                               ci_lo=lo, ci_hi=hi, steps_ok=float("nan"), spl=float("nan"), hits=float("nan"))))

    print(f"\n{a.episodes} episodes per size, seeds {a.base_seed}+size*10000+i (held out from training), "
          f"{'fixed test maze' if a.fixed else 'seeded random mazes'}")
    print(f"{'agent':<42}{'size':>5}{'success':>9}{'95% CI':>15}{'steps(ok)':>11}{'SPL':>7}{'hits':>7}")
    for name, size, s in table:
        print(f"{name:<42}{size:>5}{s['success']:>9.1%}  [{s['ci_lo']:>5.1%},{s['ci_hi']:>6.1%}]"
              f"{s['steps_ok']:>11.0f}{s['spl']:>7.2f}{s['hits']:>7.2f}")

    # ---- files
    os.makedirs(a.out, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(a.out, f"eval_{stamp}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    versions = {}
    for pkg in ("numpy", "torch", "stable-baselines3", "gymnasium", "pybullet"):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    json_path = os.path.join(a.out, f"eval_{stamp}.json")
    with open(json_path, "w") as f:
        json.dump(dict(config=dict(vars(a), models=[g.path for g in ppo_agents],
                                   curriculum_levels={g.name: g.level for g in ppo_agents}),
                       versions=versions,
                       summary=[dict(agent=n, size=sz, **s) for n, sz, s in table]), f, indent=2)
    print(f"\nWrote {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
