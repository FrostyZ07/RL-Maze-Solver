import argparse
import glob
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from envs.maze_env import MazeEnv

def solve_maze(fixed=False, episodes=10):
    # Final model if present, else newest checkpoint
    if os.path.exists("ppo_maze_final.zip"):
        model_path = "ppo_maze_final"
    else:
        files = glob.glob("logs/*.zip")
        if not files:
            print("No trained model found. Run train.py first.")
            return
        model_path = max(files, key=os.path.getctime)
    print(f"Loading {model_path}")
    model = PPO.load(model_path)

    # Same 4-frame stack as training; the vec env auto-resets when an episode ends
    env = VecFrameStack(DummyVecEnv([lambda: MazeEnv('human', fixed=fixed)]), 4)
    try:
        obs = env.reset()
        successes = 0
        for ep in range(episodes):
            steps = 0
            while True:
                action, _ = model.predict(obs, deterministic=False)
                obs, _, done, info = env.step(action)
                steps += 1
                if done[0]:
                    successes += int(info[0]["goal"])
                    print(f"Episode {ep + 1}: " + (f"goal reached in {steps} steps!" if info[0]["goal"]
                          else f"failed, ran out of time after {steps} steps.")
                          + f" ({successes}/{ep + 1} successes)")
                    break
    finally:
        env.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed", action="store_true", help="the seed-0 test maze instead of a random one")
    solve_maze(ap.parse_args().fixed)
