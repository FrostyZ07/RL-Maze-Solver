import glob
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from envs.maze_env import MazeEnv

def solve_maze():
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
    env = VecFrameStack(DummyVecEnv([lambda: MazeEnv(render_mode='human')]), 4)
    try:
        obs = env.reset()
        steps = successes = episodes = 0
        while episodes < 10:
            action, _ = model.predict(obs, deterministic=False)
            obs, _, done, info = env.step(action)
            steps += 1
            if done[0]:
                episodes += 1
                successes += bool(info[0]["goal"])
                print(f"Goal reached in {steps} steps!" if info[0]["goal"]
                      else f"Failed: ran out of time after {steps} steps.",
                      f"[{successes}/{episodes}]")
                steps = 0
    finally:
        env.close()

if __name__ == "__main__":
    solve_maze()
