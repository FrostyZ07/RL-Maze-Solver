import glob
import os
from stable_baselines3 import PPO
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

    env = MazeEnv(render_mode='human')
    try:
        obs, _ = env.reset()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated:
                print(f"Goal reached in {env.step_count} steps!")
                break
            if truncated:
                print(f"Failed: hit {env.max_steps}-step limit.")
                break
    finally:
        env.close()

if __name__ == "__main__":
    solve_maze()
