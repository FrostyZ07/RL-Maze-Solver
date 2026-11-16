import gymnasium as gym
from stable_baselines3 import PPO
from envs.maze_env import MazeEnv
import time

def test():
    # Load model
    # Find latest checkpoint in logs/
    import glob
    import os
    
    list_of_files = glob.glob('logs/*.zip') 
    if not list_of_files:
        print("No checkpoints found in logs/. Checking ppo_maze_final.zip...")
        model_path = "ppo_maze_final"
    else:
        latest_file = max(list_of_files, key=os.path.getctime)
        print(f"Loading latest checkpoint: {latest_file}")
        model_path = latest_file

    try:
        model = PPO.load(model_path)
    except FileNotFoundError:
        print("Model not found. Please train first.")
        return

    # Create environment with GUI
    env = MazeEnv(render_mode='human')
    
    try:
        obs, _ = env.reset()
        while True:
            action, _states = model.predict(obs, deterministic=True)
            # Obs is now [dist, sin, cos, lidar...]
            # We can't easily get absolute pos from obs anymore, but we can print action name
            action_names = {0: "Forward", 1: "Left", 2: "Right"}
            print(f"Dist: {obs[0]:.2f} | Heading: {obs[1]:.2f}, {obs[2]:.2f} | Action: {action_names[int(action)]}")
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                obs, _ = env.reset()
    except (KeyboardInterrupt, Exception) as e:
        print(f"Simulation stopped: {e}")
    finally:
        env.close()

if __name__ == "__main__":
    test()
