import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import os
import time
from envs.maze_env import MazeEnv

def train():
    # Create log dir
    log_dir = "logs/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize environment: a new random maze + goal every episode
    env = MazeEnv(render_mode=None) # No GUI for training speed
    
    # Check if environment follows Gym API
    check_env(env)

    # Stack the last 4 lidar frames so the agent can perceive obstacle motion
    # (same env instance: PyBullet module calls share one default client)
    # Monitor records episode reward/length -> rollout/ep_rew_mean, ep_len_mean in the logs
    env = VecFrameStack(DummyVecEnv([lambda: Monitor(env)]), 4)

    # ent_coef=0.01 is better for Discrete actions (max entropy is ~1.1)
    # Always from scratch: obs size changed (36), old checkpoints are incompatible
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, device="cpu",
                n_steps=2048, batch_size=64, gamma=0.99, ent_coef=0.03)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=log_dir, name_prefix="ppo_maze")
    
    print("Starting training...")
    total_timesteps = 2_000_000
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback, tb_log_name="PPO_run")
    
    print("Training finished.")
    
    # Save final model
    model.save("ppo_maze_final")
    env.close()

if __name__ == "__main__":
    train()
