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
    
    # Initialize environment
    env = MazeEnv(render_mode=None) # No GUI for training speed
    
    # Check if environment follows Gym API
    check_env(env)

    # Stack the last 4 lidar frames so the agent can perceive obstacle motion
    # (same env instance: PyBullet module calls share one default client)
    # Monitor records episode reward/length -> rollout/ep_rew_mean, ep_len_mean in the logs
    env = VecFrameStack(DummyVecEnv([lambda: Monitor(env)]), 4)

    # ent_coef=0.01 is better for Discrete actions (max entropy is ~1.1)
    resume = os.path.exists("ppo_maze_final.zip")
    if resume:
        # Continue from the saved model (same hyperparameters are restored)
        model = PPO.load("ppo_maze_final", env=env, device="cpu", tensorboard_log=log_dir)
        ent_coef = 0.01
    else:
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, device="cpu",
                    n_steps=2048, batch_size=64, gamma=0.99, ent_coef=0.01)

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=log_dir, name_prefix="ppo_maze")
    
    print("Starting training...")
    # Train
    # Short training for PoC, can be increased
    total_timesteps = 300000
    try:
        model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback, tb_log_name="PPO_run",
                    reset_num_timesteps=not resume)
        print("Training finished.")
    except KeyboardInterrupt:
        # Ctrl+C: keep what was learned so far
        print("Interrupted, saving current model.")
    
    # Save final model
    model.save("ppo_maze_final")
    env.close()

if __name__ == "__main__":
    train()
