import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
import os
import time
from envs.maze_env import MazeEnv

N_ENVS = 8

def make_env():
    # each subprocess gets its own PyBullet connection; Monitor records ep_rew_mean / ep_len_mean
    return Monitor(MazeEnv(render_mode=None))

def train():
    # Create log dir
    log_dir = "logs/"
    os.makedirs(log_dir, exist_ok=True)

    # Check if environment follows Gym API (separate throwaway instance)
    check_env(MazeEnv(render_mode=None))

    # 8 parallel envs, each a new random maze + goal every episode; stack the last 4 frames
    env = VecFrameStack(SubprocVecEnv([make_env] * N_ENVS), 4)

    # Always from scratch: obs size changed (17), old checkpoints are incompatible
    # n_steps is per env: 8 * 512 = 4096 samples per update
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, device="cpu",
                n_steps=512, batch_size=256, gamma=0.995, ent_coef=0.03,
                policy_kwargs=dict(net_arch=[256, 256]))

    # Checkpoint callback (save_freq counts per-env steps)
    checkpoint_callback = CheckpointCallback(save_freq=10000 // N_ENVS, save_path=log_dir, name_prefix="ppo_maze")

    print("Starting training...")
    total_timesteps = 5_000_000
    try:
        model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback, tb_log_name="PPO_run")
        print("Training finished.")
    except KeyboardInterrupt:
        print("Interrupted, saving current model.")

    # Save final model
    model.save("ppo_maze_final")
    env.close()

if __name__ == "__main__":
    train()
