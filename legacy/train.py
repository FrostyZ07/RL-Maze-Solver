import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
import os
import envs  # noqa: F401  (registers MazeNav-v0)

N_ENVS = 8
SIZES = [7, 9, 11, 13, 15]  # curriculum: maze size per level
PROMOTE_AT = 0.9            # recent success rate needed to move up (the map memory makes near-perfect play possible)

class Curriculum(BaseCallback):
    """Grow the maze once the agent wins >= PROMOTE_AT of its last 100 episodes.

    The level lives on the model (model.curriculum_level), and SB3 pickles the model's attributes in
    every save, so each checkpoint carries it and --resume continues at the same maze size."""
    def __init__(self, promote_at=PROMOTE_AT):
        super().__init__()
        self.promote_at = promote_at

    def _on_training_start(self):
        level = getattr(self.model, "curriculum_level", 0)
        self.model.curriculum_level = level
        self.training_env.env_method("set_size", SIZES[level])
        print(f"Curriculum: starting at maze size {SIZES[level]} (level {level})")

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        level = self.model.curriculum_level
        buf = self.model.ep_info_buffer
        if level < len(SIZES) - 1 and len(buf) >= 100 and sum(e["goal"] for e in buf) / len(buf) >= self.promote_at:
            level += 1
            self.model.curriculum_level = level
            self.training_env.env_method("set_size", SIZES[level])
            buf.clear()
            print(f"Curriculum: maze size -> {SIZES[level]}")

def make_env():
    # each subprocess gets its own PyBullet connection; Monitor records ep_rew_mean / ep_len_mean
    return Monitor(gym.make("MazeNav-v0", disable_env_checker=True), info_keywords=("goal",))

# PPO hyperparameters; the defaults reproduce the original run. On --resume, any value passed explicitly overrides the saved one.
HPARAM_DEFAULTS = dict(learning_rate=3e-4, n_steps=512, batch_size=256, gamma=0.995, gae_lambda=0.95,
                       ent_coef=0.01, clip_range=0.2, n_epochs=10, target_kl=None)

def linear(start, done_before, total):
    """Decay start -> 0 over this run's `total` steps. SB3's progress_remaining counts from step 0 on --resume,
    so rebase it onto the `done_before` steps already in the checkpoint."""
    def schedule(progress_remaining):
        done = (1 - progress_remaining) * (total + done_before) - done_before
        return start * max(0.0, 1 - done / total)
    return schedule

def train(seed=0, total_timesteps=5_000_000, resume=None, hp=None, lr_schedule="constant", promote_at=PROMOTE_AT, name="ppo_maze"):
    hp = {k: v for k, v in (hp or {}).items() if v is not None or k == "target_kl"}
    lr_start = hp.get("learning_rate") or HPARAM_DEFAULTS["learning_rate"]
    # Create log dir
    log_dir = "logs/"
    os.makedirs(log_dir, exist_ok=True)

    # Check if environment follows Gym API (separate throwaway instance)
    probe = gym.make("MazeNav-v0")
    check_env(probe.unwrapped)
    probe.close()

    # 8 parallel envs, each a new random maze + goal every episode; stack the last 4 frames
    env = VecFrameStack(SubprocVecEnv([make_env] * N_ENVS), 4)

    if resume:
        # keeps weights, optimiser state and curriculum level; --timesteps is then added on top of the saved count
        model = PPO.load(resume, env=env, device="cpu", tensorboard_log=log_dir,
                         custom_objects={k: v for k, v in hp.items() if v is not None})
        if lr_schedule == "linear":
            model.learning_rate = linear(lr_start, model.num_timesteps, total_timesteps)
            model._setup_lr_schedule()
        print(f"Resumed {resume} at {model.num_timesteps} steps")
    else:
        if lr_schedule == "linear":
            hp["learning_rate"] = linear(lr_start, 0, total_timesteps)
        # n_steps is per env: 8 * 512 = 4096 samples per update
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, device="cpu", seed=seed,
                    **{**HPARAM_DEFAULTS, **hp},
                    policy_kwargs=dict(net_arch=[256, 256]))

    # Checkpoint callback (save_freq counts per-env steps)
    checkpoint_callback = CheckpointCallback(save_freq=10000 // N_ENVS, save_path=log_dir, name_prefix=name)

    print("Starting training...")
    try:
        model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, Curriculum(promote_at)],
                    tb_log_name="PPO_run", reset_num_timesteps=resume is None)
        print("Training finished.")
    except KeyboardInterrupt:
        print("Interrupted, saving current model.")

    # Save final model
    model.save(f"{name}_final")
    env.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="seeds PPO, torch and the 8 envs")
    ap.add_argument("--timesteps", type=int, default=5_000_000)
    ap.add_argument("--resume", metavar="ZIP", help="continue from a checkpoint (e.g. logs/ppo_maze_100000_steps.zip)")
    ap.add_argument("--name", default="ppo_maze", help="checkpoint prefix and <name>_final.zip; use a new one when tuning so ppo_maze_final.zip is not overwritten")
    ap.add_argument("--promote-at", type=float, default=PROMOTE_AT, help="win rate over the last 100 episodes needed to grow the maze")
    ap.add_argument("--lr", dest="learning_rate", type=float, help=f"learning rate (default {HPARAM_DEFAULTS['learning_rate']})")
    ap.add_argument("--lr-schedule", choices=["constant", "linear"], default="constant", help="linear decays --lr to 0 over this run")
    ap.add_argument("--n-steps", type=int, help="rollout length per env")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--gamma", type=float)
    ap.add_argument("--gae-lambda", type=float)
    ap.add_argument("--ent-coef", type=float)
    ap.add_argument("--clip-range", type=float)
    ap.add_argument("--n-epochs", type=int)
    ap.add_argument("--target-kl", type=float, help="early-stop a PPO update when KL exceeds this (default off)")
    a = ap.parse_args()
    hp = {k: getattr(a, k) for k in HPARAM_DEFAULTS}
    train(a.seed, a.timesteps, a.resume, hp, a.lr_schedule, a.promote_at, a.name)
