import gymnasium as gym
import pytest
import numpy as np
from envs.maze_env import MazeEnv

def test_maze_env_initialization():
    env = MazeEnv(render_mode=None)
    assert env.observation_space.shape[0] == 14
    assert env.action_space.shape[0] == 2
    env.close()

def test_maze_env_reset():
    env = MazeEnv(render_mode=None)
    obs, info = env.reset()
    assert obs.shape[0] == 14
    assert isinstance(info, dict)
    env.close()

def test_maze_env_step():
    env = MazeEnv(render_mode=None)
    env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape[0] == 14
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    env.close()
