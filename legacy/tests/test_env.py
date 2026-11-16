from collections import deque

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

import envs  # noqa: F401  (registers MazeNav-v0)
from envs.maze_env import MazeEnv, EVAL_SEED_RANGE
from train import Curriculum, SIZES

# MazeEnv talks to PyBullet's default client, so only one env may be alive per process: always close.


@pytest.fixture
def env():
    e = MazeEnv(render_mode=None)
    yield e
    e.close()


OBS_DIM = 17 + 4 + 49


def test_spaces(env):
    assert env.observation_space.shape == (OBS_DIM,)
    assert isinstance(env.action_space, spaces.Discrete) and env.action_space.n == 3


def test_registered_env_passes_check_env():
    e = gym.make("MazeNav-v0", size=7)
    try:
        check_env(e.unwrapped)
        assert e.unwrapped.size == 7
    finally:
        e.close()


def test_reset_and_step_types(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32
    assert isinstance(info, dict)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs.shape == (OBS_DIM,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert "goal" in info and "hit_obstacle" in info


def test_same_seed_and_options_give_same_episode(env):
    opts = {"size": 9, "layout_seed": 123}
    o1, _ = env.reset(seed=5, options=opts)
    snap1 = (env.layout, env.start_cell, env.goal_cell, env.max_steps)
    o2, _ = env.reset(seed=5, options=opts)
    snap2 = (env.layout, env.start_cell, env.goal_cell, env.max_steps)
    assert snap1 == snap2
    assert np.array_equal(o1, o2)


def test_time_budget_does_not_depend_on_the_route(env):
    """time_left must not leak how far the hidden goal is: the budget is a function of maze size only."""
    budgets, route_lengths = set(), set()
    for seed in range(20):
        env.reset(seed=seed, options={"size": 11, "layout_seed": seed})
        budgets.add(env.max_steps)
        route_lengths.add(len(env.shortest_path(env.start_cell, env.goal_cell)))
    assert len(route_lengths) > 1, "test is only meaningful if routes differ"
    assert len(budgets) == 1


def test_time_budget_scales_with_size_and_open_tiles(env):
    by_size = []
    for size in (7, 9, 11, 13, 15):
        env.reset(seed=1, options={"size": size, "layout_seed": 1})
        open_tiles = sum(row.count(0) for row in env.layout)
        assert env.max_steps == env.STEPS_PER_TILE * open_tiles
        by_size.append(env.max_steps)
    assert by_size == sorted(by_size) and len(set(by_size)) == 5


def test_training_layouts_are_held_out_from_evaluation(env):
    for _ in range(10):
        env.reset()
        assert env.layout_seed >= EVAL_SEED_RANGE
    env.reset(seed=3, options={"layout_seed": 77})
    assert env.layout_seed == 77 < EVAL_SEED_RANGE


@pytest.mark.parametrize("size", [7, 9, 11, 13, 15])
def test_every_maze_is_solvable(env, size):
    for seed in range(10):
        env.reset(seed=seed, options={"size": size, "layout_seed": seed})
        path = env.shortest_path(env.start_cell, env.goal_cell)
        assert path is not None
        assert path[0] == env.start_cell and path[-1] == env.goal_cell
        assert env.start_cell != env.goal_cell


def test_own_map_never_contradicts_the_layout(env):
    """The agent's map is built from lidar only; whatever it marks must agree with the true layout."""
    for seed in range(5):
        env.reset(seed=seed, options={"size": 11, "layout_seed": seed})
        rng = np.random.default_rng(seed)
        for _ in range(300):
            env.step(int(rng.integers(3)))
        layout = np.array(env.layout)
        assert not np.any((env.known == 1) & (layout == 0)), "open tile marked wall"
        assert not np.any((env.known == 0) & (layout == 1)), "wall tile marked open"
        assert (env.known != -1).sum() > 1


def test_resizing_mid_episode_waits_for_the_next_reset(env):
    """The curriculum calls set_size while episodes are running; the running one must keep working."""
    env.reset(seed=1, options={"size": 7, "layout_seed": 1})
    env.set_size(11)
    for _ in range(50):
        obs, *_ = env.step(0)
        assert obs.shape == (OBS_DIM,)
    assert env.size == 7 and env.known.shape == (7, 7)
    env.reset(seed=2)
    assert env.size == 11 and env.known.shape == (11, 11)


def test_agent_knocked_outside_the_maze_does_not_crash(env):
    import pybullet as p
    env.reset(seed=1, options={"size": 7, "layout_seed": 1})
    p.resetBasePositionAndOrientation(env.agent_id, [-40.0, 25.0, 0.5], [0, 0, 0, 1])
    obs, *_ = env.step(0)
    assert obs.shape == (OBS_DIM,) and np.isfinite(obs).all()


def test_route_cue_exists_until_everything_is_explored(env):
    env.reset(seed=3, options={"size": 7, "layout_seed": 3})
    assert env.waypoint() is not None
    env.known[:] = np.where(np.array(env.layout) == 0, 0, 1)   # fully mapped...
    env.searched[:] = True                                      # ...and fully searched
    assert env.waypoint() is None
    env.searched[env.goal_cell] = False                         # one tile left to check, far from the start
    assert env.waypoint() is not None


def test_fixed_mode_has_one_layout():
    e = MazeEnv(render_mode=None, fixed=True)
    try:
        e.reset(seed=0)
        first = e.layout
        e.reset(seed=1)
        assert e.layout == first and e.layout_seed == 0
    finally:
        e.close()


def test_curriculum_level_survives_save_and_load(tmp_path):
    venv = DummyVecEnv([lambda: MazeEnv(render_mode=None, size=SIZES[0])])
    try:
        model = PPO("MlpPolicy", venv, device="cpu", n_steps=8, batch_size=8)
        model.curriculum_level = 3
        model.save(tmp_path / "ckpt")

        loaded = PPO.load(tmp_path / "ckpt", env=venv, device="cpu")
        assert loaded.curriculum_level == 3

        cb = Curriculum()
        cb.init_callback(loaded)
        cb._on_training_start()                       # what --resume does
        assert venv.get_attr("next_size")[0] == SIZES[3]

        loaded.ep_info_buffer = deque([{"goal": 1}] * 100, maxlen=100)   # 100% win rate -> promote
        cb._on_rollout_end()
        assert loaded.curriculum_level == 4
        assert venv.get_attr("next_size")[0] == SIZES[4]
        assert len(loaded.ep_info_buffer) == 0
    finally:
        venv.close()
