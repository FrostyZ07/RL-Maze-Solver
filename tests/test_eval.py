import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from envs.maze_env import MazeEnv, EVAL_SEED_RANGE
from evaluate import (wilson_ci, episode_seed, run_episode, summarise, RandomAgent, NodeCountAgent,
                      OracleAgent, PPOAgent, FrontierAgent)


@pytest.fixture
def env():
    e = MazeEnv(render_mode=None)
    yield e
    e.close()


def test_wilson_ci_known_values():
    lo, hi = wilson_ci(40, 100)
    assert (lo, hi) == pytest.approx((0.3094, 0.4980), abs=1e-3)
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2776, abs=1e-3)
    assert wilson_ci(10, 10)[1] == pytest.approx(1.0, abs=1e-6)
    assert np.isnan(wilson_ci(0, 0)[0])


def test_episode_seeds_are_unique_and_held_out():
    seeds = [episode_seed(0, s, i) for s in (7, 9, 11, 13, 15) for i in range(200)]
    assert len(set(seeds)) == len(seeds)
    assert max(seeds) < EVAL_SEED_RANGE
    with pytest.raises(AssertionError):
        episode_seed(EVAL_SEED_RANGE, 7, 0)


def test_episode_is_reproducible(env):
    a = run_episode(env, RandomAgent(), 7, 5)
    b = run_episode(env, RandomAgent(), 7, 5)
    assert a == b


def test_own_map_memory_is_enough_to_solve_every_maze(env):
    """FrontierAgent sees nothing the observation does not contain, so 100% here means the memory suffices."""
    for size in (7, 15):
        for i in range(20):
            r = run_episode(env, FrontierAgent(), size, episode_seed(0, size, i))
            assert r["success"] == 1, f"size {size} seed {r['seed']} failed after {r['steps']} steps"


@pytest.mark.parametrize("agent_cls", [RandomAgent, NodeCountAgent, FrontierAgent, OracleAgent])
def test_agents_return_well_formed_rows(env, agent_cls):
    rows = [run_episode(env, agent_cls(), 7, episode_seed(0, 7, i)) for i in range(2)]
    for r in rows:
        assert r["success"] in (0, 1)
        assert 0.0 <= r["spl"] <= 1.0
        assert 1 <= r["steps"] <= 15 * 21 + 1
        assert r["shortest"] >= 1
    s = summarise(rows)
    assert s["n"] == 2 and 0.0 <= s["ci_lo"] <= s["success"] <= s["ci_hi"] <= 1.0


def test_ppo_agent_stacking_matches_vecframestack(tmp_path):
    actions = [0, 0, 1, 1, 0, 2]

    venv = VecFrameStack(DummyVecEnv([lambda: MazeEnv(render_mode=None, size=7)]), 4)
    venv.seed(3)
    recorded = [venv.reset()[0].copy()]
    for a in actions:
        recorded.append(venv.step(np.array([a]))[0][0].copy())
    PPO("MlpPolicy", venv, device="cpu", n_steps=8, batch_size=8).save(tmp_path / "m")
    venv.close()

    env = MazeEnv(render_mode=None, size=7)
    try:
        agent = PPOAgent(str(tmp_path / "m.zip"))
        obs, _ = env.reset(seed=3)
        agent.reset(env, obs, 3)
        for t, a in enumerate([None] + actions):
            if a is not None:
                obs = env.step(a)[0]
            agent.act(env, obs)
            assert np.allclose(agent.stack, recorded[t], atol=1e-6), f"stack differs at step {t}"
    finally:
        env.close()
