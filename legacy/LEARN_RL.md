# Learn RL (PPO) Through This Maze Project

Goal of this file: understand how the agent learns, read the training logs yourself, and know what to change when it isn't working.

Files that matter: `envs/maze_env.py` (the world + rewards), `train.py` (PPO settings), `solve_maze.py` (watch the trained agent).

---

## 1. The big idea (rat in a maze)

A rat is dropped in a maze. Nobody gives it a map. It tries things, gets consequences, and slowly turns "what worked" into habit.

| Rat | This project | Where in code |
|---|---|---|
| Eyes / whiskers | **Observation**: 35 numbers (distance to goal, goal direction, 32 laser distances to walls) | `_get_obs()` in `maze_env.py` |
| Muscles | **Action**: 0 forward, 1 turn left, 2 turn right | `step()` |
| Cheese / bumped nose / hunger | **Reward** (score after every action) | `step()` |
| One trip through the maze | **Episode** (ends at goal or after 500 actions) | `max_steps` |
| The brain | **Neural network** (the "policy") | created in `train.py` by `PPO("MlpPolicy", ...)` |
| Forming habits | **PPO** (the learning algorithm) | `model.learn(...)` in `train.py` |

The key rule: **the agent is never told the route.** It only gets scores. Everything it knows, it discovered.

---

## 2. The reward: what you are really teaching it

All in `step()` of `maze_env.py`:

| Event | Reward | What it teaches |
|---|---|---|
| Every action | -0.01 | "Don't waste time" |
| First time entering a tile | +0.5 | "Exploring is good" |
| Touching a wall | -0.01 | "Walls are bad (mildly)" |
| Reaching the goal | +100 | "This is the whole point" |

Handy reference numbers:
- Open tiles in the maze: about 34, so the max from exploring alone is about +17.
- A perfect run (about 80-100 actions): roughly -1 (time) + ~7 (tiles on the way) + 100 = **about 100-107**.
- A useless run, 500 actions with no goal: about **-5** (just the time penalty), or worse if hugging walls.

**Why this matters:** any `ep_rew_mean` above about +17 *must* include goal hits. That's how you can tell the agent is solving it just from one number.

The reward is the only lever that defines "good". Change it and you change what the agent tries to become.

---

## 3. How PPO learns (the part to really understand)

PPO runs a cycle over and over:

```
1. COLLECT   play 2048 actions using the current brain (random-ish choices),
             recording: what it saw, what it did, what reward came.
2. JUDGE     for each action: was the result better or worse than expected?
3. UPDATE    nudge the brain: make better-than-expected actions more likely,
             worse-than-expected actions less likely.
4. REPEAT    each cycle = one "iteration" = one printed log table.
```

500,000 total steps / 2048 per iteration is about **244 iterations**.

### The two pieces inside the brain
- **Actor (policy):** looks at the 35 numbers and outputs probabilities, e.g. forward 70% / left 20% / right 10%. It *rolls dice* with those odds, so it explores.
- **Critic (value function):** looks at the same numbers and predicts "how much total reward will I get from here?" It's the agent's sense of "this spot feels promising".

### "Better or worse than expected" (the core trick)
```
advantage = what actually happened afterward  -  what the critic predicted
```
- Advantage **positive**: action was a pleasant surprise, so make it more likely.
- Advantage **negative**: action was a letdown, so make it less likely.

This is why a single lucky goal hit (+100) matters so much. It's a huge pleasant surprise, so everything that led up to it gets boosted.

### Why "Proximal" (the safety belt)
One update is only allowed to change the policy a *little* (the `clip_range`, default 0.2). Without this, one weird batch can wreck a good policy. Small safe steps are the whole reason PPO is popular.

### Looking ahead: `gamma`
A reward N steps in the future is worth `gamma^N` of its full value (`gamma=0.99`):
- 10 steps away: about 90%
- 100 steps away: about 37%
- 300 steps away: about 5%

So the +100 at the end still "reaches back" through a ~100-step route. A smaller `gamma` makes the agent short-sighted.

### Exploration vs. habit: `ent_coef`
"Entropy" = how random the choices are. With 3 actions, fully random = **1.0986** (that's ln 3, the maximum). Fully decisive = 0.
`ent_coef` is a bonus for staying random:
- Too low: it grabs the first "okay" habit and never finds better ones (e.g. spinning in place).
- Too high: it stays random forever and never settles on the route.

---

## 4. What the three learning phases look like

1. **Flailing:** policy is near random. Bumps walls, wastes time. `ep_rew_mean` slightly negative (-5 to -10), `ep_len_mean` = 500.
2. **Exploring:** it learns "forward in open space is good, walls are bad." `ep_rew_mean` rises to a few units positive, capped around +17.
3. **Goal found:** one lucky run hits +100, the habit spreads. `ep_rew_mean` jumps (30, 60, 90...) and `ep_len_mean` falls well below 500. Then it smooths out near the "perfect run" score of about 100.

Phase 3 often looks sudden. That's normal. Before the first success there's no signal to follow.

---

## 5. Reading the log table (with ranges)

Printed every 2048 steps. Here is every line, what it means, and what's healthy.

### `rollout/` - real results from episodes just played (MOST IMPORTANT)

| Metric | Meaning | Healthy |
|---|---|---|
| `ep_rew_mean` | Average total score per episode | Trending **up**. >17 means goal is being reached. ~100+ means solved. |
| `ep_len_mean` | Average actions per episode | Starts 500 (timeouts). Trending **down**. ~80-150 = efficient route. |

Rule of thumb: **reward up + length down = learning.** Stuck at 500 and ~-5 = never found the goal.

### `time/`

| Metric | Meaning |
|---|---|
| `total_timesteps` | Actions taken so far (goal: 500,000) |
| `iterations` | Number of 2048-step cycles done |
| `fps` | Actions per second. Just speed; higher = faster training |

### `train/` - inside the brain (use for diagnosing, not for judging progress)

| Metric | Meaning | Typical range / how to read |
|---|---|---|
| `entropy_loss` | Negative of randomness | Starts about **-1.09** (random). Drifts slowly toward -0.3 to -0.6 as it commits. If it plunges to about -0.05 early, it locked into one habit too soon. If still about -1.09 after many iterations, it isn't learning a preference. |
| `explained_variance` | How well the critic predicts outcomes | **≤ 1**. Near 0 or negative early = fine. Rising toward 0.5-0.95 = the agent understands the maze. Staying negative for long = trouble. |
| `value_loss` | Critic's prediction error | Large and spiky when new things are discovered (the goal), shrinking later. Scale depends on reward size (our +100 makes it big, e.g. tens to hundreds). Don't panic at big numbers. |
| `approx_kl` | How far the policy moved this update | Healthy: **0.005 - 0.02**. Above about 0.05 = updates too aggressive (try lower `learning_rate`). Near 0 = barely learning. |
| `clip_fraction` | % of updates the safety belt had to limit | Healthy: **0.05 - 0.3**. Above about 0.3 = too aggressive. About 0 = very timid. |
| `policy_gradient_loss` | Raw policy-update number | Tiny, jittery, around 0 (about -0.03 to +0.01). **Ignore it.** |
| `loss` | Combined loss | Noisy. **Ignore it.** |
| `learning_rate`, `n_updates` | Settings / counter | Just info |

---

## 6. Where each setting lives and what it does

### In `train.py` (the learning knobs)

| Setting | Current | Plain meaning | Try |
|---|---|---|---|
| `total_timesteps` | 500000 | How long to train | More if still improving at the end |
| `n_steps` | 2048 | Actions collected per cycle | 1024 = faster feedback, noisier. 4096 = steadier, slower. |
| `batch_size` | 64 | Chunk size when updating | Rarely matters; 64-256 |
| `gamma` | 0.99 | How far-sighted | 0.95 = short-sighted, 0.995 = very far-sighted |
| `ent_coef` | 0.03 | Randomness bonus | 0.0-0.01 once it works (so it settles). 0.03-0.05 if stuck. |
| `learning_rate` (default 3e-4) | not set | Size of each nudge | Lower (1e-4) if `approx_kl` is high or reward is jumpy. Higher (1e-3) if learning crawls. |
| `save_freq` | 10000 | Checkpoint every N steps | Saved in `logs/` so you can test early versions |

### In `maze_env.py` (the world)

| Setting | Current | What it changes |
|---|---|---|
| step penalty | -0.01 | Higher = hurries more; too high = agent gives up |
| novelty bonus | +0.5 | Higher = explores more; too high = it farms new tiles instead of going to the goal |
| wall penalty | -0.01 | Higher = avoids walls more |
| goal reward | +100 | The "point" of the task |
| `max_steps` | 500 | Time limit per episode |
| frame-skip loop | 8 | How far one action moves. Higher = coarser, faster; lower = finer, slower |
| forward/turn speed | 5.0 | How fast it moves/turns |
| `num_lidar_rays` / `lidar_range` | 32 / 10 | How much it can sense |
| start `[-3,-3]`, goal `[3,3]` | fixed | Same every time, so it may memorise the route |

---

## 7. Diagnose it yourself: symptom -> likely cause -> try

| What you see | Likely cause | What to try |
|---|---|---|
| `ep_len` stuck at 500, reward about -5 for a long time | Never reaches the goal; found "do nothing is safe" | Raise novelty bonus, raise `ent_coef`, lower wall penalty, check the goal is physically reachable in `max_steps` |
| Reward dropping, `ep_len` 500 | Hugging walls, penalised every step | Check wall penalty isn't dominating; more exploration reward |
| Reward rises then **collapses** | Updates too aggressive | Lower `learning_rate`; check `approx_kl` is under 0.05 |
| Reward rises then **plateaus** around 100 | Solved! It has learned a route | Stop, run `solve_maze.py` |
| Plateaus **below** 100 with `ep_len` fairly high | Learned a slow or inefficient route | Train longer; slightly raise step penalty |
| `entropy_loss` crashes to about -0.05 early | Locked in one habit too soon | Raise `ent_coef` |
| `entropy_loss` stays about -1.09 forever | Never settled on anything | Lower `ent_coef`, train longer, check rewards are informative |
| `explained_variance` negative for a long time | Critic confused (noisy or sparse reward) | Wait longer; make rewards less sparse |
| Very noisy reward but trending up | Normal RL | Judge the trend over about 20+ iterations, not single ones |
| Works in `solve_maze.py` but only from the fixed start | Memorised one route | Randomise start position in `reset()` |

---

## 8. How to judge progress like a pro

1. **Look at trends over many iterations, not one table.** Use TensorBoard: `tensorboard --logdir logs` -> http://localhost:6006 -> SCALARS. Turn the **smoothing** slider to about 0.6-0.9.
2. **Primary pair:** `ep_rew_mean` up and `ep_len_mean` down.
3. **Health check (second):** `entropy_loss` slowly rising toward 0, `approx_kl` roughly 0.005-0.02, `clip_fraction` under 0.3.
4. **Reality check:** run `solve_maze.py` on a checkpoint in `logs/` (e.g. `ppo_maze_100000_steps.zip`). Numbers can mislead; watching the agent doesn't.

Use the checkpoints: if reward looked great at 150k steps and then collapsed, you can still load the 150k one.

---

## 9. Experiments to play with (one change at a time!)

Always change **one** thing, delete old `logs/` so graphs don't mix, and compare `ep_rew_mean` curves.

1. **Entropy:** train with `ent_coef` 0.0, 0.01, 0.05. Watch how `entropy_loss` and speed to first goal differ.
2. **Gamma:** 0.9 vs 0.99. At 0.9 the agent may stop caring about the far goal.
3. **Novelty bonus:** 0 vs 0.5 vs 2.0. At 0 learning is probably very slow; at 2.0 it may farm tiles and ignore the goal.
4. **Learning rate:** 1e-4 vs 3e-4 vs 1e-3. See speed vs. stability.
5. **Randomise the start:** in `reset()`, pick a random open tile as the start. Compare how long it takes to learn, and whether it truly generalises.
6. **Frame-skip:** 4 vs 8 vs 16.

Remember: RL is random. A different run of the *same* settings can differ. If you're comparing two settings, run each at least twice before concluding.

---

## 10. Glossary (just what you need)

- **Policy:** the agent's decision rule (the actor network).
- **Value:** predicted future reward from a situation (the critic).
- **Episode:** one attempt, start to finish.
- **Step / timestep:** one action by the agent.
- **Iteration:** one collect-then-update cycle (2048 steps).
- **Advantage:** "was that action better than expected?"
- **Entropy:** how random the choices are (max 1.0986 with 3 actions).
- **Deterministic:** in `solve_maze.py` the agent always picks its most likely action, with no dice, so it's a clean replay of what it learned.

---

# Part 2: Everything added after the first version (evaluation, map memory, tests)

Sections 1-10 above describe the original project. This part records every change made since, so you can follow how the project grew and re-run all of it. Where this part and sections 1-10 disagree (for example the observation size, the episode budget, `PROMOTE_AT`), this part is current.

## 11. What changed, file by file

### `envs/maze_env.py` (the world)

| Change | What it does | Why |
|---|---|---|
| `EVAL_SEED_RANGE = 2**20` | Training layout seeds come from `[2**20, 2**31)`; evaluation seeds from `[0, 2**20)` | Test mazes are never seen in training, so a score measures generalisation |
| `reset(seed, options={"size", "layout_seed"})` | Pins one episode (maze size and layout) | Lets `evaluate.py` replay exactly the same mazes every time |
| `size=` constructor argument | `MazeEnv(size=7)` / `gym.make("MazeNav-v0", size=7)` | Tests and curriculum can start at any size |
| `LOOPS = 6`, `_make_layout(..., loops=None)` | Number of extra walls opened after the DFS maze | Loops make the maze less of a pure tree |
| `_time_budget(size)`, `STEPS_PER_TILE = 15` | Episode limit = 15 x number of open tiles, from the maze size only | The old budget came from the start-goal route, so `time_left` leaked how far the hidden goal was |
| `get_pose()`, `start_cell`, `goal_cell`, `layout_seed` | Public state used by the agents and tests | Oracle / Frontier / tests need the pose without touching PyBullet internals |
| `_bfs(start, passable, is_goal)` | One generic breadth-first search | `shortest_path()` (true layout) and `_route()` (own map) share it |
| Own map `known` | `size x size` array: -1 unknown, 0 open, 1 wall. Built from the 8 noisy lidar rays plus 4 world-axis scan rays and odometry | This is the memory. It is built only from what the sensors return, never from the true layout |
| `wall_tile` | Body id to tile, so a lidar hit on a wall says which tile is a wall | Exact wall marking, no guessing at corners |
| `searched` | Tiles the goal is ruled out from: visited tiles, and tiles on the clear stretch of the agent's own row/column while the goal was not visible | Without it the agent can map the whole maze, never have the goal in line of sight, and stop (this was the cause of the first 94% result) |
| `goal_seen_cell` | Remembers where the goal was last seen | Route cue can lead back to it after losing sight |
| `_route()` / `waypoint()` | BFS over the known map to the remembered goal, else to the nearest tile with an unknown neighbour or not yet searched | The route cue |
| `_crop()` | 7x7 window of `known` around the agent, rotated to the nearest 90 degree heading | Map picture the network can read |
| Out-of-maze guard | If physics knocks the agent outside the maze, mapping is skipped and the route is `None` | Found by a smoke training run (`IndexError` in a worker); has a regression test |

**Observation is now 70 numbers** (was 17):

| Index | Meaning |
|---|---|
| 0 | `time_left` (1 -> 0) |
| 1-4 | goal visible, distance, sin, cos of heading error (zeros unless in line of sight) |
| 5-8 | visit counts of the 4 neighbour tiles (ahead, left, behind, right) |
| 9-16 | 8 noisy lidar readings |
| 17-20 | route cue: has target, path length / size, sin, cos of heading error to the next tile |
| 21-69 | 7x7 map crop, +1 open, 0 unknown, -1 wall or outside |

Indices 0-16 did not move, so older helper code that slices `obs[9:17]` for lidar still works.

Reward is unchanged (novelty, revisit, wall, obstacle, step penalties, +100 goal).

### `train.py`
- `argparse`: `--seed`, `--timesteps`, `--resume`.
- `gym.make("MazeNav-v0")` instead of importing `MazeEnv` directly (registered in `envs/__init__.py`).
- `Curriculum` stores the level on the model (`model.curriculum_level`); SB3 pickles it into every checkpoint, so `--resume` continues at the same maze size.
- `PROMOTE_AT` raised **0.6 -> 0.9**: the maze only grows once the agent wins 90% of its last 100 episodes.
- `reset_num_timesteps=resume is None`, so a resumed run adds `--timesteps` on top of the saved count.

### `evaluate.py` (fixed evaluation set)
- 5 maze sizes x 200 seeded held-out mazes by default. `episode_seed = base + size*10000 + i`.
- Agents:

| Agent | What it knows | Role |
|---|---|---|
| `RandomAgent` | nothing | floor |
| `NodeCountAgent` | odometry + lidar, no map | classic map-free explorer (visits least-visited neighbour) |
| `FrontierAgent` | its own map (same sensors as PPO) | proves the memory is enough: follows the route cue |
| `OracleAgent` | true layout and goal (cheats) | upper reference |
| `PPOAgent` | the observation, 4-frame stack identical to `VecFrameStack` | the trained model |

- Reports success rate with Wilson 95% confidence interval, steps, SPL (path efficiency, 1.0 = shortest route), obstacle hits. Several `--models` also give a bootstrap mean row.

### `envs/__init__.py`
Registers `MazeNav-v0`. No `max_episode_steps`: the env truncates itself.

### `requirements.txt`
Versions pinned (`gymnasium==1.4.0`, `stable-baselines3[extra]==2.9.0`, `pybullet==3.2.7`, `torch==2.14.1+cpu`, ...). CPU torch comes from `--extra-index-url`. Python 3.12 (`.python-version`).

### `.gitignore`
Now also ignores `results/`, `.venv/`, `__pycache__/` (plus `logs/`, `ppo_maze_final.zip` as before).

### `tests/` and `conftest.py`
`conftest.py` puts the repo root on `sys.path`. 26 tests in total:

| File | Covers |
|---|---|
| `tests/test_env.py` | spaces (70-dim), `check_env`, same seed gives same episode, budget independent of route, budget scales with size, train/eval seeds held out, every maze solvable, fixed mode, curriculum level survives save/load, **own map never contradicts the layout**, **route cue exists until fully explored**, **agent knocked outside the maze does not crash** |
| `tests/test_eval.py` | Wilson CI values, unique held-out seeds, reproducible episodes, agents return well-formed rows, **FrontierAgent solves 40/40 (sizes 7 and 15)**, PPOAgent stacking matches `VecFrameStack` |

### `CLAUDE.md`
Updated for the 70-dim observation, `FrontierAgent`, and the warning that old checkpoints no longer load.

## 12. The memory idea in plain words

Before: the agent only knew how many times it had entered the 4 tiles around it (plus the last 4 frames). In a maze with loops that is not enough to be sure it has tried everywhere, so it wandered.

Now it draws its own map as it moves and gets two things from it:

1. **A picture:** the 7x7 crop around it (open / wall / unknown).
2. **A hint:** the direction of the next step toward the goal if it has seen it, otherwise toward the nearest place it has not explored or ruled out.

The hint is plain breadth-first search over the agent's own map, so the information in the observation is enough to solve every maze. The `FrontierAgent` shows that: it only follows the hint and solves essentially all of them. PPO still has to learn to use it, so 100% for PPO is a target, not a guarantee.

Honesty rules for the map: it uses noisy lidar readings, never the true layout; patrolling blocks are not walls; a tile is never overwritten once known.

## 13. All commands

Run everything from the repo root (`D:\Saved\RL\RL-Maze-Solver`). The scripts import `envs.maze_env`, and `app.py` starts `train.py` with a relative path.

### Activate the virtual environment
```
.\.venv\Scripts\Activate.ps1        # PowerShell
.venv\Scripts\activate.bat          # cmd
source .venv/Scripts/activate       # Git Bash
deactivate                          # leave it
```
Without activating, use `.\.venv\Scripts\python.exe` in place of `python`. If PowerShell refuses: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### Install
```
pip install -r requirements.txt
```
pybullet has no Windows wheel; installing it there needs the MSVC C++ Build Tools.

### `train.py`
| Flag | Meaning | Default |
|---|---|---|
| `--seed N` | seeds PPO, torch and the 8 envs | 0 |
| `--timesteps N` | steps to train; with `--resume` they are added on top | 5,000,000 |
| `--resume ZIP` | continue from a checkpoint (weights, optimiser state, curriculum level) | none |

```
python train.py
python train.py --seed 1 --timesteps 2000000
python train.py --resume logs/ppo_maze_100000_steps.zip --timesteps 500000
```
Outputs: checkpoint every 10k steps to `logs/ppo_maze_<N>_steps.zip`; TensorBoard data in `logs/PPO_run_N`; final model `ppo_maze_final.zip` in the repo root. Ctrl+C saves the current model. 8 parallel workers, no GUI.

Watch it: `tensorboard --logdir logs` then http://localhost:6006. Primary signals are `rollout/ep_rew_mean` up and `rollout/ep_len_mean` down; the console prints `Curriculum: maze size -> N` when it levels up.

### `evaluate.py`
| Flag | Meaning | Default |
|---|---|---|
| `--episodes N` | episodes per maze size | 200 |
| `--sizes 7 9 ...` | maze sizes | 7 9 11 13 15 |
| `--base-seed N` | offset for the eval seeds (must stay below `2**20`) | 0 |
| `--models a.zip b.zip` | PPO checkpoints to score | `ppo_maze_final.zip`, else newest `logs/*.zip` |
| `--deterministic` | PPO takes its most likely action instead of sampling | off (samples) |
| `--agents ...` | any of `random nodecount frontier oracle ppo` | all five |
| `--fixed` | the seed-0 test maze instead of seeded random mazes | off |
| `--out DIR` | where the result files go | `results` |

```
python evaluate.py
python evaluate.py --agents frontier oracle --episodes 200
python evaluate.py --models a.zip b.zip --sizes 11 15 --episodes 50 --deterministic
python evaluate.py --agents ppo --models logs/ppo_maze_500000_steps.zip
```

**Where the results go:** `results/eval_<timestamp>.csv` (one row per episode: agent, size, seed, success, steps, spl, shortest, path_len, obstacle_hits) and `results/eval_<timestamp>.json` (the config, library versions, and the summary table with confidence intervals). `results/` is git-ignored. Runs made while developing this feature used `--out` pointing at a temporary folder, so only `results/eval_20261017_145657.*` (the old model, 17-dim era) is in the repo; do not compare new runs against it.

### Other scripts
```
python app.py           # Flask dashboard http://localhost:5000, starts TensorBoard on :6006; Start/Stop buttons run train.py
python test_agent.py    # GUI rollout of the newest logs/*.zip (stale: no frame stack, wrong obs print)
python solve_maze.py    # GUI demo of the PPO model on random mazes
```

### Tests
```
pytest tests/
pytest tests/test_env.py::test_spaces
pytest tests/test_eval.py -q
```
About 35 seconds for all 26.

## 14. Results so far

All numbers are on held-out mazes (seeds below `2**20`), success rate.

**Before the map memory** (`results/eval_20261017_145657.json`, 200 episodes per size, 17-dim observation):

| Agent | 7 | 9 | 11 | 13 | 15 | all |
|---|---|---|---|---|---|---|
| Random | 23.5% | 25.5% | 17.5% | 12.0% | 7.5% | 17.2% |
| NodeCount (no map) | 98.5% | 97.0% | 95.5% | 91.0% | 88.5% | 94.1% |
| Oracle (cheats) | 100% | 100% | 100% | 100% | 100% | 100% |
| PPO (`ppo_maze_parallel`, old env) | 93.5% | 85.0% | 71.0% | 62.5% | 46.5% | 71.7% |

**After the map memory** (200 episodes per size, new env), `FrontierAgent`:

| Size | 7 | 9 | 11 | 13 | 15 | all |
|---|---|---|---|---|---|---|
| Success | 100% | 99.5% | 100% | 100% | 100% | **99.9%** (999/1000, CI 99.4-100%) |
| Mean steps when it wins | 74 | 117 | 206 | 321 | 494 | 242 |
| SPL | 0.77 | 0.72 | 0.63 | 0.58 | 0.52 | 0.64 |

How it got there: the first version of the route cue (visit unknown neighbours only) scored 94% because it finished mapping without ever getting the goal in line of sight and then had nothing left to do. Adding the `searched` set (rule out tiles the goal would have been visible from) took it to 100% on a 100-episode check and 99.9% on the full set. The one miss is a size-9 maze (a seed in `90000-90199`); not yet diagnosed, likely a patrolling block jamming a corridor.

**PPO on the new observation: not trained yet.** Old checkpoints (`ppo_maze_old_final.zip`, `ppo_maze_parallel.zip`, anything in `logs/`) were trained on 17 numbers and cannot load into the 70-number env. After training, run `python evaluate.py` and compare PPO to the Frontier row: the closer PPO gets to Frontier, the better it learned to use its memory.

## 15. Gotchas (new ones)

- **One env per process.** `MazeEnv` talks to PyBullet's default client. Tests and evaluation create and close envs one at a time; training uses subprocesses.
- **Retrain after any observation change.** The network's input size is baked into the checkpoint.
- **A short smoke run leaves log folders.** Each `python train.py` creates `logs/PPO_run_N`, even if it crashes; delete empty ones.
- **No learned policy is guaranteed 100%.** The Frontier row is the proof that the information is sufficient; PPO is measured against it.
- **Extra scan rays.** Besides the 8 lidar rays in the observation, the map builder casts 4 world-axis rays (with the same noise) so tile neighbours get revealed regardless of heading.
- **Slightly slower steps.** Map update plus BFS each step adds a little CPU time per environment step.

- **Resizing waits for the next episode.** The curriculum calls `set_size` while episodes are running. It now only sets `env.next_size`; `reset()` applies it (`env.size`, `env.offset`, map arrays). Doing it immediately crashed all workers with `IndexError` at the first level-up (found at ~80k steps); regression test `test_resizing_mid_episode_waits_for_the_next_reset`.

## 16. Experiments to try next (one change at a time)

1. Train PPO on the new env and compare to Frontier per size.
2. Try `PROMOTE_AT` 0.8 vs 0.9: faster levels vs. a more solid base at each size.
3. Remove the route cue (keep only the crop) and see how much of the gain came from the hint versus the picture.
4. Remove the crop (keep only the cue) for the opposite comparison.
5. Debug the single Frontier failure (size 9, seeds `90000-90199`) with a patrolling-block-aware route.
6. Evaluate several training seeds with `--models a.zip b.zip ...` to get the bootstrap mean row.
