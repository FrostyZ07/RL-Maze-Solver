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
