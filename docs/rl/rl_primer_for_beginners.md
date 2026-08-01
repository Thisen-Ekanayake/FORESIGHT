# RL Primer: What's Happening in FORESIGHT's PointGoal Navigator

This document teaches reinforcement learning (RL) from first principles using the actual RL code in this
repository as the running example. It assumes no prior RL background. For the terse reference version (what
exists, how to run it) see `docs/rl/pointnav.md` — this document is the "why does any of this work" companion.

**What this system is, in one paragraph:** we're training a virtual robot (an "agent") to walk from a random
starting point to a goal coordinate inside a scanned 3D house (a Habitat-Sim scene), using only an RGB camera
image and a compass-like vector pointing at the goal. It has no map. It learns by trial and error: it tries
moving, gets a numeric score after every action, and gradually adjusts its behavior to get higher scores. The
algorithm that does the adjusting is called **PPO** (Proximal Policy Optimization).

---

## 1. Reinforcement learning, from scratch

### 1.1 The agent-environment loop

Every RL problem is a loop between an **agent** (the thing making decisions — here, a neural network) and an
**environment** (the world it acts in — here, Habitat-Sim simulating a house):

```
        ┌─────────────┐   observation, reward     ┌─────────────┐
        │ Environment │ ───────────────────────▶ │    Agent    │
        │ (Habitat-   │                           │  (policy    │
        │  Sim house) │ ◀─────────────────────── │  network)   │
        └─────────────┘         action            └─────────────┘
```

At every discrete time step `t`:
1. The environment hands the agent an **observation** — what it can currently perceive.
2. The agent's **policy** (a function, here a neural network) looks at the observation and picks an **action**.
3. The environment applies that action, moves time forward one step, and returns a **reward** (a single number
   saying how good that step was) plus the next observation.
4. Repeat, until the **episode** ends (goal reached, or a step limit hit).

This project's instantiation of each term:

| RL term | Concrete meaning here |
|---|---|
| Environment | A Habitat-Sim scene (one of the 10 HM3D minival houses) + the PointNav task logic |
| Episode | One attempt to walk from a sampled start pose to a sampled goal coordinate, max 500 steps |
| Observation | A 256×256 RGB image + a 2D vector (distance, angle) to the goal, updated every step |
| Action | A pair of numbers: forward velocity and turning velocity, applied for 1 simulated second |
| Reward | A number combining "did you get closer to the goal", "did you hit a wall", "are you moving smoothly", and "did you just win" |
| Agent / policy | A ResNet18 image encoder + GRU memory + a small head that outputs the action, all trained together |

### 1.2 Return, discounting, and why the agent cares about the future

A single reward isn't the goal — the agent should maximize the **return**: the sum of all rewards from now
until the episode ends. But rewards further in the future are discounted by a factor `γ` (gamma, `< 1`) per
step, because a reward 100 steps away is less certain / less valuable than one right now:

```
Return_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...
```

In this project's config (`experiments/configs/rl/pointnav_continuous.yaml:87`), `gamma: 0.99` — rewards ~100
steps out still matter noticeably, which fits a 500-step episode where the goal might be far away.

### 1.3 Policy and value function — the two things being learned

- **Policy** `π(action | observation)`: given what the agent sees, what action should it take? This is what
  you actually deploy at test time (`tools/record_pointnav_demo.py` just runs the policy repeatedly).
- **Value function** `V(observation)`: given what the agent currently sees, how much total future reward do
  we *expect* from here, if we keep following the current policy? This is a training aid only — it's used to
  judge whether an action turned out better or worse than expected (see "advantage" below). It's discarded at
  deployment time.

An architecture that outputs *both* from a shared body (as this project's `PointNavResNetPolicy` does) is
called an **actor-critic**: the "actor" picks actions, the "critic" grades them.

### 1.4 On-policy learning (why PPO needs fresh data)

PPO is **on-policy**: it can only learn from data collected by the *current* version of the policy — you
can't reuse old experience once the policy has changed much (unlike, say, DQN, which is off-policy and reuses
a big replay buffer). This is why training constantly alternates between two phases:

```
[collect a batch of experience by acting in the environment] → [update the network a little] → repeat
```

That alternation is exactly what you'll see in the training loop below.

---

## 2. This project's exact reward function

Understanding the reward is the single most important thing for reasoning about RL behavior — it is the
*only* signal the agent ever gets about what "good" means. Habitat assembles it in two layers:

**Layer 1 — built into habitat-lab's `RLTaskEnv.get_reward`** (not this project's code —
`habitat/core/environments.py` in the installed package), added every step automatically:

```
reward = slack_reward                     # -0.01, every step (a "hurry up" penalty)
       + reward_measure                   # this project's custom measure, see Layer 2
       + (success_reward if episode succeeded else 0)   # +2.5, once, on success
```

`slack_reward` is a small constant penalty per step — it exists purely so that standing still forever is
worse than doing anything productive. `success_reward` is a one-time bonus for reaching the goal (defined as
being within `success_distance = 0.2 m` of it, per habitat-lab's `Success` measure default, used here as
`success_measure`).

**Layer 2 — this project's `PointNavSmoothReward`** (`src/foresight/rl/smooth_reward.py`), which supplies the
`reward_measure` term above:

```
reward_measure = geodesic_progress
                − collision_penalty · (1 if collided this step else 0)
                − angular_penalty  · |angular_velocity|
                − jerk_penalty     · (|Δlinear_velocity| + |Δangular_velocity|)
```

with `collision_penalty = angular_penalty = jerk_penalty = 0.01 / 0.004 / 0.004`
(`experiments/configs/rl/pointnav_continuous.yaml:38-40` — the only place these numbers exist; the dataclass
in `smooth_reward.py` declares them `MISSING` so they can't silently fall back to a code default). Each term
exists for a specific behavioral reason:

- **`geodesic_progress`** (`= -(new_distance_to_goal - old_distance_to_goal)`): the main "get closer to the
  goal" signal, measured along the *walkable* shortest path, not straight-line distance (so it correctly
  penalizes walking toward a wall between the agent and the goal).
- **`collision_penalty`**: discourages bumping into walls/obstacles.
- **`angular_penalty`**: discourages spinning in place to farm... nothing, actually, but it stops jittery
  turning.
- **`jerk_penalty`**: penalizes the *change* in velocity between consecutive steps (not the velocity itself),
  which is what actually produces smooth, non-jittery motion — the stated design goal of this whole RL
  addition (see `docs/rl/pointnav.md`).

Why this design matters pedagogically: the agent never sees "task success" directly — it only sees numbers.
If the numbers don't line up with what you actually want, the agent will optimize the numbers, not your
intent. Section 5 below is a real example of exactly that going wrong.

---

## 3. The action and observation spaces

**Action space** — `velocity_control`, a *continuous* 2D action (not a menu of 4 discrete moves like
`move_forward`/`turn_left`/etc., which most Habitat tutorials use):

| | Range | Meaning |
|---|---|---|
| `linear_velocity` | 0.0 – 0.25 m/s | forward speed |
| `angular_velocity` | −10 – 10 deg/s | turning rate |

Both are integrated for `time_step = 1.0 s` per environment step (`pointnav_continuous.yaml:28-34`). There is
no explicit "stop" action — stopping is *implicit*: if both velocities drop below small thresholds
(`min_abs_lin_speed`, `min_abs_ang_speed`), Habitat treats it as calling `stop`. This detail turns out to be
important (see Section 5).

Because the action is continuous (two real numbers, not a category), the policy's output layer is a
**Gaussian distribution** — it predicts a mean and standard deviation for each of the two numbers, and the
actual action is sampled from that distribution during training (`action_distribution_type: gaussian`,
`pointnav_continuous.yaml:74`). At evaluation time (`record_pointnav_demo.py --ckpt ...`, without
`--stochastic`), it instead just takes the mean — the "most likely" action — for reproducible, non-jittery
behavior.

**Observation space** — RGB only, by design (matching the project's RGB-first thesis):

| Sensor | Shape | What it gives the agent |
|---|---|---|
| `rgb_sensor` | 256×256×3 | the camera image |
| `pointgoal_with_gps_compass` | 2 numbers | (distance to goal, angle to goal) in the agent's current egocentric frame |

Note there's no depth sensor and no map — the agent has to learn to relate "what I see" and "where the goal
is relative to me" to "what motion gets me there", entirely from experience. (`--modality depth|rgbd` exists
in `train_pointnav.py` as an optional stronger baseline, but the default and the one actually trained is RGB.)

---

## 4. PPO: the training algorithm

PPO (Schulman et al., 2017) is the current default RL algorithm for continuous-control problems like this one
because it's stable and doesn't require much hyperparameter tuning compared to older policy-gradient methods.
Here's what it does, using this project's exact config values
(`experiments/configs/rl/pointnav_continuous.yaml:75-90`) as concrete numbers.

### 4.1 Step 1 — collect a rollout

Run the *current* policy in the environment for a fixed number of steps and record everything:

- `num_environments: 4` — 4 copies of the environment run in parallel (in different scenes/episodes), sized
  to fit an 8GB laptop GPU.
- `num_steps: 128` — each of those 4 environments is stepped 128 times before we stop and learn.
- So one rollout = `4 × 128 = 512` (observation, action, reward, value-estimate) tuples.

### 4.2 Step 2 — estimate the advantage (GAE)

For each recorded step, we need to know: *was this action better or worse than the critic expected?* That
quantity is the **advantage**. A raw comparison (actual return so far vs. `V(observation)` predicted) is
noisy, so PPO uses **Generalized Advantage Estimation (GAE)**, which smooths this estimate using a second
decay factor `tau` (a.k.a. `λ`, lambda):

- `use_gae: true`, `tau: 0.95` — blends short-horizon (biased but low-variance) and long-horizon (unbiased
  but high-variance) advantage estimates; 0.95 is the standard "mostly long-horizon, slightly smoothed" choice.

Intuition: a positive advantage means "do more of this"; negative means "do less of this" — this is the
actual training signal, not the raw reward.

### 4.3 Step 3 — update the policy (the "clipped" part of PPO)

Naively, you'd increase the probability of high-advantage actions and decrease it for low-advantage ones. The
danger is that a single update can overcorrect and wreck the policy (it's being applied to data collected by
an *older* version of itself, since we're reusing the 512-step batch for multiple gradient steps). PPO's core
trick is to **clip** how much the action-probability-ratio is allowed to change in one update:

- `clip_param: 0.2` — the policy's output probability for any given action isn't allowed to move by more than
  ±20% relative to what it was when the data was collected, per update. This is *the* defining feature of PPO
  ("Proximal" = stay close to the old policy).
- `ppo_epoch: 2` — the 512-step batch is reused for 2 full passes.
- `num_mini_batch: 2` — each pass is split into 2 mini-batches (256 steps each) for the gradient updates.
- `value_loss_coef: 0.5` — the critic (value function) is trained simultaneously, weighted at half the policy
  loss's importance.
- `entropy_coef: 0.01` — a small bonus for keeping the action distribution *not* overly narrow/certain, which
  keeps some randomness alive so the agent keeps exploring instead of collapsing onto one behavior too early.
- `lr: 2.5e-4`, `max_grad_norm: 0.2` — standard Adam learning rate + gradient clipping for stability.

### 4.4 Repeat

`total_num_steps: 1.0e7` — steps 1–3 repeat until 10 million total environment steps have been collected
(across all 4 envs combined). Checkpoints are saved periodically (`num_checkpoints: 25`, so roughly every
400k steps) — training is resumable; re-running `tools/train_pointnav.py` picks up the latest checkpoint
automatically.

---

## 5. The policy network architecture

`PointNavResNetPolicy` (from habitat-baselines, not custom code in this repo) — an actor-critic with:

```
RGB image (256×256×3)
        │
   ResNet18 encoder            (backbone: resnet18, train_encoder: true — trained from scratch, not frozen)
        │
   [visual features] ── concat ── [pointgoal vector]
        │
      GRU                        (rnn_type: GRU, num_recurrent_layers: 1 — gives the agent short-term memory
        │                         across steps within an episode, since a single frame can't tell you
        │                         "have I already tried turning left here")
        │
   ┌────┴────┐
   ▼         ▼
 actor      critic
 head       head
   │         │
Gaussian   scalar
(lin, ang)  V(obs)
```

The GRU (a recurrent layer) matters specifically for navigation: without memory, the agent can't distinguish
"I'm passing this doorway for the first time" from "I already tried this doorway and it was a dead end" —
both look like the same single RGB frame. `hidden_size: 512` is the size of that memory vector.

---

## 6. Code tour — reading order

The custom code in this repo is deliberately small; nearly everything is habitat-lab/habitat-baselines
configuration. Read in this order to build understanding bottom-up:

1. **`src/foresight/rl/velocity_action.py`** — `SmoothVelocityAction`. Not an RL concept at all, just a
   NumPy-compatibility fix: habitat-baselines' gym wrapper hands velocities as shape-`(1,)` arrays, and the
   stock habitat-lab `VelocityAction.step` crashes building a NumPy array from mismatched shapes under
   NumPy ≥1.24. This subclass squeezes to plain floats first, then defers to the real (stock) implementation.
   Registered as `SmoothVelocityAction` so the config can select it.

2. **`src/foresight/rl/smooth_reward.py`** — `PointNavSmoothReward`, the custom reward measure described in
   Section 2. Read `update_metric()` — it's called every step and directly computes the formula above.
   Registers itself with both Habitat's measure registry and Hydra's `ConfigStore` (so YAML can reference
   `pointnav_smooth_reward` by name).

3. **`src/foresight/rl/__init__.py`** — one line, `import foresight.rl` triggers the two registrations above.
   Every entry-point script imports this package first, purely for that side effect.

4. **`experiments/configs/rl/pointnav_continuous.yaml`** — ties everything together: composes habitat-lab's
   `pointnav_hm3d` task/dataset defaults with this project's action + reward, plus habitat-baselines' PPO
   trainer config. This is the file to edit when tuning hyperparameters (Section 4's numbers all live here).

5. **`tools/generate_pointnav_episodes.py`** — a one-time data-prep script (not part of the training loop
   itself): samples (start, goal) pairs on the 10 local HM3D scenes and writes train/val `.json.gz` datasets,
   since the full HM3D training split isn't downloaded locally. Run once before training.

6. **`tools/train_pointnav.py`** — the entry point that actually runs PPO (delegates to
   `habitat_baselines.run.execute_exp`, which implements the loop in Section 4). `prepare_config()` is worth
   reading: it strips the discrete actions and picks the visual modality *in Python*, not via Hydra overrides,
   because the base config re-adds them internally in a way Hydra's deletion syntax couldn't reliably undo.

7. **`tools/record_pointnav_demo.py`** — runs a trained checkpoint (no more learning, no more exploration
   noise by default) on one held-out episode and renders a video. This is what "deploying the policy" looks
   like in this project.

---

## 7. Running it and reading the metrics

The four steps below run in order, so there is a one-command wrapper for them —
`scripts/run_rl_pipeline.sh` (`--list`, `--dry-run`, `--stages eval,demo` for a subset). To run them
individually:

```bash
PY=/home/thisen-ekanayake/miniforge3/envs/habitat/bin/python

# 1. One-time: generate train/val episodes
DISPLAY=:1 $PY tools/generate_pointnav_episodes.py \
    --set splits.train.num_episodes_per_scene=300 splits.val.num_episodes_per_scene=30

# 2. Train (resumable)
DISPLAY=:1 $PY tools/train_pointnav.py
tensorboard --logdir results/runs/pointnav/tb

# 3. Evaluate a checkpoint on held-out episodes
DISPLAY=:1 $PY tools/train_pointnav.py --eval

# 4. Record a demo video
DISPLAY=:1 $PY tools/record_pointnav_demo.py \
    --ckpt results/runs/pointnav/checkpoints/latest.pth \
    --set episode_index=0 render.duration=15 render.fps=50
```

Every setting these scripts use comes from a YAML config, never a Python default: `--set dotted.key=value`
edits the script config (`experiments/configs/rl/{train_pointnav,generate_episodes,record_demo}.yaml`), while
`--habitat-set` passes Hydra overrides to the task/PPO config (`pointnav_continuous.yaml`). See
`docs/rl/pointnav.md#configuration`.

What to watch in TensorBoard / eval output, and what each metric actually tells you:

| Metric | What it measures | What to conclude if it's bad |
|---|---|---|
| `reward` | The full reward sum per episode (Section 2) | Going up over training = learning *something*, but doesn't tell you what |
| `success` | Fraction of episodes ending within `success_distance` of the goal | The real task metric — this is what you actually care about |
| `spl` (Success weighted by Path Length) | Success, penalized for taking a longer-than-optimal path | High success + low SPL = reaching the goal but wandering |
| `collisions` | Count of steps where the agent contacted geometry | Rising = policy is being reckless; this project's reward explicitly penalizes it |

---

## 8. Case study: a real reward-shaping bug from this project

This actually happened during training (see `logs/2026-07-24.md`) and is a good worked example of why
"reward went up" ≠ "the agent learned the task."

**Setup:** full 10M-step training run completed. Collisions: 0/episode (great). Reward: trending up but
plateauing around −9. **Success: only 5%.**

**Symptom:** watching a rendered demo, the agent's path closely follows the geodesic-optimal route toward the
goal... and then drifts, ending up ~6–8 m short, having wandered for the rest of the 500-step episode.

**Diagnosis (the actual RL reasoning, not guesswork):**
1. Stop is *implicit* (Section 3) — it only happens when both velocities are near zero. There's no dedicated
   "I'm done" action with its own learnable probability.
2. The success radius is tight (`success_distance = 0.2 m`) relative to how imprecisely a velocity-controlled
   agent can be expected to stop, especially early in training when actions are close to random.
3. Consequence: during exploration, the agent almost *never* stumbles into the success condition by chance.
   Since PPO only learns from advantages computed over experience it actually collected, and it essentially
   never collected a `+2.5` success reward, there is no gradient teaching it "stopping here near the goal is
   good" — only the dense `geodesic_progress` term, which rewards *approaching* but is silent about *stopping*.
4. Result: the agent reliably learns "go toward the goal" (that reward is dense and constant) but never
   learns "stop precisely when you arrive" (that reward is sparse to the point of being statistically
   invisible during early random exploration).

This is a textbook **sparse-reward exploration problem**: a correct-looking reward function can still fail to
teach the intended behavior if the rewarding event is too rare for the agent to ever stumble into during
random exploration, especially early in training when the policy is closest to random.

**Recommended fixes** (documented, not yet applied at time of writing — check `PROGRESS.md` milestone #9 and
`experiments/configs/rl/pointnav_continuous.yaml` for current state):
1. Enlarge `success_distance` (e.g. 0.2 → 0.5 m) so the success bonus is discoverable by chance more often.
2. Raise `success_reward` and/or lower `slack_reward` so the goal signal dominates once it is discovered.
3. Replace the `angular_penalty` term with jerk-only smoothing — penalizing *any* turning discourages the
   necessary final-approach turns toward the goal, which may itself be fighting against reaching it.
4. More parallel environments and/or longer training, to increase the raw number of chance successes seen.

The general lesson worth taking from this: **when an RL agent's behavior looks "almost right but stuck", the
first thing to check is whether the sub-goal it's failing at ever produces a reward signal an untrained
(basically random) policy could realistically stumble into.** If the answer is no, more training won't fix it
— the reward function needs to change.

---

## 9. Glossary

| Term | Meaning |
|---|---|
| Agent | The decision-maker; here, the policy network |
| Environment | The simulated world the agent acts in; here, Habitat-Sim + PointNav task |
| Episode | One full attempt from reset to termination |
| Observation | What the agent perceives at a given step (RGB + pointgoal vector) |
| Action | What the agent outputs at a given step (linear + angular velocity) |
| Reward | The scalar feedback signal after each step |
| Return | Sum of (discounted) future rewards from a given step onward |
| Discount factor (γ) | How much future reward is worth relative to immediate reward (0.99 here) |
| Policy (π) | The function mapping observations to actions; what gets deployed |
| Value function (V) | Predicts expected return from a given observation; training aid only |
| Advantage | How much better/worse an action was than the value function expected |
| GAE | Generalized Advantage Estimation — a smoothed way to compute advantage |
| On-policy | Can only learn from data collected by the (near-)current policy |
| Actor-critic | An architecture producing both a policy (actor) and value estimate (critic) |
| PPO | Proximal Policy Optimization — the specific on-policy algorithm used here |
| Clipping (PPO) | Limiting how much the policy is allowed to change in one update, for stability |
| Rollout | A batch of experience collected by running the current policy |
| Entropy bonus | Reward for keeping the policy's action distribution from collapsing too early |
| SPL | Success weighted by Path Length — success metric that penalizes inefficient paths |
| Sparse reward | A reward signal that only fires rarely, making it hard to discover by exploration |
| Reward shaping | Adding auxiliary reward terms (like `geodesic_progress`) to make sparse goals learnable |

---

## 10. Where this fits in the project, and what's next

This RL navigator is a **trained comparison baseline**, not the project's core deliverable — FORESIGHT's main
thesis is a *zero-shot* RGB pipeline (monocular depth + VLM, no training). This PPO agent exists to give that
zero-shot pipeline something concrete to be compared against. See `PROGRESS.md` milestone #9 for current
status, and `docs/rl/pointnav.md` for the terse reference (file list, run commands, design table) without the
teaching material in this document.
