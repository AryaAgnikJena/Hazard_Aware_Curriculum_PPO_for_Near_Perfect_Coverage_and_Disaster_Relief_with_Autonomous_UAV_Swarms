# Multi-UAV Coverage & Disaster Relief with PPO

Five cooperating drones learn to map a 100 × 100 m disaster area, avoid buildings, birds, wind and each other, manage fuel and fly home, using a single PPO policy trained with a four-phase curriculum and lethal-collision "lives" schedule.


# Table of contents

1. [Overview](#overview)
2. [Key features](#key-features)
3. [Repository structure](#repository-structure)
4. [Installation](#installation)
5. [The environment](#the-environment)
6. [Reward design](#reward-design)
7. [Curriculum and lethal collisions](#curriculum-and-lethal-collisions)
8. [Evaluation scenarios](#evaluation-scenarios)
9. [Usage](#usage)
10. [Outputs](#outputs)
11. [Configuration reference](#configuration-reference)
12. [Reproducibility notes](#reproducibility-notes)
13. [Roadmap / known limitations](#roadmap--known-limitations)
14. [License](#license)

---

# Overview

This project is a custom [Gymnasium](https://gymnasium.farama.org/) environment plus a full training and evaluation pipeline for **multi-UAV area coverage** in a simulated disaster zone. A team of 5 quadrotor-like drones must:

- **Cover** a 20 × 20 grid (400 cells) of a 100 × 100 m map as quickly as possible. Each drone scans a circle of radius `r(h) = h · tan(15°)`, so altitude matters: the optimal scan height is 20 m (r ≈ 5.4 m).
- **Survive** an environment of up to 25 buildings (three size classes, 4–30 m tall), 10 moving birds, and a gusting wind field.
- **Avoid** drone–drone, drone–bird and drone–building collisions. In the final phase these are *lethal*: a drone that runs out of lives is removed from the episode.
- **Manage fuel** and return to the spawn zone, earning a landing bonus.
- **Rescue victims** (Phase 4) by flying within range of them.

A single centralised actor-critic (PPO, Stable-Baselines3) controls all drones: 15 continuous actions (3 per drone) from one shared observation vector.

# Key features

- **Custom physics**: mass, thrust limits, quadratic drag on *relative* airspeed, a slowly-drifting mean wind with gusts, drone–drone repulsion, soft walls.
- **Hover-centred vertical action**: `a_z = 0` means "hold altitude", so exploration noise doesn't push drones into the floor.
- **Height-dependent scanning** with a piecewise height profile `y(h)` used both as a coverage multiplier and a bounded altitude penalty.
- **Event-based collision accounting**: one-time onset penalty plus a small per-step penalty, per kind (drone / bird / building), plus smooth soft barriers *before* contact so there's always a gradient to escape along.
- **Four-phase curriculum** (empty map → buildings → buildings + birds → lethal collisions with victims and building observations).
- **"Lives" curriculum** in Phase 4: lives per drone are tightened `6 → 4 → 3 → 2 → 1` as the policy proves itself.
- **Reproducible scenario suite**: 10 scenario types × 5 seeds = 50 fixed test episodes, plus a disjoint validation suite used for checkpoint selection.
- **Burst-based threshold trainer** with resume, smoothed best-model selection, milestone snapshots, reward-version archiving and warm-starting (zero-padded weights when the observation grows).
- **Rich reporting**: research-style evaluation figures, per-scenario breakdowns, TensorBoard reward-component logging, and compact `.npz` **replays** that can be re-rendered to GIF/MP4 (top-down or 3-D) without the model or environment.

# Repository structure

```
.
├── configs/
│   └── config.py                  # every tunable: physics, rewards, curriculum, fuel, scenarios
├── env/
│   ├── multi_uav_env.py           # MultiUAVEnv + Phase1Env … Phase4Env
│   └── scenarios.py               # 10 scenario types × 5 seeds (test / validation suites)
├── training/
│   ├── train.py                   # phase-by-phase training entry point
│   ├── continue_training.py       # continue / extend an existing run
│   ├── train_until_threshold.py   # burst trainer with curriculum, resume, best-model logic
│   └── callbacks.py               # CheckpointCallback, CSVLoggerCallback, RewardBreakdownCallback
├── evaluation/
│   ├── evaluate.py                # final / per-phase evaluation + figures
│   ├── metrics.py                 # lightweight evaluation helper (CSV + basic plots)
│   ├── replay.py                  # replay recorder + GIF/MP4 renderer (CLI)
│   └── network_viz.py             # actor-critic architecture diagram *
├── visualization/
│   └── visualize.py               # live 3-D rollout visualiser (+ saves replays and figures)
├── models/                        # saved models + VecNormalize stats (created at runtime)
└── logs/ results/                 # TensorBoard, CSVs, figures, replays (created at runtime)
```

\* `visualize.py` imports `network_viz` from `visualization/` and falls back to `evaluation/`, so either location works.

> Make sure each folder contains an `__init__.py` so that `python -m training.…`, `python -m evaluation.…` etc. work.

# Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install numpy pandas matplotlib pillow torch gymnasium stable-baselines3 tensorboard tqdm rich
```

- `tqdm` and `rich` are needed for SB3's training progress bar.
- **ffmpeg** is optional; it is only needed for MP4 output. Without it the renderer falls back to GIF.
- `visualize.py` uses the interactive **TkAgg** backend. For headless machines change `matplotlib.use("TkAgg")` to `"Agg"`.
- A GPU is not required; the policy is a small MLP and training uses a single-process `DummyVecEnv`.

# The environment

`MultiUAVEnv` (in `env/multi_uav_env.py`) is shared by all phases. Phases differ only in what they spawn:

| Phase | Class | Buildings | Birds | Victims | Lethal collisions | Building obs |
|-------|-------|:---------:|:-----:|:-------:|:-----------------:|:------------:|
| 1 | `Phase1Env` | 0 | 0 | 0 | – | – |
| 2 | `Phase2Env` | 10 | 0 | 0 | – | – |
| 3 | `Phase3Env` | 25 | 10 | 0 | – | – |
| 4 | `Phase4Env` | 25 | 10 | 10 | ✔ | ✔ |

**World**: 100 × 100 m map, altitude 2–40 m, 1000 steps per episode at `DT = 0.1 s`. The spawn zone is a 36 × 36 m square in a random corner; buildings never overlap it.

**Buildings** come in three classes: small (4–8 m tall), medium (9–18 m), large (19–30 m).

**Observation** (per drone, 20 values): position, velocity, scan radius, nearest hazard distance + direction, nearest teammate distance + direction, direction to its Voronoi-assigned uncovered cell (or home when low on fuel), four wall distances, fuel fraction. Global values: coverage fraction, step fraction, and optionally the mean wind (2). In Phase 4, four extra values per drone are appended: nearest-building distance, unit direction, and roof clearance (so the policy can tell "fly over" from "fly around").

| Phase | Observation size |
|-------|-----------------|
| 1–3 | 5 × 20 + 2 + 2 = **104** |
| 4 | 104 + 5 × 4 = **124** |

**Action**: 15 continuous values in `[-1, 1]` (ax, ay, az per drone). Horizontal thrust is `a · 20 N`; vertical thrust is `m·g + a_z · (20 − m·g)`.

**Fuel & return to base**: each drone has about 886 steps of fuel. Below 15 % fuel it is guided toward the spawn zone and may land for a bonus. Landed and dead drones are frozen.

**Termination**: all drones landed/dead, all fuel empty, or the 1000-step limit.

# Reward design

The per-step reward is a sum of named components (all logged in `info["r_comp"]` and to TensorBoard):

| Component | Purpose |
|-----------|---------|
| `coverage` | reward for each newly covered cell, scaled by the height profile; extra weight near buildings while learning to navigate |
| `milestone` | one-time bonuses at 30/50/70/80/95 % coverage |
| `checkpoint`, `rescue` | checkpoint grid hits, victims reached |
| `progress` | shaping toward each drone's Voronoi target cell |
| `height` | bounded penalty `−6·(1 − y(h))`, zero at 20 m |
| `wall`, `oob`, `roof`, `ceiling`, `sustained_height`, `lazy` | small penalties against wall hugging, roof skimming, flying too high, or hovering idle |
| `drone_hit`, `bird_hit`, `building_hit` | onset penalty + per-step penalty + soft proximity barrier |
| `fuel`, `rtb` | empty-tank penalty, landing bonus |
| `mission` | `5000 / step` bonus the first time coverage reaches 95 % |
| `death` | −400 per dead drone (never clipped) |

The total per-step reward (excluding deaths) is clipped to ±300. **Bump `REWARD_VERSION` in `config.py` whenever you change the reward**: the trainer then archives the old run (policy and `VecNormalize` stats are meaningless under a different reward) unless the old version is listed in `REWARD_COMPATIBLE_WITH`, in which case the policy is kept and only the reward-normalisation statistics are reset.

# Curriculum and lethal collisions

Phases 1–3 train the basics: altitude discipline, coverage, then navigation around buildings and birds. **Phase 4** adds lethal contacts. Switching a policy that makes ~12 contacts per episode directly to "one contact = death" collapses it, so the trainer walks a lives schedule:

```
LIVES_SCHEDULE = [6, 4, 3, 2, 1]
```

A stage is promoted when, over the last `SMOOTH_N` bursts, smoothed coverage ≥ 75 % and deaths/episode ≤ 0.5 (or after `MAX_STAGE_BURSTS` if coverage is acceptable). Training finishes at 1 life when smoothed coverage ≥ 85 % and deaths/episode ≤ 0.3. "Best" is chosen by `coverage − 5 × deaths` over a rolling window, not a single noisy burst.

# Evaluation scenarios

`env/scenarios.py` defines **10 scenario types**, each with **5 fixed seeds** (50 episodes total):

| Type | What it stresses |
|------|------------------|
| `baseline` | the training distribution |
| `large_only` | only 19–30 m towers: must route around |
| `small_dense` | 40 low blocks: lots of obstacles, flying over possible |
| `crowded` | 45 mixed buildings, little free space |
| `tight_corners` | clustered buildings: L-shapes, dead ends |
| `street_canyon` | lattice of medium/large blocks with narrow streets |
| `perimeter` | buildings along the borders: corner traps + wall penalties |
| `bird_swarm` | 3× birds, 1.5× faster |
| `high_wind` | 10 m/s mean wind |
| `stress` | clustered crowd + more birds + stronger wind |

Seed ranges are disjoint so nothing leaks between splits:

| Split | Seeds | Used for |
|-------|-------|----------|
| Test | `10_000 + 100·type + k` | final evaluation, never trained or selected on |
| Validation | `20_000 + 100·type + k` | per-burst checkpoint selection |
| Training | unseeded | PPO rollouts |

# Usage

# Train

```bash
# Phase 4 (lethal) threshold trainer: resumes from the latest checkpoint
python -m training.train_until_threshold

# archive the old run and start over
python -m training.train_until_threshold --fresh

# resume from the best checkpoint instead of the latest
python -m training.train_until_threshold --from-best

# limit a session
python -m training.train_until_threshold --bursts 10
python -m training.train_until_threshold --hours 8

# named runs (each owns its own models, curves and curriculum state)
python -m training.train_until_threshold --run-name phase4_2 --start-lives 4
```

Other flags: `--restore-best`, `--no-warm-start`, `--warm-from PATH`, `--keep-old`. Edit the constants at the top of `train_until_threshold.py` (`PHASE_IDX`, `RUN_NAME`, `BURST_TIMESTEPS`, PPO hyper-parameters, …) to configure a run. Each burst trains 250 k steps, evaluates on the validation suite, updates the best/latest checkpoints, writes reports and updates `training_curve.jpeg`.

Monitor with TensorBoard:

```bash
tensorboard --logdir logs/tensorboard
```

# Evaluate

```bash
# 50 fixed test scenarios (10 types × 5 seeds): the "paper" numbers
python -m evaluation.evaluate --phase 4 --suite full

# 1 random seed from each type (10 episodes), reproducible draw
python -m evaluation.evaluate --phase 4 --suite test10 --test-rng 42

# unseeded random maps
python -m evaluation.evaluate --phase 4 --suite random --episodes 100

# also save every episode as a replay
python -m evaluation.evaluate --phase 4 --suite full --save-replays
```

# Visualise

```bash
python -m visualization.visualize
```

Opens a live 3-D rollout (buildings, drones, birds, wind field, coverage and fuel panels), runs 10 test scenarios (one per type), flags collisions, and saves summary figures, map snapshots from three camera angles and the network diagram when finished.

# Render replays to GIF / MP4

```bash
python -m evaluation.replay path/to/replays --mode top --format gif
python -m evaluation.replay path/to/ep001.npz --mode 3d --format mp4 --fps 15 --stride 3
```

A replay is a compact `.npz` (~0.3 MB per 1000-step episode) holding the static scene, per-step drone/bird state, fuel, lives, wind, coverage and collision flags, so it can be re-rendered at any time without the model or environment.

# Outputs

| Path | Contents |
|------|----------|
| `models/final_model.zip`, `models/vecnormalize.pkl` | canonical best model and its observation-normalisation stats |
| `models/threshold_<run>_{latest,best}.zip` | rolling checkpoints (each with `_vnorm.pkl`) |
| `models/milestone_<run>_covXX.zip` | snapshots when coverage milestones (70/80/85/90 %) are first hit |
| `logs/tensorboard/` | PPO logs + `reward_components/*` and `collisions/*` |
| `results/threshold_train_<run>/` | `eval_history.csv`, `training_curve.jpeg`, per-burst CSVs |
| `logs/results/final_eval/{csv,img}/` | final evaluation CSVs, 12-panel research figure, per-scenario breakdown |
| `results/visualization_res/` | visualiser figures, map snapshots, replays and rendered GIFs |
| `logs/networks/network_graph_final.jpeg` | actor-critic architecture diagram |

> **Important:** the model and its `VecNormalize` statistics go together. A policy evaluated or visualised without its matching `vecnormalize.pkl` behaves as if it had never been trained. The scripts look for `<model>_vnorm.pkl` first, then `models/vecnormalize.pkl`.

# Configuration reference

Everything lives in `configs/config.py`. The most commonly adjusted values:

| Setting | Default | Meaning |
|---------|---------|---------|
| `NUM_DRONES` | 5 | swarm size |
| `MAP_SIZE`, `GRID_SIZE` | 100, 20 | map side (m), coverage grid (20 × 20 = 400 cells) |
| `MAX_STEPS` | 1000 | episode length |
| `WIND_MEAN_SPEED`, `WIND_MODEL` | 6.0, `relative_drag` | wind strength and model |
| `OPTIMAL_SCAN_HEIGHT` | 20 m | altitude of maximum scan reward |
| `COVERAGE_REWARD` | 35 | per newly covered cell |
| `*_HIT_ONSET_PENALTY` | 40 / 30 / 150 | drone / bird / building contact onset |
| `COLLISION_LIVES`, `LIVES_SCHEDULE` | 1, `[6,4,3,2,1]` | lethal-collision curriculum |
| `DEATH_PENALTY` | 400 | per dead drone |
| `PHASE4_COLLISION_MODE` | `drone_dies` | or `episode_ends` |
| `DRONE_FUEL_STEPS`, `FUEL_LOW_FRAC` | ≈886, 0.15 | fuel budget and return threshold |
| `RTB_REQUIRES_POSITION` | `False` | if `True`, a drone only lands inside the spawn zone (raise `FUEL_LOW_FRAC` to ~0.30) |
| `REWARD_VERSION` | 4 | bump on any reward change |

PPO defaults (in `train_until_threshold.py`): learning rate 1e-4, clip 0.2, entropy 0.003, 4 envs × 1024 steps, batch 1024, 10 epochs, `target_kl` 0.03, network `[512, 256, 128]` for both policy and value heads.

# Reproducibility notes

- The environment draws everything from `self.np_random`, so the same `(scenario type, seed)` reproduces the same map, wind direction and bird tracks for every method you compare.
- Test seeds (`10xxx`) are never used for checkpoint selection; validation seeds (`20xxx`) are.
- `train_until_threshold.py` keeps `DummyVecEnv` on purpose: the lazy-tick curriculum mutates `config` in-process, which `SubprocVecEnv` workers would not see.
- Evaluation through a `VecEnv` auto-resets on episode end, so the terminal frame is not captured in replays (flagged as `terminal_frame_missing` in the replay metadata).

# Roadmap / known limitations

- Single centralised policy: observation and action size scale with the number of drones, so the swarm size is fixed at training time.
- Drones are point-mass models with simplified aerodynamics; no attitude control or sensor noise.
- Victims are rewarded when reached, but there is no explicit communication or relay model beyond a distance-based link visualisation.
- Ideas: decentralised/parameter-shared policy, sensor noise and partial observability, dynamic victims, sim-to-real transfer.

# License

Add your license here (e.g. MIT) and include a `LICENSE` file.
