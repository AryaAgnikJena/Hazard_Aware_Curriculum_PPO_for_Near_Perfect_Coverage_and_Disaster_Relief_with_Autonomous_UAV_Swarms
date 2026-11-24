"""
training/train_until_threshold.py   (v2.1)

Splits training_curve.jpeg into two readable 1×2 figures:
  • training_curve_coverage.jpeg  (Mean coverage per burst | Coverage vs cumulative steps)
  • training_curve_safety.jpeg    (Collisions per burst    | Deaths & lives curriculum)

CLI
  python -m training.train_until_threshold              # resume latest
  python -m training.train_until_threshold --fresh      # archive + start over
  python -m training.train_until_threshold --from-best  # resume from best
  python -m training.train_until_threshold --restore-best
  python -m training.train_until_threshold --keep-old   # skip auto-archive
"""

import os
import sys
import csv
import time
import glob
import shutil
import argparse
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import configs.config as config
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from training.callbacks import CSVLoggerCallback, RewardBreakdownCallback
from env.multi_uav_env import PHASE_ENVS
from env.scenarios import build_suite, VAL_SEED_BASE

from evaluation.evaluate import (
    RESULTS_ROOT,
    save_research_figure,
    results_to_dataframe,
    _render_env_map,
    _apply_research_style,
    _reset_style,
    _run_eval_loop,
    _categorize_collision_severity,
    C_BLUE, C_GREEN, C_ORANGE, C_YELLOW, C_GREY, C_RED, C_PURPLE, C_TEAL,
)

# ══════════════════════════════════════════════════════════════════════════
#  Configure here
# ══════════════════════════════════════════════════════════════════════════
PHASE_IDX          = 3            # 0-based → Phase 4 (use 2 for Phase 3)

# ── Run name ───────────────────────────────────────────────────────────────
RUN_NAME           = "phase4_2"
WARM_FROM          = None         # None → auto (see _auto_warm_source); or a model path w/o .zip
START_LIVES        = None         # None → LIVES_SCHEDULE[0]; must be a value of LIVES_SCHEDULE
MAX_PLOT_BARS      = 150          # training_curve bins bursts so it never has > this many bars

# ── Phase 4 (lethal collisions) ────────────────────────────────────────────
LETHAL             = bool(PHASE_IDX == 3 and getattr(config, "PHASE4_LETHAL", True))
FINAL_COVERAGE     = 85.0
FINAL_MAX_DEATHS   = 0.3
PROMOTE_COVERAGE   = 75.0
PROMOTE_MAX_DEATHS = 0.5
MAX_STAGE_BURSTS   = 10           # promote anyway after this many bursts at a stage
DEATH_SCORE_WEIGHT = 5.0          # "best" score = coverage − weight × deaths/episode
WARM_START         = True         # start Phase N from Phase N-1's best

COVERAGE_THRESHOLD = FINAL_COVERAGE if LETHAL else 80.0
BURST_TIMESTEPS    = 250_000
MAX_BURSTS         = 1            # bursts PER SESSION; override: --bursts N or --hours H
EVAL_EPISODES      = 50
STRATIFIED_VAL     = True         # per-burst eval on the fixed 10-type x 5-seed VALIDATION suite
N_ENVS             = 4            # DummyVecEnv (single process)

# ── PPO ────────────────────────────────────────────────────────────────────
LEARNING_RATE      = 1e-4
CLIP_RANGE         = 0.20
ENT_COEF           = 0.003
N_STEPS            = config.N_STEPS_TRAIN     # per env → 4×1024 = 4096 / rollout
BATCH_SIZE         = 1024
N_EPOCHS           = 10
TARGET_KL          = 0.03
LOAD_EXISTING      = True
STD_CLAMP_THRESHOLD = 1.0
STD_CLAMP_TARGET    = 0.6

# ── Resume / best-model policy ─────────────────────────────────────────────
RESUME_FROM         = "latest"    # "latest" | "best"
SMOOTH_N            = 5           # rolling window for "best" + stop test
REGRESSION_GUARD    = False
REGRESSION_DROP     = 8.0
REGRESSION_PATIENCE = 15

MILESTONES = ([70.0, 80.0, 85.0, 90.0] if LETHAL
              else [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0])

# ── Paths ──────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(config.MODEL_DIR, "final_model")      # = best (canonical)
VNORM_PATH = os.path.join(config.MODEL_DIR, "vecnormalize.pkl")


def _configure_paths(name: str):
    """(Re)compute every per-run path.  name="phase4" reproduces the original names."""
    global RUN_NAME, _TAG, LATEST_MODEL_PATH, LATEST_VNORM_PATH, BEST_MODEL_PATH
    global BEST_VNORM_PATH, VERSION_FILE, STAGE_FILE
    RUN_NAME          = name
    _TAG              = f"threshold_{name}"
    LATEST_MODEL_PATH = os.path.join(config.MODEL_DIR, f"{_TAG}_latest")
    LATEST_VNORM_PATH = LATEST_MODEL_PATH + "_vnorm.pkl"
    BEST_MODEL_PATH   = os.path.join(config.MODEL_DIR, f"{_TAG}_best")
    BEST_VNORM_PATH   = BEST_MODEL_PATH + "_vnorm.pkl"
    VERSION_FILE      = os.path.join(config.MODEL_DIR, f"{_TAG}_reward_version.txt")
    STAGE_FILE        = os.path.join(config.MODEL_DIR, f"{_TAG}_stage.txt")


_configure_paths(RUN_NAME)


def _run_glob(root: str, prefix: str):
    """glob(prefix*) but never files of a run whose name merely EXTENDS this one."""
    return [f for f in glob.glob(os.path.join(root, prefix + "*"))
            if not os.path.basename(f)[len(prefix):][:1].isdigit()]


_HIST_HEADER = [
    "burst", "cumulative_steps",
    "mean_coverage", "std_coverage", "smoothed_coverage",
    "mean_reward",
    "mean_collision_steps", "mean_collision_events",
    "clean_rate", "heavy_rate",
    "mean_steps", "mean_deaths", "lives", "milestone_hit",
]


# ══════════════════════════════════════════════════════════════════════════
#  Small helpers
# ══════════════════════════════════════════════════════════════════════════

def _ensure(*parts) -> str:
    d = os.path.join(*parts)
    os.makedirs(d, exist_ok=True)
    return d


def _burst_base() -> str:
    return _ensure(RESULTS_ROOT, f"threshold_train_{RUN_NAME}")


def _milestone_base(pct: float) -> str:
    return _ensure(RESULTS_ROOT, f"milestone_{RUN_NAME}_cov{int(pct):02d}")


def _rolling(values, n):
    return [float(np.mean(values[max(0, i - n + 1): i + 1]))
            for i in range(len(values))]


def _save_model(model, path: str):
    saved = model._logger
    model._logger = None            # TB logger holds an unpicklable lock
    try:
        model.save(path)
    finally:
        model._logger = saved


def _lazy_ticks_for(n_hit: int) -> int:
    return config.lazy_ticks_for(n_hit, len(MILESTONES))


# ══════════════════════════════════════════════════════════════════════════
#  History CSV (migrates older files)
# ══════════════════════════════════════════════════════════════════════════

def _migrate_history(path: str):
    """Rewrite an old-format eval_history.csv with the new header."""
    if not os.path.exists(path):
        return
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
        header = rows and list(rows[0].keys())
    if not rows or header == _HIST_HEADER:
        return
    shutil.copy(path, path.replace(".csv", "_legacy.csv"))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_HIST_HEADER)
        for r in rows:
            w.writerow([r.get(c, "") for c in _HIST_HEADER])
    print(f"  Migrated {path} to the v2 column layout (backup: *_legacy.csv)")


def _read_history(csv_d: str) -> list:
    path = os.path.join(csv_d, "eval_history.csv")
    if not os.path.exists(path):
        return []
    try:
        with open(path, newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _hist_float(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _resume_state(csv_d: str):
    rows = _read_history(csv_d)
    means  = [_hist_float(r, "mean_coverage") for r in rows]
    steps  = [int(_hist_float(r, "cumulative_steps")) for r in rows]
    colls  = [_hist_float(r, "mean_collision_steps") for r in rows]
    deaths = [_hist_float(r, "mean_deaths") for r in rows]
    lives  = [_hist_float(r, "lives") for r in rows]
    hit = {}
    for r in rows:
        for part in (r.get("milestone_hit") or "").split("+"):
            part = part.strip().rstrip("%")
            try:
                pct = float(part)
                hit.setdefault(pct, (int(_hist_float(r, "burst")),
                                     int(_hist_float(r, "cumulative_steps"))))
            except ValueError:
                pass
    return means, steps, colls, hit, deaths, lives


def _load_stage():
    """(lives, bursts_at_this_stage) — persisted so a restart keeps the curriculum."""
    sched = list(config.LIVES_SCHEDULE)
    try:
        lv, n = open(STAGE_FILE).read().split()
        lv, n = int(lv), int(n)
        if lv in sched:
            return lv, n
    except Exception:
        pass
    return (START_LIVES if START_LIVES in sched else sched[0]), 0


def _save_stage(lives: int, stage_bursts: int):
    os.makedirs(os.path.dirname(STAGE_FILE), exist_ok=True)
    with open(STAGE_FILE, "w") as f:
        f.write(f"{int(lives)} {int(stage_bursts)}")


def _resume_burst_num(csv_d: str) -> int:
    highest = 0
    if os.path.isdir(csv_d):
        for name in os.listdir(csv_d):
            if name.startswith("burst_") and name.endswith("_training.csv"):
                try:
                    highest = max(highest, int(name.split("_")[1]))
                except (IndexError, ValueError):
                    pass
    return highest + 1


def _append_hist_row(path: str, row: list):
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(_HIST_HEADER)
        w.writerow(row)


# ══════════════════════════════════════════════════════════════════════════
#  Run archiving / version stamp / compatibility
# ══════════════════════════════════════════════════════════════════════════

def _archive_old_run():
    """Move old results + model files aside (nothing is deleted)."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    res_arch = _ensure(RESULTS_ROOT, f"_archive_{stamp}")
    mod_arch = _ensure(config.MODEL_DIR, f"_archive_{stamp}")
    for d in [os.path.join(RESULTS_ROOT, f"threshold_train_{RUN_NAME}"),
              os.path.join(RESULTS_ROOT, f"{RUN_NAME}_latest"),
              os.path.join(RESULTS_ROOT, f"{RUN_NAME}_best"),
              *_run_glob(RESULTS_ROOT, f"milestone_{RUN_NAME}_")]:
        if os.path.isdir(d):
            shutil.move(d, os.path.join(res_arch, os.path.basename(d)))
    for f in [VERSION_FILE, STAGE_FILE,
              *_run_glob(config.MODEL_DIR, f"{_TAG}_"),
              *_run_glob(config.MODEL_DIR, f"milestone_{RUN_NAME}_")]:
        if os.path.isfile(f):
            shutil.move(f, os.path.join(mod_arch, os.path.basename(f)))
    print(f"  Archived old run → {res_arch}  and  {mod_arch}")


def _restore_best_to_final():
    if not (os.path.exists(BEST_MODEL_PATH + ".zip")
            and os.path.exists(BEST_VNORM_PATH)):
        print("  No best checkpoint found; nothing restored.")
        return
    shutil.copy(BEST_MODEL_PATH + ".zip", MODEL_PATH + ".zip")
    shutil.copy(BEST_VNORM_PATH, VNORM_PATH)
    print(f"  Restored {BEST_MODEL_PATH}.zip → {MODEL_PATH}.zip")


def _pick_resume_paths(from_best: bool):
    order = [(BEST_MODEL_PATH, BEST_VNORM_PATH),
             (LATEST_MODEL_PATH, LATEST_VNORM_PATH)]
    if not from_best and RESUME_FROM != "best":
        order.reverse()
    for m, v in order:
        if os.path.exists(m + ".zip") and os.path.exists(v):
            return m, v
    return None, None


def _stored_version():
    try:
        return int(open(VERSION_FILE).read().strip())
    except Exception:
        return None


def _write_version():
    os.makedirs(os.path.dirname(VERSION_FILE), exist_ok=True)
    with open(VERSION_FILE, "w") as f:
        f.write(str(int(config.REWARD_VERSION)))


def _has_saved_run() -> bool:
    return (os.path.exists(LATEST_MODEL_PATH + ".zip")
            or os.path.exists(BEST_MODEL_PATH + ".zip"))


def _saved_model_obs_mismatch(EnvClass, model_path) -> bool:
    try:
        probe = PPO.load(model_path, device="cpu")
        return (tuple(probe.observation_space.shape)
                != tuple(EnvClass().observation_space.shape))
    except Exception as exc:
        print(f"  Could not inspect saved model ({exc}); treating as incompatible.")
        return True


_RESET_REWARD_STATS = False


def _check_run_compat(EnvClass, from_best: bool, keep_old: bool):
    """Archive the stored run if it was made under an incompatible reward / obs."""
    global _RESET_REWARD_STATS
    if keep_old or not _has_saved_run():
        return
    reason = None
    stored = _stored_version()
    if stored != int(config.REWARD_VERSION):
        if stored in list(getattr(config, "REWARD_COMPATIBLE_WITH", [])):
            print(f"\n  ℹ Reward changed (v{stored} → v{config.REWARD_VERSION}); the stored policy "
                  f"is still a valid starting point → keeping it and resetting the "
                  f"reward-normalisation statistics.")
            _RESET_REWARD_STATS = True
        else:
            reason = f"reward version {stored} → {config.REWARD_VERSION}"
    if reason is None and not _RESET_REWARD_STATS:
        m, _ = _pick_resume_paths(from_best)
        if m and _saved_model_obs_mismatch(EnvClass, m):
            reason = "observation size changed"
    if reason:
        print(f"\n  ⚠ Stored run is incompatible ({reason}).")
        print("    Its policy and VecNormalize reward stats were learned under a")
        print("    different reward → archiving it and starting fresh "
              "(use --keep-old to skip).")
        _archive_old_run()


# ══════════════════════════════════════════════════════════════════════════
#  Optimizer / policy utilities
# ══════════════════════════════════════════════════════════════════════════

def _configure_optimizer(model):
    def _lr_fn(_, v=float(LEARNING_RATE)): return v
    def _clip_fn(_, v=float(CLIP_RANGE)):  return v
    model.learning_rate = _lr_fn
    model.clip_range    = _clip_fn
    model.ent_coef      = ENT_COEF
    model._setup_lr_schedule()
    for pg in model.policy.optimizer.param_groups:
        pg["lr"] = LEARNING_RATE


def _policy_std(model) -> float:
    return float(model.policy.log_std.exp().mean().item())


def _clamp_std(model):
    if STD_CLAMP_THRESHOLD is None:
        return
    with torch.no_grad():
        if _policy_std(model) > STD_CLAMP_THRESHOLD:
            print(f"  Clamping log_std: std={_policy_std(model):.2f} → {STD_CLAMP_TARGET}")
            model.policy.log_std.data.fill_(torch.log(torch.tensor(STD_CLAMP_TARGET)))


def _ppo_overrides():
    return dict(n_steps=N_STEPS, batch_size=BATCH_SIZE,
                n_epochs=N_EPOCHS, target_kl=TARGET_KL)


def _expand_state(old_state: dict, new_state: dict) -> dict:
    """
    Copy `old_state` into the layout of `new_state`. Equal shapes are copied;
    a 2-D weight that only gained INPUT columns keeps its old columns and gets
    zeros for the new ones. Anything else keeps its fresh init.
    """
    out = {}
    for k, nv in new_state.items():
        ov = old_state.get(k)
        if ov is None:
            out[k] = nv
        elif tuple(ov.shape) == tuple(nv.shape):
            out[k] = ov.clone() if hasattr(ov, "clone") else ov.copy()
        elif (len(ov.shape) == 2 and len(nv.shape) == 2
              and ov.shape[0] == nv.shape[0] and ov.shape[1] < nv.shape[1]):
            t = nv * 0
            t[:, :ov.shape[1]] = ov
            out[k] = t
        else:
            out[k] = nv
    return out


def _auto_warm_source():
    """(model path, env class of that checkpoint, description) a NEW run starts from."""
    if WARM_FROM:
        return WARM_FROM, PHASE_ENVS[PHASE_IDX], "explicit --warm-from"
    if RUN_NAME == f"phase{PHASE_IDX + 1}":
        if PHASE_IDX >= 1:
            return (os.path.join(config.MODEL_DIR, f"threshold_phase{PHASE_IDX}_best"),
                    PHASE_ENVS[PHASE_IDX - 1], "previous phase best")
        return None, None, ""
    base = f"phase{PHASE_IDX + 1}"
    cands = [os.path.join(config.MODEL_DIR, f"milestone_{base}_cov{p:02d}")
             for p in (90, 85, 80, 70)]
    cands += [os.path.join(config.MODEL_DIR, f"threshold_{base}_best"),
              os.path.join(config.MODEL_DIR, f"threshold_{base}_latest")]
    for c in cands:
        if os.path.exists(c + ".zip") and os.path.exists(c + "_vnorm.pkl"):
            return c, PHASE_ENVS[PHASE_IDX], f"auto: {os.path.basename(c)}"
    return None, None, ""


def _warm_start(raw_env):
    """Initialise a NEW run from another checkpoint. Returns (env, model) or None."""
    if not WARM_START:
        return None
    src, src_env, why = _auto_warm_source()
    if src is None or not (os.path.exists(src + ".zip")
                           and os.path.exists(src + "_vnorm.pkl")):
        print(f"  (no warm-start checkpoint found{': ' + str(src) if src else ''} — cannot warm-start)")
        return None

    print(f"  Warm-starting from {src}.zip   [{why}]")
    if os.path.basename(src).endswith(("_best", "_latest")) and START_LIVES is None:
        print(f"  ⚠ {os.path.basename(src)} comes from the FINAL stage of the old run; the "
              f"curriculum will still start at lives={config.LIVES_SCHEDULE[0]} "
              f"(use --start-lives to change).")
    old = PPO.load(src, device="cpu")
    old_vn = VecNormalize.load(src + "_vnorm.pkl", DummyVecEnv([src_env]))

    env = VecNormalize(raw_env, norm_obs=True, norm_reward=True,
                       clip_obs=10.0, gamma=config.PPO_GAMMA)
    model = PPO("MlpPolicy", env, verbose=1,
                learning_rate=LEARNING_RATE, gamma=config.PPO_GAMMA,
                gae_lambda=0.95, clip_range=CLIP_RANGE, ent_coef=ENT_COEF,
                tensorboard_log=config.LOG_DIR,
                policy_kwargs=old.policy_kwargs, **_ppo_overrides())
    model.policy.load_state_dict(
        _expand_state(old.policy.state_dict(), model.policy.state_dict()))

    n_old = int(old_vn.obs_rms.mean.shape[0])
    n_new = int(env.obs_rms.mean.shape[0])
    env.obs_rms.mean[:n_old] = old_vn.obs_rms.mean
    env.obs_rms.var[:n_old]  = old_vn.obs_rms.var
    env.obs_rms.count        = old_vn.obs_rms.count
    old_vn.close()
    print(f"  Observation size {n_old} → {n_new}  "
          f"(+{n_new - n_old} new inputs start at weight 0)")

    bak = os.path.join(config.MODEL_DIR,
                       f"final_model_phase{PHASE_IDX}" if RUN_NAME == f"phase{PHASE_IDX + 1}"
                       else f"final_model_before_{RUN_NAME}")
    if os.path.exists(MODEL_PATH + ".zip") and not os.path.exists(bak + ".zip"):
        shutil.copy(MODEL_PATH + ".zip", bak + ".zip")
        if os.path.exists(VNORM_PATH):
            shutil.copy(VNORM_PATH, bak + "_vnorm.pkl")
        print(f"  Backed up previous final_model → {bak}.zip")
    _clamp_std(model)
    return env, model


def _load_or_create(raw_env, from_best: bool):
    """Returns (train_env, model)."""
    m_path, v_path = _pick_resume_paths(from_best)
    if LOAD_EXISTING and m_path is not None:
        print(f"  Loading existing model: {m_path}.zip")
        env = VecNormalize.load(v_path, raw_env)
        env.training, env.norm_reward = True, True
        model = PPO.load(m_path, env=env, **_ppo_overrides())
        model.set_env(env)
        _clamp_std(model)
        return env, model

    ws = _warm_start(raw_env)
    if ws is not None:
        return ws
    print(f"  Creating fresh model for Phase {PHASE_IDX + 1}.")
    env = VecNormalize(raw_env, norm_obs=True, norm_reward=True,
                       clip_obs=10.0, gamma=config.PPO_GAMMA)
    model = PPO(
        "MlpPolicy", env, verbose=1,
        learning_rate=LEARNING_RATE,
        gamma=config.PPO_GAMMA, gae_lambda=0.95,
        clip_range=CLIP_RANGE, ent_coef=ENT_COEF,
        tensorboard_log=config.LOG_DIR,
        policy_kwargs=dict(net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
                           log_std_init=-0.5),
        **_ppo_overrides(),
    )
    return env, model


def _restore_best(train_env):
    new_env = VecNormalize.load(BEST_VNORM_PATH, train_env.venv)
    new_env.training, new_env.norm_reward = True, True
    model = PPO.load(BEST_MODEL_PATH, env=new_env, **_ppo_overrides())
    model.set_env(new_env)
    _configure_optimizer(model)
    _clamp_std(model)
    return new_env, model


# ══════════════════════════════════════════════════════════════════════════
#  Evaluation + reports
# ══════════════════════════════════════════════════════════════════════════

def _quick_eval(model, EnvClass, vnorm_path: str, episodes: int) -> dict:
    vec = DummyVecEnv([EnvClass])
    if os.path.exists(vnorm_path):
        vec = VecNormalize.load(vnorm_path, vec)
        vec.training, vec.norm_reward = False, False
    else:
        vec = VecNormalize(vec, norm_obs=True, norm_reward=False,
                           clip_obs=10.0, gamma=config.PPO_GAMMA)
        vec.training = False
    results = _run_eval_loop(
        model, vec, episodes, label="eval",
        scenarios=build_suite(VAL_SEED_BASE) if STRATIFIED_VAL else None)
    vec.close()
    return results


def _write_report(results, burst_num, total_steps, mean_cov,
                  csv_d, img_d, label, prefix):
    cov = np.array(results["coverage_list"])
    cs  = np.array(results["collision_steps_ep"], float)
    st  = np.array(results["step_list"], float)
    clean, light, heavy = _categorize_collision_severity(cs, st)

    ep_csv = os.path.join(csv_d, f"{prefix}_eval.csv")
    results_to_dataframe(results).to_csv(ep_csv, index=False)

    summary_csv = os.path.join(csv_d, f"{prefix}_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["burst", "total_steps", "mean_cov", "std_cov", "min_cov",
                    "max_cov", "mean_reward", "std_reward",
                    "mean_collision_steps", "mean_collision_events",
                    "mean_coll_drone", "mean_coll_bird", "mean_coll_building",
                    "clean_rate", "light_rate", "heavy_rate",
                    "height_event_rate", "lazy_event_rate",
                    "mean_steps", "mean_landed", "mean_landed_in_zone", "mean_dead"])
        w.writerow([
            burst_num, total_steps,
            round(float(cov.mean()), 2), round(float(cov.std()), 2),
            round(float(cov.min()), 2), round(float(cov.max()), 2),
            round(float(np.mean(results["reward_list"])), 1),
            round(float(np.std(results["reward_list"])), 1),
            round(float(cs.mean()), 2),
            round(float(np.mean(results["collision_events_ep"])), 2),
            round(float(np.mean(results["coll_kind_steps"]["drone"])), 2),
            round(float(np.mean(results["coll_kind_steps"]["bird"])), 2),
            round(float(np.mean(results["coll_kind_steps"]["building"])), 2),
            round(clean, 3), round(light, 3), round(heavy, 3),
            round(float(np.mean(results["height_killed_ep"])), 3),
            round(float(np.mean(results["lazy_killed_ep"])), 3),
            round(float(st.mean()), 1),
            round(float(np.mean(results["landed_count_ep"])), 2),
            round(float(np.mean(results["landed_in_zone_ep"])), 2),
            round(float(np.mean(results["drones_dead_ep"])), 3),
        ])

    research = save_research_figure(
        results, img_d, f"{prefix}_research.jpeg",
        title_suffix=(f"  ·  {RUN_NAME} {label}  ·  Burst {burst_num}  ·  "
                      f"{total_steps:,} steps  ·  μcov={mean_cov:.1f}%  ·  "
                      f"coll steps/ep={cs.mean():.0f}"))

    env_map = None
    raw_env = results.get("last_raw_env")
    if raw_env is not None:
        env_map = os.path.join(img_d, f"{prefix}_env_map.jpeg")
        try:
            _render_env_map(raw_env, env_map,
                            title=(f"{RUN_NAME} — {label}  "
                                   f"(burst {burst_num}, {total_steps:,} steps)"))
        except Exception as exc:
            print(f"  [WARNING] Env map failed: {exc}")
            env_map = None
    return dict(ep_csv=ep_csv, summary_csv=summary_csv, research_fig=research,
                env_map=env_map, clean=clean, light=light, heavy=heavy)


def _print_burst_results(burst_num, results, mean_cov, std_cov, info):
    cs = np.array(results["collision_steps_ep"], float)
    st = np.array(results["step_list"], float)
    kinds = results["coll_kind_steps"]
    print(f"\n  ── Burst {burst_num} results ─────────────────────────")
    print(f"  Mean coverage   : {mean_cov:.1f}%  (σ={std_cov:.1f}%)")
    print(f"  Mean return     : {np.mean(results['reward_list']):,.0f}")
    print(f"  Collision steps : {cs.mean():.1f}/ep  "
          f"({100 * cs.sum() / max(st.sum(), 1):.1f}% of steps)  "
          f"events={np.mean(results['collision_events_ep']):.1f}/ep")
    print(f"     by kind      : drone={np.mean(kinds['drone']):.1f}  "
          f"bird={np.mean(kinds['bird']):.1f}  "
          f"building={np.mean(kinds['building']):.1f}   (steps/ep)")
    print(f"  Severity        : clean={info['clean'] * 100:.0f}%  "
          f"light={info['light'] * 100:.0f}%  heavy={info['heavy'] * 100:.0f}%")
    print(f"  Mean steps      : {st.mean():.0f}")
    print(f"  Landed / zone   : {np.mean(results['landed_count_ep']):.1f} / "
          f"{np.mean(results['landed_in_zone_ep']):.1f} of {config.NUM_DRONES}")
    if LETHAL:
        d = np.array(results["drones_dead_ep"], float)
        print(f"  Drones dead     : {d.mean():.2f}/ep   episodes with 0 deaths: "
              f"{np.mean(d == 0) * 100:.0f}%   (lives per drone = {config.COLLISION_LIVES})")


def _save_latest_burst(burst_num, total_steps, mean_cov, results):
    base = _ensure(RESULTS_ROOT, f"{RUN_NAME}_latest")
    return _write_report(results, burst_num, total_steps, mean_cov,
                         _ensure(base, "csv"), _ensure(base, "img"),
                         "latest", "latest")


def _save_best_burst(model, env, burst_num, total_steps, mean_cov, results, score):
    base = _ensure(RESULTS_ROOT, f"{RUN_NAME}_best")
    _save_model(model, BEST_MODEL_PATH)
    env.save(BEST_VNORM_PATH)
    _save_model(model, MODEL_PATH)               # canonical final_model = best
    env.save(VNORM_PATH)
    _write_report(results, burst_num, total_steps, mean_cov,
                  _ensure(base, "csv"), _ensure(base, "img"),
                  "best", "best")
    print(f"  ★ New best (smoothed {score:.1f}%) → {BEST_MODEL_PATH}.zip  "
          f"and {MODEL_PATH}.zip")


def _save_milestone_snapshot(model, env, pct, burst_num, total_steps, results):
    tag  = f"cov{int(pct):02d}"
    base = _milestone_base(pct)
    m_path = os.path.join(config.MODEL_DIR, f"milestone_{RUN_NAME}_{tag}")
    _save_model(model, m_path)
    env.save(m_path + "_vnorm.pkl")
    cov_mean = float(np.mean(results["coverage_list"]))
    _write_report(results, burst_num, total_steps, cov_mean,
                  _ensure(base, "csv"), _ensure(base, "img"),
                  f"milestone {pct:.0f}%", f"milestone_{tag}")
    print(f"  ★★ MILESTONE {pct:.0f}% (burst {burst_num}, {total_steps:,} steps) → {m_path}.zip")


# ══════════════════════════════════════════════════════════════════════════
#  Training-curve figures (split into 2 readable 1×2 images)
# ══════════════════════════════════════════════════════════════════════════

def _bin_blocks(a, k, last=False):
    """Collapse every k consecutive values into one (mean, or the last value)."""
    a = np.asarray(a, float)
    if k <= 1 or len(a) == 0:
        return a
    out = np.full(int(np.ceil(len(a) / k)), np.nan)
    for i in range(len(out)):
        blk = a[i * k:(i + 1) * k]
        blk = blk[np.isfinite(blk)]
        if len(blk):
            out[i] = blk[-1] if last else blk.mean()
    return out


def _save_training_curve(means, steps, colls, milestones_hit, out_path,
                         deaths=None, lives_list=None):
    """Save training curves as two readable 1×2 figures (coverage & safety)."""
    _apply_research_style()
    stem, ext = os.path.splitext(out_path)
    ext = ext or ".jpeg"
    saved_paths = []

    k = max(1, int(np.ceil(len(means) / MAX_PLOT_BARS)))
    smooth_raw  = _rolling(means, SMOOTH_N)
    coll_sm_raw = _rolling(list(np.nan_to_num(np.array(colls, float))), SMOOTH_N)
    means   = _bin_blocks(means, k)
    steps   = _bin_blocks(steps, k, last=True)
    colls   = _bin_blocks(colls, k)
    smooth  = _bin_blocks(smooth_raw, k)
    coll_sm = _bin_blocks(coll_sm_raw, k)
    if deaths is not None:
        deaths = _bin_blocks(deaths, k)
    if lives_list is not None:
        lives_list = _bin_blocks(lives_list, k, last=True)
    b = np.arange(len(means)) * k + (k + 1) / 2
    bin_note = f"  (1 bar = {k} bursts)" if k > 1 else ""
    ms_cmap = plt.cm.RdYlGn
    sorted_ms = sorted(MILESTONES)
    footer = (f"Phase {PHASE_IDX + 1} · LR={LEARNING_RATE} · clip={CLIP_RANGE} · "
              f"N_ENVS={N_ENVS}×{N_STEPS} · batch={BATCH_SIZE} · burst={BURST_TIMESTEPS:,}")

    # ── Figure 1: Coverage (per burst & vs cumulative steps) ─────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    fig.patch.set_facecolor("#ffffff")
    fig.suptitle(f"Threshold Training — {RUN_NAME} · Coverage "
                 f"(target ≥ {COVERAGE_THRESHOLD:.0f}% smoothed)",
                 fontsize=12.5, fontweight="bold", color="#1f2328", y=1.01)

    ax = axes[0]
    ax.bar(b, means, color=[ms_cmap(min(m, 100) / 100) for m in means],
           alpha=0.75, edgecolor="none", width=0.8 * k)
    ax.plot(b, smooth, color="#1f2328", lw=1.8, label=f"rolling mean ({SMOOTH_N})")
    for i, ms in enumerate(sorted_ms):
        ax.axhline(ms, color=ms_cmap((i + 1) / len(sorted_ms)), ls=":", lw=0.9, alpha=0.6)
    ax.axhline(COVERAGE_THRESHOLD, color=C_ORANGE, ls="--", lw=1.3,
               label=f"target {COVERAGE_THRESHOLD:.0f}%")
    if len(smooth):
        ib = int(np.argmax(smooth))
        ax.annotate(f"best smoothed {smooth[ib]:.1f}%", xy=(b[ib], smooth[ib]),
                    xytext=(b[ib], min(smooth[ib] + 14, 100)), ha="center",
                    fontsize=8.5, color=C_TEAL,
                    arrowprops=dict(arrowstyle="-", color=C_TEAL, lw=0.8))
    ax.set_ylim(0, 105); ax.set_xlabel("Burst #"); ax.set_ylabel("Mean coverage %")
    ax.set_title("Mean coverage per burst" + bin_note); ax.legend(fontsize=8)

    ax = axes[1]
    if len(steps):
        ax.plot(steps, means, "o-", color=C_BLUE, lw=1.1, ms=3.5, alpha=0.5, label="burst mean")
        ax.plot(steps, smooth, color="#1f2328", lw=1.9, label=f"rolling ({SMOOTH_N})")
        for pct, (mb, ms_step) in milestones_hit.items():
            ax.axvline(ms_step, color=C_YELLOW, ls=":", lw=0.8, alpha=0.55)
        ax.axhline(COVERAGE_THRESHOLD, color=C_ORANGE, ls="--", lw=1.3)
    ax.set_ylim(0, 105); ax.set_xlabel("Cumulative training steps")
    ax.set_ylabel("Mean coverage %"); ax.set_title("Coverage vs cumulative steps")
    ax.legend(fontsize=8)

    fig.text(0.99, 0.0, footer, ha="right", va="bottom",
             fontsize=7.5, color=C_GREY, style="italic")
    fig.tight_layout()
    p1 = f"{stem}_coverage{ext}"
    fig.savefig(p1, dpi=150, format="jpeg", bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    saved_paths.append(p1)

    # ── Figure 2: Safety (collisions & deaths/lives curriculum) ──────────
    has_deaths = bool(LETHAL and deaths is not None)
    fig, axes = plt.subplots(1, 2 if has_deaths else 1,
                             figsize=(15 if has_deaths else 8.5, 5.8))
    fig.patch.set_facecolor("#ffffff")
    fig.suptitle(f"Threshold Training — {RUN_NAME} · Safety & Curriculum",
                 fontsize=12.5, fontweight="bold", color="#1f2328", y=1.01)

    ax = axes[0] if has_deaths else axes
    c = np.array(colls, float)
    ok = np.isfinite(c)
    if ok.any():
        ax.bar(b[ok], c[ok], color=C_RED, alpha=0.7, width=0.8 * k, edgecolor="none",
               label="collision steps / episode")
        ax.plot(b[ok], coll_sm[ok], color="#1f2328", lw=1.8, label=f"rolling ({SMOOTH_N})")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no collision data yet", transform=ax.transAxes,
                ha="center", va="center", color=C_GREY, style="italic")
    ax.set_xlabel("Burst #"); ax.set_ylabel("Collision steps per episode")
    ax.set_title("Collisions per episode (eval), per burst" + bin_note)

    if has_deaths:
        ax = axes[1]
        d = np.array(deaths, float)
        okd = np.isfinite(d)
        if okd.any():
            ax.bar(b[okd], d[okd], color=C_PURPLE, alpha=0.75, width=0.8 * k,
                   edgecolor="none", label="drones dead / episode")
            ax.axhline(FINAL_MAX_DEATHS, color=C_GREEN, ls="--", lw=1.2,
                       label=f"final target ≤ {FINAL_MAX_DEATHS}")
            ax.set_ylabel("Drones dead per episode")
        ax2 = ax.twinx()
        lv = np.array(lives_list if lives_list is not None else [], float)
        okl = np.isfinite(lv)
        if okl.any():
            ax2.step(b[:len(lv)][okl], lv[okl], where="mid", color=C_YELLOW, lw=1.8,
                     label="lives per drone")
            ax2.set_ylabel("Lives per drone", color=C_YELLOW)
            ax2.grid(False)
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8)
        ax.set_xlabel("Burst #"); ax.set_title("Deaths per episode and lives curriculum" + bin_note)

    fig.text(0.99, 0.0, footer, ha="right", va="bottom",
             fontsize=7.5, color=C_GREY, style="italic")
    fig.tight_layout()
    p2 = f"{stem}_safety{ext}"
    fig.savefig(p2, dpi=150, format="jpeg", bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    saved_paths.append(p2)

    _reset_style()
    return saved_paths


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

def train_until_threshold(from_best=False, keep_old=False,
                          max_bursts=None, max_hours=None):
    EnvClass   = PHASE_ENVS[PHASE_IDX]
    n_bursts   = int(max_bursts or MAX_BURSTS)
    _ensure(config.LOG_DIR); _ensure(config.MODEL_DIR)

    _check_run_compat(EnvClass, from_best, keep_old)      # may archive old run
    burst_base = _burst_base()
    csv_d      = _ensure(burst_base, "csv")
    img_d      = _ensure(burst_base, "img")
    hist_csv   = os.path.join(csv_d, "eval_history.csv")
    _migrate_history(hist_csv)

    start_burst_num = _resume_burst_num(csv_d)
    (burst_means, burst_total_steps, burst_coll, milestones_hit,
     burst_deaths, burst_lives) = _resume_state(csv_d)
    total_steps_done = (start_burst_num - 1) * BURST_TIMESTEPS
    remaining_ms = sorted(ms for ms in MILESTONES if ms not in milestones_hit)
    config.LAZY_KILL_TICKS = _lazy_ticks_for(len(milestones_hit))

    # curriculum state (Phase 4)
    if LETHAL:
        lives, stage_bursts = _load_stage()
        config.COLLISION_LIVES = lives
    else:
        lives, stage_bursts = None, 0

    def _comp(m, d):                      # per-burst "best model" score
        return m - DEATH_SCORE_WEIGHT * (d if d == d else 0.0)
    composite = [_comp(m, d) for m, d in zip(burst_means, burst_deaths)]
    stage_hist = (composite[-stage_bursts:] if stage_bursts else []) if LETHAL else list(composite)
    best_score = max(_rolling(stage_hist, SMOOTH_N), default=-1e9)
    best_burst = start_burst_num - 1
    resuming = start_burst_num > 1

    print(f"\n{'=' * 64}")
    print(f"  Threshold Trainer v2.1 — run '{RUN_NAME}' (Phase {PHASE_IDX + 1}: {EnvClass.__name__})"
          f"{'  [LETHAL collisions]' if LETHAL else ''}")
    if resuming:
        print(f"  RESUMING from burst {start_burst_num} ({total_steps_done:,} steps done)")
    print(f"  Session limit   : {n_bursts} bursts"
          + (f" or {max_hours} h" if max_hours else "")
          + "  (rerun the same command to continue)")
    if LETHAL:
        print(f"  Curriculum      : lives {list(config.LIVES_SCHEDULE)}  → now {lives} "
              f"({stage_bursts} bursts at this stage)")
        print(f"  Promote when    : coverage ≥ {PROMOTE_COVERAGE:.0f}% and deaths ≤ "
              f"{PROMOTE_MAX_DEATHS}/ep (smoothed over {SMOOTH_N} bursts)")
        print(f"  Finish when     : lives=1, coverage ≥ {FINAL_COVERAGE:.0f}% and deaths ≤ "
              f"{FINAL_MAX_DEATHS}/ep")
        print(f"  Collision mode  : {getattr(config, 'PHASE4_COLLISION_MODE', 'drone_dies')}")
    else:
        print(f"  Target          : smoothed coverage ≥ {COVERAGE_THRESHOLD:.0f}%")
    print(f"  Resume source   : {'best' if (from_best or RESUME_FROM == 'best') else 'latest'}")
    print(f"  Rollout         : {N_ENVS}×{N_STEPS} = {N_ENVS * N_STEPS}  "
          f"batch {BATCH_SIZE}  epochs {N_EPOCHS}  target_kl {TARGET_KL}")
    print(f"  LR / clip / ent : {LEARNING_RATE} / {CLIP_RANGE} / {ENT_COEF}")
    print(f"  Regression guard: {'ON' if REGRESSION_GUARD else 'off'}")
    print(f"{'=' * 64}\n")

    raw_env = make_vec_env(EnvClass, n_envs=N_ENVS, vec_env_cls=DummyVecEnv)
    train_env, model = _load_or_create(raw_env, from_best)
    if _RESET_REWARD_STATS:
        from stable_baselines3.common.running_mean_std import RunningMeanStd
        train_env.ret_rms = RunningMeanStd()
        print("  Reward-normalisation statistics reset.")
    _configure_optimizer(model)
    _write_version()
    print(f"  Effective LR : {model.policy.optimizer.param_groups[0]['lr']}\n")

    overall_start = time.time()
    bad_streak = 0
    finished = False

    for burst_num in range(start_burst_num, start_burst_num + n_bursts):
        print(f"\n{'─' * 64}\n  Burst {burst_num}  (total so far: {total_steps_done:,} steps)"
              + (f"   lives/drone = {lives}" if LETHAL else "") + f"\n{'─' * 64}")

        callbacks = [
            CSVLoggerCallback(os.path.join(csv_d, f"burst_{burst_num:02d}_training.csv")),
            RewardBreakdownCallback(),
        ]
        t0 = time.time()
        try:
            model.learn(total_timesteps=BURST_TIMESTEPS, callback=callbacks,
                        reset_num_timesteps=True, progress_bar=True)
        except KeyboardInterrupt:
            print("\n  ⏹ Interrupted — saving the current model as 'latest' "
                  "(this partial burst is not evaluated).")
            _save_model(model, LATEST_MODEL_PATH)
            train_env.save(LATEST_VNORM_PATH)
            return
        total_steps_done += BURST_TIMESTEPS
        el = time.time() - t0
        print(f"  Trained in {el:.0f}s ({BURST_TIMESTEPS / el:.0f} steps/s)")

        _save_model(model, LATEST_MODEL_PATH)
        train_env.save(LATEST_VNORM_PATH)

        print(f"  Evaluating {EVAL_EPISODES} episodes …")
        results = _quick_eval(model, EnvClass, LATEST_VNORM_PATH, EVAL_EPISODES)

        cov = np.array(results["coverage_list"])
        mean_cov, std_cov = float(cov.mean()), float(cov.std())
        mean_rew = float(np.mean(results["reward_list"]))
        cs = np.array(results["collision_steps_ep"], float)
        st = np.array(results["step_list"], float)
        mean_cs = float(cs.mean())
        mean_ce = float(np.mean(results["collision_events_ep"]))
        mean_dead = float(np.mean(results["drones_dead_ep"]))
        clean, light, heavy = _categorize_collision_severity(cs, st)

        burst_means.append(mean_cov)
        burst_total_steps.append(total_steps_done)
        burst_coll.append(mean_cs)
        burst_deaths.append(mean_dead)
        burst_lives.append(float(lives) if LETHAL else float("nan"))
        stage_hist.append(_comp(mean_cov, mean_dead))
        if LETHAL:
            stage_bursts += 1
        score = float(np.mean(stage_hist[-SMOOTH_N:]))
        sm_cov = float(np.mean(burst_means[-SMOOTH_N:]))
        sm_dead = float(np.mean(burst_deaths[-SMOOTH_N:]))

        _print_burst_results(burst_num, results, mean_cov, std_cov,
                             dict(clean=clean, light=light, heavy=heavy))
        print(f"  Smoothed        : coverage {sm_cov:.1f}%"
              + (f"   deaths {sm_dead:.2f}/ep" if LETHAL else "")
              + f"   (best score {max(best_score, score):.1f})")
        print(f"  Policy std      : {_policy_std(model):.2f}")

        # ── milestones ────────────────────────────────────────────────
        newly = [ms for ms in remaining_ms if mean_cov >= ms]
        for ms in newly:
            milestones_hit[ms] = (burst_num, total_steps_done)
            _save_milestone_snapshot(model, train_env, ms, burst_num,
                                     total_steps_done, results)
        remaining_ms = [ms for ms in remaining_ms if ms not in newly]
        if newly:
            new_ticks = _lazy_ticks_for(len(milestones_hit))
            if new_ticks != config.LAZY_KILL_TICKS:
                print(f"  ⚡ lazy ticks {config.LAZY_KILL_TICKS} → {new_ticks}")
                config.LAZY_KILL_TICKS = new_ticks
        milestone_label = "+".join(f"{m:.0f}%" for m in newly)

        # ── best (smoothed, per curriculum stage) ─────────────────────
        if score > best_score:
            best_score, best_burst = score, burst_num
            _save_best_burst(model, train_env, burst_num, total_steps_done,
                             mean_cov, results, score)

        _save_latest_burst(burst_num, total_steps_done, mean_cov, results)

        _append_hist_row(hist_csv, [
            burst_num, total_steps_done,
            round(mean_cov, 2), round(std_cov, 2), round(sm_cov, 2),
            round(mean_rew, 1),
            round(mean_cs, 2), round(mean_ce, 2),
            round(clean, 3), round(heavy, 3),
            round(float(st.mean()), 1),
            round(mean_dead, 3), (lives if LETHAL else ""),
            milestone_label,
        ])

        curve_base = os.path.join(img_d, "training_curve.jpeg")
        curve_paths = _save_training_curve(burst_means, burst_total_steps, burst_coll,
                                           milestones_hit, curve_base,
                                           deaths=burst_deaths, lives_list=burst_lives)
        latest_img = _ensure(RESULTS_ROOT, f"{RUN_NAME}_latest", "img")
        for cp in curve_paths:
            shutil.copy(cp, os.path.join(latest_img, os.path.basename(cp)))

        # ── curriculum / stop test ────────────────────────────────────
        if LETHAL:
            sched = list(config.LIVES_SCHEDULE)
            at_final = (lives == sched[-1])
            enough = stage_bursts >= SMOOTH_N
            if at_final:
                finished = (enough and sm_cov >= FINAL_COVERAGE
                            and sm_dead <= FINAL_MAX_DEATHS)
                print(f"\n  Final stage (lives={lives}): coverage {sm_cov:.1f}/{FINAL_COVERAGE:.0f}%  "
                      f"deaths {sm_dead:.2f}/{FINAL_MAX_DEATHS}  →  "
                      f"{'✓ REACHED' if finished else ('✗ keep training' if enough else f'(need {SMOOTH_N - stage_bursts} more bursts at this stage)')}")
            else:
                ok = enough and sm_cov >= PROMOTE_COVERAGE and sm_dead <= PROMOTE_MAX_DEATHS
                forced = (not ok) and stage_bursts >= MAX_STAGE_BURSTS and sm_cov >= PROMOTE_COVERAGE
                print(f"\n  Stage lives={lives} ({stage_bursts}/{MAX_STAGE_BURSTS} bursts): "
                      f"coverage {sm_cov:.1f}/{PROMOTE_COVERAGE:.0f}%  "
                      f"deaths {sm_dead:.2f}/{PROMOTE_MAX_DEATHS}  →  "
                      f"{'✓ PROMOTE' if ok else ('⏩ PROMOTE (stage time limit)' if forced else ('✗ stay' if enough else f'(need {SMOOTH_N - stage_bursts} more bursts)'))}")
                if ok or forced:
                    lives = sched[sched.index(lives) + 1]
                    config.COLLISION_LIVES = lives
                    stage_bursts, stage_hist, best_score, bad_streak = 0, [], -1e9, 0
                    print(f"  ⬆ Curriculum: drones now have {lives} "
                          f"{'life' if lives == 1 else 'lives'} each")
            _save_stage(lives, stage_bursts)
        else:
            finished = (sm_cov >= COVERAGE_THRESHOLD and len(burst_means) >= 3)
            print(f"\n  Target {COVERAGE_THRESHOLD:.0f}% (smoothed): "
                  f"{'✓ REACHED' if finished else f'✗  ({COVERAGE_THRESHOLD - sm_cov:.1f}% to go)'}")

        if REGRESSION_GUARD:
            bad_streak = bad_streak + 1 if score < best_score - REGRESSION_DROP else 0
            if (bad_streak >= REGRESSION_PATIENCE
                    and os.path.exists(BEST_MODEL_PATH + ".zip")
                    and os.path.exists(BEST_VNORM_PATH)):
                print(f"  ⟲ {bad_streak} bursts > {REGRESSION_DROP:.0f} pts below best "
                      f"— reloading best model + vnorm")
                train_env, model = _restore_best(train_env)
                bad_streak = 0

        if finished:
            break
        if max_hours and (time.time() - overall_start) / 3600 >= max_hours:
            print(f"\n  ⏱ Time budget of {max_hours} h used — stopping after burst {burst_num}.")
            break

    print(f"\n{'=' * 64}")
    print("  COMPLETE — final targets reached!" if finished
          else f"  SESSION ENDED — best score {best_score:.1f} at burst {best_burst}\n"
               f"  Run the same command again to continue from the latest model.")
    if LETHAL:
        print(f"  Curriculum now at lives = {lives} ({stage_bursts} bursts at this stage)")
    print("\n  Milestones:")
    for ms in sorted(MILESTONES):
        if ms in milestones_hit:
            b, sps = milestones_hit[ms]
            print(f"    {ms:5.0f}% → burst {b:3d} ({sps:,} steps)")
        else:
            print(f"    {ms:5.0f}% → not reached")
    print(f"\n  Total steps : {total_steps_done:,}")
    print(f"  Total time  : {(time.time() - overall_start) / 3600:.2f} h")
    print(f"  Model       : {MODEL_PATH}.zip")
    print(f"{'=' * 64}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true",
                    help="archive the old run and start from scratch")
    ap.add_argument("--restore-best", action="store_true",
                    help="copy the best checkpoint over final_model, then continue")
    ap.add_argument("--from-best", action="store_true",
                    help="resume from the best checkpoint instead of latest")
    ap.add_argument("--bursts", type=int, default=None,
                    help=f"max bursts this session (default {MAX_BURSTS})")
    ap.add_argument("--hours", type=float, default=None,
                    help="stop after the burst that crosses this many hours")
    ap.add_argument("--no-warm-start", action="store_true",
                    help="do not initialise from the previous phase's best model")
    ap.add_argument("--run-name", default=None,
                    help=f"run to train/resume (default '{RUN_NAME}'; 'phase4' = the original run)")
    ap.add_argument("--warm-from", default=None,
                    help="checkpoint (path without .zip) a NEW run starts from; default: auto")
    ap.add_argument("--start-lives", type=int, default=None,
                    help="initial lives per drone for a NEW run (must be in LIVES_SCHEDULE)")
    ap.add_argument("--keep-old", action="store_true",
                    help="do not auto-archive an incompatible stored run")
    args = ap.parse_args()
    if args.no_warm_start:
        WARM_START = False
    if args.run_name:
        _configure_paths(args.run_name)
    if args.warm_from:
        WARM_FROM = args.warm_from
    if args.start_lives is not None:
        if args.start_lives not in config.LIVES_SCHEDULE:
            ap.error(f"--start-lives must be one of {list(config.LIVES_SCHEDULE)}")
        START_LIVES = args.start_lives

    if args.fresh:
        _archive_old_run()
    elif args.restore_best:
        _restore_best_to_final()
    train_until_threshold(from_best=args.from_best, keep_old=args.keep_old,
                          max_bursts=args.bursts, max_hours=args.hours)