"""
evaluation/evaluate.py   (v2)

New in v2
─────────
• Collision graphs per episode:
    – collision STEPS per episode, stacked by kind (drone / bird / building)
    – collision EVENTS per episode (distinct new contacts)
    – severity buckets (Clean / Light / Heavy) instead of a binary
      "any hit = 100 %" flag
• Return breakdown: mean reward per component per episode, so you can see
  what is actually driving the return.
• Coverage-vs-time (mean ± σ) with P(collision) per step on a twin axis.
• Honest RTB: drones landed vs landed INSIDE the spawn zone.
• Altitude panel (mean altitude, % of drone-steps inside the scan band).
• Fixed: heatmap ticks/log scale, wrong "%/step" label, heatmap polluted by
  the post-reset frame, duplicated eval loops.

Output layout
─────────────
logs/results/final_eval/{csv,img}/…
logs/results/train_phase{N}/{csv,img}/…

CLI:  python -m evaluation.evaluate --phase 3 --episodes 50 --model models/final_model
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgrid
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.multi_uav_env import PHASE_ENVS, REWARD_COMPONENTS
from evaluation.replay import ReplayRecorder, write_manifest
from env.scenarios import (build_suite, sample_test_set, TYPE_NAMES,
                           TEST_SEED_BASE, VAL_SEED_BASE, describe)
import configs.config as config

# ── Style ──────────────────────────────────────────────────────────────────
RESEARCH_STYLE = {
    "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d", "axes.labelcolor": "#c9d1d9",
    "axes.titlecolor": "#f0f6fc", "axes.titlesize": 10.5, "axes.labelsize": 9,
    "axes.grid": True, "grid.color": "#21262d", "grid.linewidth": 0.6,
    "grid.alpha": 0.8, "xtick.color": "#8b949e", "ytick.color": "#8b949e",
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.facecolor": "#21262d", "legend.edgecolor": "#30363d",
    "legend.labelcolor": "#c9d1d9", "legend.fontsize": 7.5,
    "lines.linewidth": 1.6, "text.color": "#c9d1d9", "font.family": "DejaVu Sans",
}
C_GREEN, C_BLUE, C_ORANGE, C_RED = "#3fb950", "#58a6ff", "#f0883e", "#f85149"
C_PURPLE, C_YELLOW, C_TEAL = "#bc8cff", "#e3b341", "#39d353"
C_PINK, C_GREY = "#ff7b72", "#8b949e"

EPISODES            = 100
PHASE_EVAL_EPISODES = 20
EVAL_PHASE_IDX      = 3                       # final eval runs on Phase 4
MODEL_PATH          = os.path.join(config.MODEL_DIR, "final_model")
VNORM_PATH          = os.path.join(config.MODEL_DIR, "vecnormalize.pkl")
RESULTS_ROOT        = os.path.join("logs", "results")
FINAL_EVAL_DIR      = os.path.join(RESULTS_ROOT, "final_eval")

KIND_COLORS = (("drone", C_RED), ("bird", C_ORANGE), ("building", C_PURPLE))


# ── Small helpers ──────────────────────────────────────────────────────────

def _csv_dir(base):
    d = os.path.join(base, "csv"); os.makedirs(d, exist_ok=True); return d


def _img_dir(base):
    d = os.path.join(base, "img"); os.makedirs(d, exist_ok=True); return d


def unwrap_env(env):
    if hasattr(env, "unwrapped"):
        return env.unwrapped
    raw = env
    while hasattr(raw, "env"):
        raw = raw.env
    return raw


def _apply_research_style():
    plt.rcParams.update(RESEARCH_STYLE)


def _reset_style():
    plt.rcParams.update(plt.rcParamsDefault)


def _sparkline_band(ax, x, y, color, label):
    ax.plot(x, y, color=color, lw=1.8, label=label, zorder=3)
    w = max(len(y) // 8, 3)
    if len(y) >= w:
        rm = np.convolve(y, np.ones(w) / w, mode="valid")
        ax.plot(x[w - 1:], rm, color=color, lw=0.8, alpha=0.45,
                linestyle="--", zorder=2)


def _stat_annotation(ax, values, color, unit="", decimals=1):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return
    txt = (f"μ={v.mean():.{decimals}f}{unit}\nσ={v.std():.{decimals}f}{unit}"
           f"\nmed={np.median(v):.{decimals}f}{unit}")
    ax.text(0.97, 0.97, txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=7, color=color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#0d1117",
                      edgecolor=color, alpha=0.85), zorder=10)


def _heatmap_ax(ax, counts, title, cmap="plasma"):
    """counts = raw visit counts (50×50).  log1p scale, 97th-pct clip."""
    hm = np.log1p(np.asarray(counts, float))
    vmax = float(np.percentile(hm[hm > 0], 97)) if np.any(hm > 0) else 1.0
    im = ax.imshow(hm, origin="lower", cmap=cmap, aspect="auto", vmin=0, vmax=vmax)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("log(1 + visits)", fontsize=7, color=C_GREY)
    cb.ax.tick_params(colors=C_GREY, labelsize=6)
    ax.set_title(title, fontsize=10, color="#f0f6fc", pad=6)
    ticks = [0, 10, 20, 29, 39, 49]                 # bin i ↔ i/49·100 m
    labels = [0, 20, 40, 60, 80, 100]
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=7)
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("X (m)", fontsize=8); ax.set_ylabel("Y (m)", fontsize=8)
    ax.grid(False)


def _categorize_collision_severity(coll_steps, steps, heavy_frac=None):
    """
    Mutually-exclusive buckets (always sum to 1):
      Clean  — 0 collision steps
      Light  — some, but < heavy_frac of the episode's steps
      Heavy  — ≥ heavy_frac of the episode's steps had a hit
    """
    if heavy_frac is None:
        heavy_frac = float(getattr(config, "COLLISION_HEAVY_FRAC", 0.05))
    n = len(coll_steps)
    if n == 0:
        return 0.0, 0.0, 0.0
    clean = light = heavy = 0
    for c, s in zip(coll_steps, steps):
        if c <= 0:
            clean += 1
        elif c / max(s, 1) >= heavy_frac:
            heavy += 1
        else:
            light += 1
    return clean / n, light / n, heavy / n


def _render_env_map(raw_env, save_path: str, title: str = "Environment map"):
    _apply_research_style()
    ms  = raw_env.map_size

    fig, axes = plt.subplots(1, 2, figsize=(14, 7),
                              gridspec_kw={"width_ratios": [2, 1]})
    fig.patch.set_facecolor("#0d1117")

    ax  = axes[0]
    ax2 = axes[1]

    ax.set_facecolor("#0d1117")
    ax.set_xlim(0, ms); ax.set_ylim(0, ms)
    ax.set_xlabel("X (m)", fontsize=9, color=C_GREY)
    ax.set_ylabel("Y (m)", fontsize=9, color=C_GREY)
    ax.set_title(title, fontsize=11, color="#f0f6fc", pad=8)
    ax.set_aspect("equal")
    ax.tick_params(colors=C_GREY, labelsize=7)
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")

    for v in np.linspace(0, ms, 11):
        ax.axhline(v, color="#21262d", lw=0.4, alpha=0.8)
        ax.axvline(v, color="#21262d", lw=0.4, alpha=0.8)

    x0, y0, x1, y1 = raw_env._spawn_zone
    spawn_rect = mpatches.FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0.5", linewidth=1.8,
        edgecolor=C_GREEN, facecolor="#3fb95015")
    ax.add_patch(spawn_rect)
    ax.text((x0+x1)/2, (y0+y1)/2, "SPAWN\nZONE", ha="center", va="center",
            color=C_GREEN, fontsize=8, fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2, foreground="#0d1117")])

    CLASS_COLORS  = {0: "#5a8a6a", 1: "#5a7a9a", 2: "#9a5a7a"}
    CLASS_LABELS  = {0: "Small (4–8 m)", 1: "Medium (9–18 m)", 2: "Large (19–30 m)"}
    class_ids     = getattr(raw_env, "building_class_ids", None)
    class_counts  = {0: 0, 1: 0, 2: 0}

    for k, b in enumerate(raw_env.buildings):
        bx, by, bh, bw = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        cid = int(class_ids[k]) if class_ids is not None and k < len(class_ids) else 0
        class_counts[cid] = class_counts.get(cid, 0) + 1
        fc  = CLASS_COLORS.get(cid, "#5a5a7a")
        rect = mpatches.Rectangle((bx, by), bw, bw,
                                   linewidth=0.6, edgecolor="#aaaaaa55",
                                   facecolor=fc, alpha=0.75)
        ax.add_patch(rect)
        if bh >= 9.0:
            ax.text(bx + bw/2, by + bw/2, f"{bh:.0f}m",
                    ha="center", va="center", color="white",
                    fontsize=5.5, alpha=0.9,
                    path_effects=[pe.withStroke(linewidth=1, foreground="#000000")])

    if len(raw_env.checkpoints):
        ax.scatter(raw_env.checkpoints[:, 0], raw_env.checkpoints[:, 1],
                   s=30, c=C_YELLOW, marker="D", zorder=4,
                   edgecolors="#c9a22788", linewidths=0.7, label="Checkpoints")

    ax.scatter(raw_env.coverage_points[:, 0],
               raw_env.coverage_points[:, 1],
               s=2, c="#ffffff08", zorder=1)

    if len(raw_env.victims):
        ax.scatter(raw_env.victims[:, 0], raw_env.victims[:, 1],
                   s=55, c=C_RED, marker="*", zorder=5,
                   edgecolors="#cc000066", linewidths=0.7, label="Victims")

    legend_handles = [
        mpatches.Patch(color=CLASS_COLORS[0], alpha=0.8, label=CLASS_LABELS[0]),
        mpatches.Patch(color=CLASS_COLORS[1], alpha=0.8, label=CLASS_LABELS[1]),
        mpatches.Patch(color=CLASS_COLORS[2], alpha=0.8, label=CLASS_LABELS[2]),
        mpatches.Patch(color=C_GREEN, alpha=0.4, label="Spawn zone"),
    ]
    if len(raw_env.checkpoints):
        legend_handles.append(
            mpatches.Patch(color=C_YELLOW, label=f"Checkpoints ({len(raw_env.checkpoints)})"))
    if len(raw_env.victims):
        legend_handles.append(
            mpatches.Patch(color=C_RED, label=f"Victims ({len(raw_env.victims)})"))
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7,
              framealpha=0.85, facecolor="#161b22", edgecolor="#30363d")

    # ── Stats panel
    ax2.set_facecolor("#0d1117")
    ax2.axis("off")
    for sp in ax2.spines.values(): sp.set_visible(False)

    bldgs = raw_env.buildings
    n_bldg = len(bldgs)
    stats_rows = [
        ("Map size",       f"{ms} × {ms} m"),
        ("Spawn zone",     f"{int(x1-x0)} × {int(y1-y0)} m"),
        ("Total buildings",f"{n_bldg}"),
        ("  Small",        f"{class_counts.get(0,0)}"),
        ("  Medium",       f"{class_counts.get(1,0)}"),
        ("  Large",        f"{class_counts.get(2,0)}"),
        ("Checkpoints",    f"{len(raw_env.checkpoints)}"),
        ("Victims",        f"{len(raw_env.victims)}"),
        ("Num drones",     f"{raw_env.num_drones}"),
        ("Max steps",      f"{raw_env.max_steps}"),
        ("Grid size",      f"{config.GRID_SIZE}×{config.GRID_SIZE}"),
        ("Coverage cells", f"{config.GRID_SIZE**2}"),
    ]
    if n_bldg > 0:
        heights = bldgs[:, 2]
        stats_rows += [
            ("Max bldg height", f"{float(np.max(heights)):.1f} m"),
            ("Kill altitude",   f"{float(getattr(raw_env, '_kill_alt', 0)):.1f} m"),
        ]

    y_pos = 0.97
    ax2.text(0.05, y_pos, "Environment Parameters",
             transform=ax2.transAxes, fontsize=10, fontweight="bold",
             color="#f0f6fc", va="top")
    y_pos -= 0.07
    for key, val in stats_rows:
        col_k = C_GREY if key.startswith(" ") else "#c9d1d9"
        ax2.text(0.08, y_pos, key, transform=ax2.transAxes,
                 fontsize=8, color=col_k, va="top")
        ax2.text(0.65, y_pos, val, transform=ax2.transAxes,
                 fontsize=8, color=C_BLUE, va="top", fontweight="bold")
        y_pos -= 0.065

    if n_bldg > 0:
        y_pos -= 0.03
        ax2.text(0.05, y_pos, "Building height distribution",
                 transform=ax2.transAxes, fontsize=8.5,
                 color="#c9d1d9", va="top", fontweight="bold")
        y_pos -= 0.04
        inset = ax2.inset_axes([0.05, y_pos - 0.22, 0.9, 0.22])
        inset.set_facecolor("#161b22")
        for sp in inset.spines.values(): sp.set_edgecolor("#30363d")
        inset.hist(heights, bins=12, color=C_BLUE, edgecolor="#0d1117",
                   alpha=0.85, linewidth=0.5)
        inset.set_xlabel("Height (m)", fontsize=6.5, color=C_GREY)
        inset.set_ylabel("Count", fontsize=6.5, color=C_GREY)
        inset.tick_params(colors=C_GREY, labelsize=6)
        inset.axvline(float(np.mean(heights)), color=C_ORANGE, lw=1.2,
                      linestyle="--", label=f"μ={float(np.mean(heights)):.1f}m")
        inset.legend(fontsize=6, frameon=False, labelcolor=C_ORANGE)
        inset.grid(color="#21262d", lw=0.4, alpha=0.8)

    fig.suptitle(f"Environment Map  ·  {title}",
                 fontsize=12, fontweight="bold", color="#f0f6fc", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight",
                facecolor="#0d1117", format="jpeg")
    plt.close()
    _reset_style()



# ── Core evaluation loop ──────────────────────────────────────────────────

def _run_eval_loop(model, vec_env, episodes, label="", scenarios=None, replay_dir=None):
    """scenarios: optional list of env/scenarios.py dicts.  If given, episode i is
    built from scenarios[i] (fixed seed + type) and `episodes` is ignored."""
    if scenarios is not None:
        episodes = len(scenarios)
    heatmap = np.zeros((50, 50))
    keys = ("coverage_list", "reward_list", "step_list", "rescued_list",
            "collision_steps_ep", "collision_events_ep", "collision_episode",
            "height_killed_ep", "lazy_killed_ep", "fuel_remaining_ep",
            "landed_count_ep", "landed_in_zone_ep", "alt_mean_ep", "alt_band_ep",
            "drones_dead_ep")
    R = {k: [] for k in keys}
    R["coll_kind_steps"] = {k: [] for k, _ in KIND_COLORS}
    R["comp_ep"], R["cov_traj"], R["coll_traj"] = [], [], []
    R["scenario_type"], R["scenario_seed"] = [], []
    last_raw_env = None
    rec = ReplayRecorder() if replay_dir else None   # NB: VecEnv auto-resets on done, so the
                                                     # terminal frame is not captured

    for ep in range(episodes):
        if scenarios is not None:
            unwrap_env(vec_env.envs[0]).set_next_scenario(scenarios[ep])
            R["scenario_type"].append(scenarios[ep]["type"])
            R["scenario_seed"].append(scenarios[ep]["seed"])
        else:
            R["scenario_type"].append("random")
            R["scenario_seed"].append(-1)
        obs = vec_env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        if rec is not None:
            rec.start(unwrap_env(vec_env.envs[0]), dict(label=label, episode=ep + 1))
        done = False
        ep_ret, steps, peak_rescued, max_cov = 0.0, 0, 0, 0.0
        coll_steps = coll_events = 0
        kind_steps = {k: 0 for k, _ in KIND_COLORS}
        hk = lazy = False
        alt_sum = band_sum = 0.0
        alt_n = 0
        comp_sum = {}
        cov_t, coll_t = [], []
        last_fuel = last_landed = last_zone = None
        dead_n = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(action)
            done = bool(dones[0])
            ep_ret += float(rewards[0]); steps += 1

            info = infos[0] if isinstance(infos, (list, tuple)) else infos
            raw_env = unwrap_env(vec_env.envs[0]); last_raw_env = raw_env
            if rec is not None and not done:
                rec.step(raw_env, info, float(rewards[0]))

            cov = float(info.get("coverage", raw_env.get_coverage_percent()))
            max_cov = max(max_cov, cov)
            cov_t.append(cov)
            peak_rescued = max(peak_rescued, int(info.get("rescued", 0)))

            hit = bool(info.get("collision_hit", False))
            coll_t.append(1.0 if hit else 0.0)
            if hit:
                coll_steps += 1
            coll_events += int(info.get("collision_onsets", 0))
            for k, _ in KIND_COLORS:
                kind_steps[k] += int(bool(info.get(f"hit_{k}", False)))
            hk   = hk   or bool(info.get("height_killed", False))
            lazy = lazy or bool(info.get("lazy_killed", False))

            a_m = info.get("alt_mean", float("nan"))
            if a_m == a_m:                                  # not NaN
                alt_sum += a_m; band_sum += float(info.get("alt_in_band", 0.0))
                alt_n += 1
            for k, v in (info.get("r_comp") or {}).items():
                comp_sum[k] = comp_sum.get(k, 0.0) + float(v)

            dead_n = int(info.get("drones_dead", dead_n))
            if "fuel" in info:   last_fuel   = np.array(info["fuel"])
            if "landed" in info: last_landed = np.array(info["landed"])
            if "landed_in_zone" in info:
                last_zone = np.array(info["landed_in_zone"])

            # heatmap: skip the terminal step (env was auto-reset → spawn pos)
            if not done:
                xi = np.clip((raw_env.positions[:, 0] / raw_env.map_size * 49)
                             .astype(int), 0, 49)
                yi = np.clip((raw_env.positions[:, 1] / raw_env.map_size * 49)
                             .astype(int), 0, 49)
                np.add.at(heatmap, (yi, xi), 1)

        if rec is not None:
            rec.save(os.path.join(
                replay_dir, f"ep{ep + 1:03d}_{R['scenario_type'][-1]}_{R['scenario_seed'][-1]}.npz"),
                extra={"final_coverage": round(max_cov, 2), "return": round(ep_ret, 1),
                       "steps": steps, "drones_dead": dead_n, "collision_steps": coll_steps,
                       "collision_events": coll_events, "rescued": peak_rescued,
                       "terminal_frame_missing": True})
        R["coverage_list"].append(max_cov)
        R["reward_list"].append(ep_ret)
        R["step_list"].append(steps)
        R["rescued_list"].append(peak_rescued)
        R["collision_steps_ep"].append(coll_steps)
        R["collision_events_ep"].append(coll_events)
        R["collision_episode"].append(1.0 if coll_steps > 0 else 0.0)
        for k, _ in KIND_COLORS:
            R["coll_kind_steps"][k].append(kind_steps[k])
        R["height_killed_ep"].append(1.0 if hk else 0.0)
        R["lazy_killed_ep"].append(1.0 if lazy else 0.0)
        R["fuel_remaining_ep"].append(
            last_fuel if last_fuel is not None else np.zeros(config.NUM_DRONES))
        R["landed_count_ep"].append(
            int(last_landed.sum()) if last_landed is not None else 0)
        R["landed_in_zone_ep"].append(
            int(last_zone.sum()) if last_zone is not None else 0)
        R["drones_dead_ep"].append(dead_n)
        R["alt_mean_ep"].append(alt_sum / alt_n if alt_n else float("nan"))
        R["alt_band_ep"].append(band_sum / alt_n if alt_n else float("nan"))
        R["comp_ep"].append(comp_sum)
        R["cov_traj"].append(cov_t)
        R["coll_traj"].append(coll_t)

        tag = f"[{label}] " if label else ""
        if scenarios is not None:
            tag += f"{scenarios[ep]['type']}#{scenarios[ep]['seed']} "
        print(f"  {tag}Ep {ep + 1:03d}  cov={max_cov:5.1f}%  ret={ep_ret:9.0f}  "
              f"steps={steps}  coll_steps={coll_steps:4d} (events {coll_events:3d}; "
              f"d/b/B {kind_steps['drone']}/{kind_steps['bird']}/{kind_steps['building']})  "
              f"hk={'Y' if hk else 'N'} lazy={'Y' if lazy else 'N'}  "
              f"landed={R['landed_count_ep'][-1]} (in zone {R['landed_in_zone_ep'][-1]})  "
              f"dead={dead_n}")

    R["heatmap"] = heatmap
    R["last_raw_env"] = last_raw_env
    R["comp_keys"] = list(REWARD_COMPONENTS)
    R["has_victims"] = bool(last_raw_env is not None
                            and getattr(last_raw_env, "num_victims", 0) > 0)
    return R


def results_to_dataframe(results):
    n = len(results["reward_list"])
    steps = results["step_list"]
    data = {
        "scenario_type":         results["scenario_type"],
        "scenario_seed":         results["scenario_seed"],
        "coverage_peak":         results["coverage_list"],
        "reward":                results["reward_list"],
        "steps":                 steps,
        "rescued_victims":       results["rescued_list"],
        "collision_steps":       results["collision_steps_ep"],
        "collision_events":      results["collision_events_ep"],
        "coll_steps_drone":      results["coll_kind_steps"]["drone"],
        "coll_steps_bird":       results["coll_kind_steps"]["bird"],
        "coll_steps_building":   results["coll_kind_steps"]["building"],
        "collision_step_frac":   [c / max(s, 1) for c, s in
                                  zip(results["collision_steps_ep"], steps)],
        "height_killed":         results["height_killed_ep"],
        "lazy_killed":           results["lazy_killed_ep"],
        "drones_landed":         results["landed_count_ep"],
        "drones_landed_in_zone": results["landed_in_zone_ep"],
        "drones_dead":           results["drones_dead_ep"],
        "mean_alt":              results["alt_mean_ep"],
        "alt_in_band_frac":      results["alt_band_ep"],
        "mean_fuel_pct": [float(np.mean(f)) / config.DRONE_FUEL_STEPS * 100
                          for f in results["fuel_remaining_ep"]],
    }
    for k in results.get("comp_keys", []):
        data[f"r_{k}"] = [c.get(k, 0.0) for c in results["comp_ep"]]
    assert all(len(v) == n for v in data.values())
    return pd.DataFrame(data)


# ── Per-scenario-type summary ─────────────────────────────────────────────

def per_type_table(results):
    """One row per scenario type (mean over its seeds) + an ALL row."""
    df = results_to_dataframe(results)
    g = df.groupby("scenario_type", sort=False)
    out = pd.DataFrame({
        "n_seeds":           g.size(),
        "coverage_mean":     g["coverage_peak"].mean(),
        "coverage_std":      g["coverage_peak"].std(ddof=0),
        "coverage_min":      g["coverage_peak"].min(),
        "deaths_mean":       g["drones_dead"].mean(),
        "zero_death_rate":   g["drones_dead"].apply(lambda x: float((x == 0).mean())),
        "coll_events_mean":  g["collision_events"].mean(),
        "return_mean":       g["reward"].mean(),
        "steps_mean":        g["steps"].mean(),
    })
    allrow = pd.DataFrame([{
        "n_seeds": len(df), "coverage_mean": df["coverage_peak"].mean(),
        "coverage_std": df["coverage_peak"].std(ddof=0),
        "coverage_min": df["coverage_peak"].min(),
        "deaths_mean": df["drones_dead"].mean(),
        "zero_death_rate": float((df["drones_dead"] == 0).mean()),
        "coll_events_mean": df["collision_events"].mean(),
        "return_mean": df["reward"].mean(), "steps_mean": df["steps"].mean()}],
        index=["ALL"])
    return pd.concat([out, allrow]).round(3)


def save_scenario_figure(results, img_dir, filename="scenario_breakdown.jpeg",
                         title_suffix=""):
    """Coverage and deaths per scenario type; bars = mean, dots = individual seeds."""
    _apply_research_style()
    os.makedirs(img_dir, exist_ok=True)
    df = results_to_dataframe(results)
    types = [t for t in TYPE_NAMES if t in set(df["scenario_type"])] or ["random"]
    fig, axes = plt.subplots(1, 2, figsize=(18, 5.6))
    fig.patch.set_facecolor("#0d1117")
    nd = config.NUM_DRONES
    for ax, col, title, color, ylim in (
            (axes[0], "coverage_peak", "Coverage % by scenario type", C_GREEN, (0, 105)),
            (axes[1], "drones_dead", f"Drones dead per episode (of {nd})", C_RED, (0, nd + 0.5))):
        means = [df.loc[df.scenario_type == t, col].mean() for t in types]
        ax.bar(range(len(types)), means, color=color, alpha=0.55, width=0.7, edgecolor="none")
        for i, t in enumerate(types):
            v = df.loc[df.scenario_type == t, col].values
            ax.scatter(np.full(len(v), i) + np.linspace(-0.18, 0.18, len(v)), v,
                       s=22, color="#f0f6fc", zorder=3, edgecolors="none")
            ax.text(i, ylim[1] * 0.02, f"{means[i]:.1f}", ha="center", va="bottom",
                    fontsize=8, color="#f0f6fc", fontweight="bold")
        ax.set_xticks(range(len(types)))
        ax.set_xticklabels(types, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(*ylim); ax.set_title(title)
    axes[0].axhline(95, color=C_ORANGE, ls="--", lw=1.2, label="95% target")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Per-scenario breakdown{title_suffix}", fontsize=13,
                 fontweight="bold", color="#f0f6fc")
    plt.tight_layout()
    out = os.path.join(img_dir, filename)
    plt.savefig(out, dpi=140, format="jpeg", bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig); _reset_style()
    return out


# ── Research figure (12 panels) ───────────────────────────────────────────

def save_research_figure(results, img_dir, filename="evaluation_research.jpeg",
                         title_suffix=""):
    _apply_research_style()
    os.makedirs(img_dir, exist_ok=True)

    cov   = np.array(results["coverage_list"], float)
    rew   = np.array(results["reward_list"], float)
    steps = np.array(results["step_list"], float)
    cs    = np.array(results["collision_steps_ep"], float)
    ce    = np.array(results["collision_events_ep"], float)
    kinds = results["coll_kind_steps"]
    n     = len(rew)
    ep    = np.arange(1, n + 1)

    fig = plt.figure(figsize=(22, 21))
    gs = mgrid.GridSpec(4, 3, figure=fig, hspace=0.52, wspace=0.34,
                        left=0.06, right=0.97, top=0.945, bottom=0.04)
    axs = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(4)]

    # ── Row 0: heatmap | return | coverage ──────────────────────────────
    _heatmap_ax(axs[0][0], results["heatmap"], "UAV visit density (log)")

    _sparkline_band(axs[0][1], ep, rew, C_BLUE, "Return")
    axs[0][1].axhline(rew.mean(), color=C_BLUE, ls=":", lw=1, alpha=0.6)
    axs[0][1].set_title(f"Episode return  (μ={rew.mean():,.0f}  σ={rew.std():,.0f})")
    axs[0][1].set_xlabel("Episode"); axs[0][1].set_ylabel("Return")
    _stat_annotation(axs[0][1], rew, C_BLUE, "", 0)

    _sparkline_band(axs[0][2], ep, cov, C_GREEN, "Coverage %")
    axs[0][2].axhline(95, color=C_ORANGE, ls="--", lw=1.2, alpha=0.85, label="95% target")
    axs[0][2].axhline(cov.mean(), color=C_GREEN, ls=":", lw=1, alpha=0.6)
    axs[0][2].set_ylim(0, 105)
    axs[0][2].set_title(f"Coverage  (μ={cov.mean():.1f}%  σ={cov.std():.1f}%)")
    axs[0][2].set_xlabel("Episode"); axs[0][2].set_ylabel("Coverage %")
    axs[0][2].legend(fontsize=7)
    _stat_annotation(axs[0][2], cov, C_GREEN, "%", 1)

    # ── Row 1: collisions ───────────────────────────────────────────────
    ax = axs[1][0]
    bottom = np.zeros(n)
    for name, col in KIND_COLORS:
        vals = np.array(kinds.get(name, [0] * n), float)
        ax.bar(ep, vals, bottom=bottom, color=col, alpha=0.85, width=0.8,
               edgecolor="none", label=name)
        bottom += vals
    ax.plot(ep, cs, color="#f0f6fc", lw=1.0, marker="o", ms=2.5,
            label="steps with ≥1 hit")
    ax.axhline(cs.mean(), color=C_YELLOW, ls="--", lw=1.3,
               label=f"mean {cs.mean():.0f}")
    pct = 100 * cs.sum() / max(steps.sum(), 1)
    ax.set_title(f"Collision STEPS per episode  (μ={cs.mean():.0f} = {pct:.1f}% of steps)\n"
                 f"stacked by kind — kinds can overlap within a step", fontsize=9.5)
    ax.set_xlabel("Episode"); ax.set_ylabel("Steps in collision")
    ax.legend(fontsize=7, ncol=2)

    ax = axs[1][1]
    ax.bar(ep, ce, color=C_PINK, alpha=0.85, width=0.8, edgecolor="none")
    ax.axhline(ce.mean(), color=C_YELLOW, ls="--", lw=1.3,
               label=f"mean {ce.mean():.1f}")
    ax.set_title(f"Collision EVENTS per episode  (new contacts, μ={ce.mean():.1f})\n"
                 f"a long scrape = 1 event but many steps", fontsize=9.5)
    ax.set_xlabel("Episode"); ax.set_ylabel("New contacts")
    ax.legend(fontsize=7)
    _stat_annotation(ax, ce, C_PINK, "", 1)

    clean, light, heavy = _categorize_collision_severity(cs, steps)
    heavy_frac = float(getattr(config, "COLLISION_HEAVY_FRAC", 0.05))
    vals = [clean, light, heavy]
    bars = axs[1][2].bar(["Clean", "Light", "Heavy"], vals,
                         color=[C_GREEN, C_ORANGE, C_RED], alpha=0.85,
                         edgecolor="none", width=0.6)
    for b, v in zip(bars, vals):
        axs[1][2].text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v * 100:.1f}%",
                       ha="center", va="bottom", fontsize=9, color="#f0f6fc",
                       fontweight="bold")
    axs[1][2].set_ylim(0, 1.2)
    hk_rate = float(np.mean(results["height_killed_ep"]))
    lz_rate = float(np.mean(results["lazy_killed_ep"]))
    axs[1][2].set_title(
        f"Collision severity  (heavy ≥ {heavy_frac * 100:.0f}% of steps)\n"
        f"episodes with height-event={hk_rate * 100:.0f}%  lazy-event={lz_rate * 100:.0f}%",
        fontsize=9.5)
    axs[1][2].set_ylabel("Fraction of episodes")

    # ── Row 2: reward breakdown | coverage vs time | coll vs coverage ───
    ax = axs[2][0]
    comps = results.get("comp_ep") or []
    if comps:
        means = {k: float(np.mean([c.get(k, 0.0) for c in comps]))
                 for k in results["comp_keys"]}
        items = sorted([(k, v) for k, v in means.items() if abs(v) > 0.05],
                       key=lambda kv: kv[1])
        if items:
            names = [k for k, _ in items]
            vals = [v for _, v in items]
            ax.barh(names, vals, alpha=0.85, height=0.7,
                    color=[C_GREEN if v >= 0 else C_RED for v in vals])
            for yy, v in enumerate(vals):
                ax.text(v, yy, f" {v:,.0f} " , va="center", fontsize=7,
                        ha="left" if v >= 0 else "right", color="#f0f6fc")
            ax.axvline(0, color=C_GREY, lw=0.8)
            ax.margins(x=0.28)
        ax.set_title(f"Return breakdown — mean per episode (Σ = {sum(means.values()):,.0f})",
                     fontsize=9.5)
        ax.set_xlabel("Reward per episode")
    else:
        ax.axis("off")

    ax = axs[2][1]
    L = max(len(c) for c in results["cov_traj"])
    C = np.full((n, L), np.nan); H = np.full((n, L), np.nan)
    for i, (a, b) in enumerate(zip(results["cov_traj"], results["coll_traj"])):
        C[i, :len(a)] = a; H[i, :len(b)] = b
    x = np.arange(1, L + 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mu, sd = np.nanmean(C, axis=0), np.nanstd(C, axis=0)
        rate = np.nanmean(H, axis=0)
    ax.plot(x, mu, color=C_GREEN, lw=1.8, label="coverage μ")
    ax.fill_between(x, mu - sd, mu + sd, color=C_GREEN, alpha=0.15)
    ax.set_ylim(0, 105); ax.set_xlabel("Step"); ax.set_ylabel("Coverage %", color=C_GREEN)
    w = min(25, max(L // 10, 1))
    smooth = np.convolve(rate, np.ones(w) / w, mode="same")
    ax2 = ax.twinx()
    ax2.plot(x, smooth, color=C_RED, lw=1.3, alpha=0.9, label="P(collision at step)")
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("P(collision)", color=C_RED)
    ax2.grid(False); ax2.tick_params(axis="y", colors=C_RED)
    ax.set_title("Coverage and collision probability over time", fontsize=9.5)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="center right")

    ax = axs[2][2]
    sc = ax.scatter(cs, cov, c=ep, cmap="plasma", s=35, alpha=0.85,
                    edgecolors="none", zorder=3)
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Episode", fontsize=7, color=C_GREY)
    cb.ax.tick_params(colors=C_GREY, labelsize=6)
    if n > 2 and np.std(cs) > 0:
        z = np.polyfit(cs, cov, 1)
        px = np.linspace(cs.min(), cs.max(), 50)
        ax.plot(px, np.polyval(z, px), color=C_ORANGE, ls="--", lw=1.4,
                label=f"trend ({z[0]:+.3f} %/step)")
        ax.legend(fontsize=7)
    ax.set_title("Coverage vs collision steps  (do collisions cost coverage?)", fontsize=9.5)
    ax.set_xlabel("Collision steps in episode"); ax.set_ylabel("Coverage %")

    # ── Row 3: victims|altitude | landing | CDF ─────────────────────────
    ax = axs[3][0]
    if results.get("has_victims"):
        resc = np.array(results["rescued_list"], float)
        mv = int(getattr(config, "NUM_VICTIMS", 0))
        ax.bar(ep, resc, color=[C_TEAL if r >= mv else C_BLUE for r in resc],
               alpha=0.85, width=0.8, edgecolor="none")
        ax.axhline(mv, color=C_ORANGE, ls=":", lw=1.2, label=f"total {mv}")
        ax.axhline(resc.mean(), color=C_TEAL, ls="--", lw=1.3,
                   label=f"mean {resc.mean():.1f}")
        ax.set_ylim(0, mv + 1.5); ax.legend(fontsize=7)
        ax.set_title(f"Victims rescued  (μ={resc.mean():.1f}/{mv})  ·  line = mean altitude")
        ax.set_xlabel("Episode"); ax.set_ylabel("Rescued")
        alt = np.array(results["alt_mean_ep"], float)
        axa = ax.twinx(); axa.grid(False)
        axa.plot(ep, alt, color=C_YELLOW, lw=1.4, marker="o", ms=2.5, label="mean altitude")
        axa.axhspan(config.OPTIMAL_SCAN_HEIGHT - config.SCAN_HEIGHT_TOL,
                    config.OPTIMAL_SCAN_HEIGHT + config.SCAN_HEIGHT_TOL,
                    color=C_YELLOW, alpha=0.08)
        axa.set_ylim(0, config.MAX_HEIGHT)
        axa.set_ylabel("Altitude (m)", color=C_YELLOW)
        axa.tick_params(axis="y", colors=C_YELLOW)
    else:
        alt  = np.array(results["alt_mean_ep"], float)
        band = np.array(results["alt_band_ep"], float)
        ax.plot(ep, alt, color=C_BLUE, marker="o", ms=3)
        ax.axhspan(config.OPTIMAL_SCAN_HEIGHT - config.SCAN_HEIGHT_TOL,
                   config.OPTIMAL_SCAN_HEIGHT + config.SCAN_HEIGHT_TOL,
                   color=C_GREEN, alpha=0.12, label="optimal scan band")
        ax.set_ylim(0, config.MAX_HEIGHT)
        ax.legend(fontsize=7)
        ax.set_title(f"Mean drone altitude  (in band {np.nanmean(band) * 100:.0f}% of drone-steps)")
        ax.set_xlabel("Episode"); ax.set_ylabel("Altitude (m)")

    ax = axs[3][1]
    land = np.array(results["landed_count_ep"], float)
    zone = np.array(results["landed_in_zone_ep"], float)
    dead = np.array(results["drones_dead_ep"], float)
    other = np.maximum(land - zone, 0)
    ax.bar(ep, zone, color=C_TEAL, alpha=0.95, width=0.8, edgecolor="none",
           label="landed inside spawn zone")
    ax.bar(ep, other, bottom=zone, color=C_YELLOW, alpha=0.55, width=0.8,
           edgecolor="none", label="flagged landed elsewhere")
    ax.bar(ep, dead, bottom=zone + other, color=C_RED, alpha=0.9, width=0.8,
           edgecolor="none", label="dead (collision)")
    ax.axhline(config.NUM_DRONES, color=C_GREEN, ls=":", lw=1.2)
    ax.set_ylim(0, config.NUM_DRONES + 1.5)
    ax.set_title(f"Drone fate: zone {zone.mean():.1f} · elsewhere {other.mean():.1f} · "
                 f"dead {dead.mean():.2f} of {config.NUM_DRONES}  ·  episode μ={steps.mean():.0f} steps",
                 fontsize=9.5)
    ax.set_xlabel("Episode"); ax.set_ylabel("Drones"); ax.legend(fontsize=7, ncol=3)

    ax = axs[3][2]
    s_cov = np.sort(cov); cdf = np.arange(1, n + 1) / n
    ax.plot(s_cov, cdf, color=C_GREEN, lw=2.0)
    ax.fill_between(s_cov, cdf, alpha=0.12, color=C_GREEN)
    ax.axvline(95, color=C_ORANGE, ls="--", lw=1.2, alpha=0.85, label="95% target")
    ax.axvline(80, color=C_YELLOW, ls=":", lw=1.2, alpha=0.85, label="80%")
    ax.set_xlim(0, 105); ax.set_ylim(0, 1.05)
    ax.set_title(f"Coverage CDF  ({np.mean(cov >= 80) * 100:.0f}% of episodes ≥ 80%, "
                 f"{np.mean(cov >= 95) * 100:.0f}% ≥ 95%)", fontsize=9.5)
    ax.set_xlabel("Coverage %"); ax.set_ylabel("Cumulative probability")
    ax.legend(fontsize=7)

    fig.suptitle(f"Evaluation — Research Metrics{title_suffix}", fontsize=14,
                 fontweight="bold", color="#f0f6fc", y=0.975)
    out = os.path.join(img_dir, filename)
    plt.savefig(out, dpi=140, format="jpeg", bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    _reset_style()
    return out


def _save_compact_summary(results, path):
    _apply_research_style()
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle("Evaluation Summary", fontsize=13, fontweight="bold", color="#f0f6fc")
    _heatmap_ax(ax[0, 0], results["heatmap"], "Visit heatmap (log)")
    cov = results["coverage_list"]
    ax[0, 1].plot(cov, "s-", color=C_GREEN, ms=4)
    ax[0, 1].axhline(np.mean(cov), color=C_GREEN, ls=":", alpha=0.7)
    ax[0, 1].axhline(95, color=C_ORANGE, ls="--", alpha=0.8)
    ax[0, 1].set_ylim(0, 105); ax[0, 1].set_title(f"Coverage  μ={np.mean(cov):.1f}%")
    ax[0, 1].set_xlabel("Episode")
    cs = np.array(results["collision_steps_ep"])
    ax[1, 0].bar(np.arange(1, len(cs) + 1), cs, color=C_RED, alpha=0.85)
    ax[1, 0].axhline(cs.mean(), color=C_YELLOW, ls="--")
    ax[1, 0].set_title(f"Collision steps / episode  μ={cs.mean():.0f}")
    ax[1, 0].set_xlabel("Episode")
    rew = results["reward_list"]
    ax[1, 1].plot(rew, "o-", color=C_BLUE, ms=4)
    ax[1, 1].axhline(np.mean(rew), color=C_BLUE, ls=":", alpha=0.7)
    ax[1, 1].set_title(f"Return  μ={np.mean(rew):,.0f}"); ax[1, 1].set_xlabel("Episode")
    plt.tight_layout()
    plt.savefig(path, dpi=150, format="jpeg", bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    _reset_style()


def _print_summary(results):
    cs = np.array(results["collision_steps_ep"], float)
    st = np.array(results["step_list"], float)
    clean, light, heavy = _categorize_collision_severity(cs, st)
    print(f"  Mean coverage        : {np.mean(results['coverage_list']):.1f}%")
    print(f"  Mean return          : {np.mean(results['reward_list']):,.0f}")
    print(f"  Collision steps/ep   : {cs.mean():.1f}  ({100 * cs.sum() / max(st.sum(), 1):.1f}% of steps)")
    print(f"  Collision events/ep  : {np.mean(results['collision_events_ep']):.1f}")
    for k, _ in KIND_COLORS:
        print(f"    {k:<9} steps/ep : {np.mean(results['coll_kind_steps'][k]):.1f}")
    print(f"  Severity             : clean={clean * 100:.0f}%  light={light * 100:.0f}%  heavy={heavy * 100:.0f}%")
    print(f"  Mean episode length  : {st.mean():.0f}")
    print(f"  Drones dead / ep     : {np.mean(results['drones_dead_ep']):.2f} of {config.NUM_DRONES}"
          f"  (episodes with 0 deaths: {np.mean(np.array(results['drones_dead_ep']) == 0) * 100:.0f}%)")
    print(f"  Landed / in zone     : {np.mean(results['landed_count_ep']):.1f} / "
          f"{np.mean(results['landed_in_zone_ep']):.1f} of {config.NUM_DRONES}")


# ── Public entry points ───────────────────────────────────────────────────

def _make_eval_env(env_class, vnorm_path):
    vec_env = DummyVecEnv([env_class])
    vec_env = VecNormalize.load(vnorm_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    return vec_env


def evaluate(live=False, output_dir=None, episodes=None, phase_idx=EVAL_PHASE_IDX,
             model_path=MODEL_PATH, vnorm_path=VNORM_PATH,
             suite="full", test_rng=None, save_replays=False):
    """Final evaluation.  (`live` is accepted for backwards compatibility and ignored.)

    suite = "full"   → the 50 fixed scenarios (10 types × 5 seeds)      [paper numbers]
            "test10" → 1 random seed from each type (10 episodes); the draw is
                       reproducible via test_rng and is printed
            "random" → old behaviour: `episodes` unseeded random maps
    """
    scenarios = None
    if suite == "full":
        scenarios = build_suite(TEST_SEED_BASE)
    elif suite == "test10":
        scenarios = sample_test_set(test_rng, TEST_SEED_BASE)
        print(f"  test10 draw (test_rng={test_rng}): {describe(scenarios)}")
    episodes = len(scenarios) if scenarios else (episodes or EPISODES)
    output_dir = output_dir or FINAL_EVAL_DIR
    csv_d, img_d = _csv_dir(output_dir), _img_dir(output_dir)

    if not os.path.exists(model_path + ".zip"):
        print(f"Missing {model_path}.zip — train first."); return
    if not os.path.exists(vnorm_path):
        print(f"Missing {vnorm_path} — train first."); return

    env_class = PHASE_ENVS[phase_idx]
    vec_env = _make_eval_env(env_class, vnorm_path)
    model = PPO.load(model_path, env=vec_env)
    replay_dir = os.path.join(output_dir, "replays") if save_replays else None
    results = _run_eval_loop(model, vec_env, episodes, label=f"P{phase_idx + 1}",
                             scenarios=scenarios, replay_dir=replay_dir)
    if replay_dir:
        write_manifest(replay_dir)

    csv_path = os.path.join(csv_d, "eval_final.csv")
    results_to_dataframe(results).to_csv(csv_path, index=False)
    research = save_research_figure(results, img_d, "evaluation_research.jpeg",
                                    f"  ·  Phase {phase_idx + 1}  ·  {episodes} episodes")
    _save_compact_summary(results, os.path.join(img_d, "evaluation_summary.jpeg"))
    if scenarios is not None:
        tbl = per_type_table(results)
        tbl.to_csv(os.path.join(csv_d, "eval_per_type.csv"))
        save_scenario_figure(results, img_d, "scenario_breakdown.jpeg",
                             f"  ·  Phase {phase_idx + 1}")
        print("\n===== PER-SCENARIO-TYPE =====")
        print(tbl.to_string())

    print("\n===== EVALUATION COMPLETE =====")
    _print_summary(results)
    print(f"\nSaved CSV  → {csv_path}\nSaved figs → {img_d}")
    return results


def evaluate_phase(phase_idx, model_path, vnorm_path, env_class,
                   episodes=None, output_dir=None):
    """Per-phase evaluation (signature unchanged — called from train.py)."""
    episodes = episodes or PHASE_EVAL_EPISODES
    output_dir = output_dir or os.path.join(RESULTS_ROOT, f"train_phase{phase_idx + 1}")
    csv_d, img_d = _csv_dir(output_dir), _img_dir(output_dir)

    if not os.path.exists(model_path + ".zip"):
        print(f"  [eval_phase{phase_idx + 1}] model not found: {model_path}.zip"); return
    if not os.path.exists(vnorm_path):
        print(f"  [eval_phase{phase_idx + 1}] vnorm not found: {vnorm_path}"); return

    print(f"\n  ── Post-phase {phase_idx + 1} evaluation ({episodes} episodes) ──")
    vec_env = _make_eval_env(env_class, vnorm_path)
    model = PPO.load(model_path, env=vec_env)
    results = _run_eval_loop(model, vec_env, episodes, label=f"P{phase_idx + 1}")

    csv_p = os.path.join(csv_d, f"eval_phase{phase_idx + 1}.csv")
    results_to_dataframe(results).to_csv(csv_p, index=False)
    save_research_figure(results, img_d, f"eval_phase{phase_idx + 1}_research.jpeg",
                         f"  ·  Phase {phase_idx + 1}")
    raw_env = results.get("last_raw_env")
    if raw_env is not None:
        try:
            _render_env_map(raw_env, os.path.join(
                img_d, f"eval_phase{phase_idx + 1}_env_map.jpeg"),
                title=f"Phase {phase_idx + 1} — Environment Map")
        except Exception as exc:
            print(f"  [WARNING] env-map figure failed: {exc}")
    _print_summary(results)
    print(f"  Saved CSV  → {csv_p}\n  Saved imgs → {img_d}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, default=EVAL_PHASE_IDX + 1, help="1-4")
    ap.add_argument("--episodes", type=int, default=EPISODES)
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--vnorm", default=None)
    ap.add_argument("--suite", choices=["full", "test10", "random"], default="full",
                    help="full = 50 fixed seeds; test10 = 1 random seed per type; "
                         "random = unseeded maps (old behaviour)")
    ap.add_argument("--test-rng", type=int, default=None,
                    help="seed for the test10 draw; set it so the 10 picks are reproducible")
    ap.add_argument("--save-replays", action="store_true",
                    help="save every evaluation episode as a replay .npz (render with evaluation.replay)")
    a = ap.parse_args()
    vn = a.vnorm or (a.model + "_vnorm.pkl" if os.path.exists(a.model + "_vnorm.pkl")
                     else VNORM_PATH)
    evaluate(episodes=a.episodes, phase_idx=a.phase - 1, model_path=a.model, vnorm_path=vn,
             suite=a.suite, test_rng=a.test_rng, save_replays=a.save_replays)