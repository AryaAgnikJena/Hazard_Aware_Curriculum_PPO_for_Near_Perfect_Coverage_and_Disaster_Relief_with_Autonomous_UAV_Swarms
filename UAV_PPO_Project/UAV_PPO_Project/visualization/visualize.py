"""
visualization/visualize.py

3-D animated rollout visualiser for the multi-UAV environment.

Outputs saved when all epochs finish (results/visualization_res/):
  visualize_summary.png
  visualize_coverage_graph.png
  visualize_epoch_stats_performance.jpeg  ← 2×2 (Coverage, Return, Length, CDF)
  visualize_epoch_stats_mission.jpeg      ← 2×2 (Collisions, Victims, RTB, Return vs Cov)
  visualize_epoch_stats_table.jpeg        ← 1×1 (Per-Epoch Summary Table)
  map_snapshots/map_{top_down,isometric,low_side}.jpeg
  logs/networks/network_graph_final.jpeg
"""

import os
import sys
import glob
import time
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("TkAgg")           # interactive; switch to Agg for headless
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
import matplotlib.ticker as _mticker
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from stable_baselines3 import PPO

import configs.config as config
from env.multi_uav_env import PHASE_ENVS
from env.scenarios import sample_test_set, describe
from evaluation.replay import (ReplayRecorder, write_manifest, render_replay,
                               launch_background_render, render_async)

if sys.stdout.encoding != "utf-8":            # Windows consoles: never die on a non-ASCII print
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
try:
    from visualization.network_viz import draw_uav_network
except ImportError:                           # file lives in evaluation/ in your tree
    from evaluation.network_viz import draw_uav_network

VIZ_PHASE_IDX = 3                             # 0-based: 3 = Phase 4 (lethal collisions)

# ── Output directory ──────────────────────────────────────────────────────
VIZ_OUT_DIR = os.path.join("results", "visualization_res")

# ── Replays of the test episodes ──────────────────────────────────────────
SAVE_REPLAYS = True        # one .npz per epoch in results/visualization_res/replays/<timestamp>/
RENDER_GIFS  = True        # render GIFs in a BACKGROUND process
RENDER_MODE  = "top"       # "top" (fast 2-D) or "3d"

# ── Collision tolerance ────────────────────────────────────────────────────
COLLISION_VIZ_TOLERANCE = float(getattr(config, "DRONE_COLLISION_RADIUS", 3.0))

MAX_TRAIL_LEN  = 70
INTERVAL_MS    = 50
ROLLING_WINDOW = 600
WIND_GRID      = 8

_WIND_CAP = max(
    float(getattr(config, "WIND_MEAN_SPEED", 6.0))
    + float(getattr(config, "WIND_GUST_AMP",   3.0)),
    6.0,
)

# Research-style light palette (matches evaluate.py)
C_GREEN  = "#1a7f37"
C_BLUE   = "#0969da"
C_ORANGE = "#bc4c00"
C_RED    = "#cf222e"
C_PURPLE = "#8250df"
C_YELLOW = "#9a6700"
C_GREY   = "#57606a"
C_PINK   = "#bf3989"
C_TEAL   = "#1b7c83"
C_TEXT   = "#1f2328"


# ── Building renderer ─────────────────────────────────────────────────────────

def _draw_buildings(ax, buildings, class_ids=None):
    CLASS_COLORS = {
        0: "#6e8f7a",
        1: "#8f755a",
        2: "#5a7a9a",
    }
    DEFAULT_COLOR = "#7a7a7a"
    polys = []
    for k, (x, y, h, w) in enumerate(buildings):
        fc = (CLASS_COLORS.get(int(class_ids[k]), DEFAULT_COLOR)
              if class_ids is not None and k < len(class_ids)
              else DEFAULT_COLOR)
        verts = np.array([
            [x,   y,   0], [x+w, y,   0], [x+w, y+w, 0], [x,   y+w, 0],
            [x,   y,   h], [x+w, y,   h], [x+w, y+w, h], [x,   y+w, h],
        ])
        faces = [
            [verts[0], verts[1], verts[2], verts[3]],
            [verts[4], verts[5], verts[6], verts[7]],
            [verts[0], verts[1], verts[5], verts[4]],
            [verts[1], verts[2], verts[6], verts[5]],
            [verts[2], verts[3], verts[7], verts[6]],
            [verts[3], verts[0], verts[4], verts[7]],
        ]
        poly = Poly3DCollection(
            faces, facecolors=fc, edgecolors="#444444",
            alpha=0.72, linewidth=0.35)
        ax.add_collection3d(poly)
        polys.append(poly)
    return polys


# ── Collision checker ─────────────────────────────────────────────────────────

def _check_drone_collisions(positions, tol=COLLISION_VIZ_TOLERANCE):
    """
    Return list of (i, j) pairs where drones i and j are within `tol` metres.
    Uses pairwise XYZ Euclidean distance — matches the env's physics exactly.
    """
    n     = len(positions)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(positions[i] - positions[j]))
            if d <= tol:
                pairs.append((i, j))
    return pairs


# ── Research-grade epoch-stats figures (split into 3 readable files) ─────────

def _save_epoch_stats_figure(epoch_records: list, save_path: str):
    """
    Generate three readable per-epoch stats figures:
      <stem>_performance.jpeg  (2×2: Coverage, Return, Episode Length, CDF)
      <stem>_mission.jpeg      (2×2: Collisions, Victims, RTB, Return vs Coverage)
      <stem>_table.jpeg        (1×1: Per-Epoch Summary Table)
    """
    n = len(epoch_records)
    if n == 0:
        return

    stem, ext = os.path.splitext(save_path)
    ext = ext or ".jpeg"

    ep_ax        = np.arange(1, n + 1)
    coverages    = [r["final_coverage"] for r in epoch_records]
    rewards      = [r["total_reward"]   for r in epoch_records]
    steps_ep     = [r["steps"]          for r in epoch_records]
    rescues      = [r["peak_rescued"]   for r in epoch_records]
    coll_frames  = [r["collisions"]     for r in epoch_records]
    landed       = [r["landed_count"]   for r in epoch_records]

    bg, panel, edge = "#ffffff", "#f8f9fa", "#d0d7de"

    plt.rcParams.update({
        "figure.facecolor": bg, "axes.facecolor": panel,
        "axes.edgecolor": edge, "axes.labelcolor": "#24292f",
        "axes.titlecolor": C_TEXT, "axes.titlesize": 11.5,
        "axes.labelsize": 10, "axes.grid": True,
        "grid.color": edge, "grid.linewidth": 0.6,
        "xtick.color": C_GREY, "ytick.color": C_GREY,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.facecolor": bg, "legend.edgecolor": edge,
        "legend.labelcolor": "#24292f", "legend.fontsize": 8.5,
        "text.color": "#24292f", "font.family": "DejaVu Sans",
    })

    def _stat_box(ax, vals, color):
        mu, sig = float(np.mean(vals)), float(np.std(vals))
        ax.text(0.97, 0.97,
                f"μ={mu:.1f}\nσ={sig:.1f}\nmin={min(vals):.1f}\nmax={max(vals):.1f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                color=color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=bg,
                          edgecolor=color, alpha=0.9), zorder=10)

    def _save(fig, suffix):
        p = f"{stem}_{suffix}{ext}"
        fig.savefig(p, dpi=160, format="jpeg", bbox_inches="tight", facecolor=bg)
        plt.close(fig)
        print(f"  Epoch stats figure saved → {p}")

    # ── Figure 1: Performance (2×2) ──────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    fig.patch.set_facecolor(bg)
    fig.suptitle("Visualisation — Per-Epoch Performance",
                 fontsize=13.5, fontweight="bold", color=C_TEXT, y=0.98)

    bar_cols = [C_TEAL if c >= 95 else (C_BLUE if c >= 70 else C_RED)
                for c in coverages]
    axes[0, 0].bar(ep_ax, coverages, color=bar_cols, alpha=0.85, edgecolor="none", width=0.7)
    axes[0, 0].axhline(95, color=C_ORANGE, ls="--", lw=1.3, alpha=0.9, label="95% target")
    axes[0, 0].axhline(float(np.mean(coverages)), color=C_GREEN, ls=":", lw=1.2,
                       alpha=0.8, label=f"μ={np.mean(coverages):.1f}%")
    axes[0, 0].set_ylim(0, 108)
    axes[0, 0].set_title("Coverage per Epoch")
    axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Coverage %")
    axes[0, 0].legend(fontsize=8)
    _stat_box(axes[0, 0], coverages, C_GREEN)

    axes[0, 1].plot(ep_ax, rewards, "o-", color=C_BLUE, ms=5.5, lw=1.6)
    axes[0, 1].axhline(float(np.mean(rewards)), color=C_BLUE, ls=":", lw=1.1, alpha=0.7)
    if n >= 4:
        w = max(n // 5, 2)
        rm = np.convolve(rewards, np.ones(w) / w, mode="valid")
        axes[0, 1].plot(ep_ax[w - 1:], rm, color=C_ORANGE, lw=1.8, ls="--",
                        alpha=0.9, label=f"rolling mean (w={w})")
        axes[0, 1].legend(fontsize=8)
    axes[0, 1].set_title("Total Return per Epoch")
    axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("Cumulative reward")
    _stat_box(axes[0, 1], rewards, C_BLUE)

    _s_max  = max(steps_ep) if steps_ep else 1
    _s_mean = float(np.mean(steps_ep)) if steps_ep else 1
    _y_max  = min(max(_s_max * 1.15, _s_mean * 3, 10), config.MAX_STEPS * 1.05)
    axes[1, 0].plot(ep_ax, steps_ep, "s-", color=C_YELLOW, ms=5.5, lw=1.6)
    axes[1, 0].axhline(_s_mean, color=C_YELLOW, ls=":", lw=1.1, alpha=0.7)
    if _s_max >= config.MAX_STEPS * 0.80:
        axes[1, 0].axhline(config.MAX_STEPS, color=C_GREY, ls="--", lw=1.0,
                           alpha=0.6, label=f"MAX={config.MAX_STEPS}")
        axes[1, 0].legend(fontsize=8)
    axes[1, 0].set_ylim(0, _y_max)
    axes[1, 0].yaxis.set_major_locator(_mticker.MaxNLocator(nbins=6, integer=True))
    axes[1, 0].yaxis.set_major_formatter(_mticker.FuncFormatter(
        lambda x, _: f"{int(x // 1000)}k" if x >= 1000 else str(int(x))))
    axes[1, 0].set_title(f"Episode Length per Epoch  (μ={_s_mean:.0f}  max={_s_max})")
    axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("Steps")
    _stat_box(axes[1, 0], steps_ep, C_YELLOW)

    sorted_cov = np.sort(coverages)
    cdf = np.arange(1, n + 1) / n
    axes[1, 1].plot(sorted_cov, cdf, color=C_GREEN, lw=2.0)
    axes[1, 1].axvline(95, color=C_ORANGE, ls="--", lw=1.2, alpha=0.85, label="95% target")
    axes[1, 1].fill_between(sorted_cov, cdf, alpha=0.12, color=C_GREEN)
    pct_above = float(np.mean(np.array(coverages) >= 95)) * 100
    axes[1, 1].set_title(f"Coverage CDF  ({pct_above:.1f}% epochs ≥ 95%)")
    axes[1, 1].set_xlabel("Coverage %"); axes[1, 1].set_ylabel("Cum. probability")
    axes[1, 1].set_xlim(0, 105); axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "performance")

    # ── Figure 2: Mission & Safety (2×2) ─────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    fig.patch.set_facecolor(bg)
    fig.suptitle("Visualisation — Per-Epoch Safety & Mission Outcomes",
                 fontsize=13.5, fontweight="bold", color=C_TEXT, y=0.98)

    _bot = np.zeros(n)
    for _k, _c in (("coll_drone", C_RED), ("coll_bird", C_ORANGE),
                   ("coll_building", C_PURPLE)):
        _v = np.array([r.get(_k, 0) for r in epoch_records], float)
        axes[0, 0].bar(ep_ax, _v, bottom=_bot, color=_c, alpha=0.85,
                       edgecolor="none", width=0.7, label=_k[5:])
        _bot += _v
    axes[0, 0].plot(ep_ax, coll_frames, "o-", color=C_TEXT, ms=3.5, lw=1.1,
                    label="steps with ≥1 hit")
    _ev = np.array([r.get("collision_events", 0) for r in epoch_records], float)
    axes[0, 0].plot(ep_ax, _ev, "s--", color=C_YELLOW, ms=3.5, lw=1.1,
                    label=f"events (μ={_ev.mean():.1f})")
    axes[0, 0].axhline(float(np.mean(coll_frames)), color=C_ORANGE, ls="--",
                       lw=1.2, alpha=0.85, label=f"μ={np.mean(coll_frames):.1f} steps")
    axes[0, 0].set_title("Collisions per Epoch  (steps stacked by kind, + events)")
    axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Count")
    axes[0, 0].legend(fontsize=7.5, ncol=2)

    max_v = int(getattr(config, "NUM_VICTIMS", 0))
    if max_v > 0 and any(r > 0 for r in rescues):
        bar_cols2 = [C_TEAL if r >= max_v else C_BLUE for r in rescues]
        axes[0, 1].bar(ep_ax, rescues, color=bar_cols2, alpha=0.85, edgecolor="none", width=0.7)
        axes[0, 1].axhline(max_v, color=C_ORANGE, ls=":", lw=1.2, alpha=0.8,
                           label=f"total victims={max_v}")
        axes[0, 1].axhline(float(np.mean(rescues)), color=C_TEAL, ls="--", lw=1.2,
                           alpha=0.8, label=f"μ={np.mean(rescues):.1f}")
        axes[0, 1].set_ylim(0, max_v + 1.5)
        axes[0, 1].set_title(f"Victims Rescued per Epoch  (μ={np.mean(rescues):.1f}/{max_v})")
        axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("Rescued")
        axes[0, 1].legend(fontsize=8)
    else:
        axes[0, 1].set_facecolor(bg)
        axes[0, 1].text(0.5, 0.5, "No victims\nin this phase",
                        ha="center", va="center", transform=axes[0, 1].transAxes,
                        fontsize=12, color=C_GREY, style="italic")
        axes[0, 1].set_title("Victims Rescued")
        for sp in axes[0, 1].spines.values(): sp.set_visible(False)
        axes[0, 1].set_xticks([]); axes[0, 1].set_yticks([])

    bar_cols3 = [C_TEAL if l == config.NUM_DRONES else
                 (C_YELLOW if l > 0 else C_RED) for l in landed]
    axes[1, 0].bar(ep_ax, landed, color=bar_cols3, alpha=0.85, edgecolor="none", width=0.7)
    axes[1, 0].axhline(config.NUM_DRONES, color=C_GREEN, ls=":", lw=1.2,
                       alpha=0.8, label=f"total={config.NUM_DRONES}")
    axes[1, 0].axhline(float(np.mean(landed)), color=C_PURPLE, ls="--", lw=1.2,
                       alpha=0.8, label=f"μ={np.mean(landed):.1f}")
    axes[1, 0].set_ylim(0, config.NUM_DRONES + 1.5)
    axes[1, 0].set_title(f"Drones RTB per Epoch  (μ={np.mean(landed):.1f}/{config.NUM_DRONES})")
    axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("Drones landed")
    axes[1, 0].legend(fontsize=8)

    sc = axes[1, 1].scatter(coverages, rewards, c=ep_ax, cmap="plasma",
                            s=55, alpha=0.85, edgecolors="none", zorder=3)
    cbar = plt.colorbar(sc, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cbar.set_label("Epoch", fontsize=8, color=C_GREY)
    cbar.ax.tick_params(colors=C_GREY, labelsize=7.5)
    if n > 2:
        z = np.polyfit(coverages, rewards, 1)
        px = np.linspace(min(coverages), max(coverages), 100)
        axes[1, 1].plot(px, np.polyval(z, px), color=C_ORANGE, lw=1.5, ls="--",
                        alpha=0.85, label="trend")
        axes[1, 1].legend(fontsize=8)
    axes[1, 1].set_title("Return vs Coverage  (colour = epoch index)")
    axes[1, 1].set_xlabel("Coverage %"); axes[1, 1].set_ylabel("Return")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save(fig, "mission")

    # ── Figure 3: Per-Epoch Summary Table (1×1) ──────────────────────────
    fig_h = max(4.5, 1.4 + 0.42 * min(n, 25))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.axis("off")

    hdr = ["Epoch", "Cov%", "Return", "Steps", "Colls", "Kill", "Resc", "RTB"]
    rows = [
        [str(r["epoch"] + 1), f"{r['final_coverage']:.1f}", f"{r['total_reward']:.0f}",
         str(r["steps"]), str(r["collisions"]),
         "Y" if r.get("collision_terminated", False) else "N",
         str(r["peak_rescued"]), str(r["landed_count"])]
        for r in epoch_records
    ]

    col_x = [0.02, 0.14, 0.27, 0.41, 0.53, 0.65, 0.76, 0.88]
    hdr_y = 0.95
    row_h = min(0.82 / max(len(rows), 1), 0.085)

    for cx, h in zip(col_x, hdr):
        ax.text(cx, hdr_y, h, transform=ax.transAxes,
                fontsize=10, color=C_TEXT, fontweight="bold", va="top")
    ax.plot([0.01, 0.99], [hdr_y - 0.035, hdr_y - 0.035],
            transform=ax.transAxes, color=edge, lw=1.0, zorder=5, clip_on=False)

    for ri, row in enumerate(rows):
        y_pos = hdr_y - 0.07 - ri * row_h
        if y_pos < 0.02:
            ax.text(0.5, 0.01, f"… {len(rows) - ri} more epochs",
                    transform=ax.transAxes, ha="center",
                    fontsize=8.5, color=C_GREY, va="bottom")
            break
        if ri % 2 == 0:
            rect = mpatches.FancyBboxPatch(
                (0.0, y_pos - row_h * 0.65), 1.0, row_h * 0.95,
                transform=ax.transAxes, boxstyle="square,pad=0",
                linewidth=0, facecolor="#eef1f4", zorder=1)
            ax.add_patch(rect)
        cov_val  = float(row[1])
        cov_col  = C_TEAL if cov_val >= 95 else (C_BLUE if cov_val >= 70 else C_RED)
        coll_col = C_RED if int(row[4]) > 0 else C_GREEN
        for ci, (cx, cell) in enumerate(zip(col_x, row)):
            cell_col = "#24292f"
            if ci == 1: cell_col = cov_col
            if ci == 4: cell_col = coll_col
            if ci == 5: cell_col = C_RED if cell == "Y" else C_GREEN
            ax.text(cx, y_pos, cell, transform=ax.transAxes,
                    fontsize=9.5, color=cell_col, va="top")

    ax.set_title("Per-Epoch Summary Table", color=C_TEXT, fontsize=13,
                 fontweight="bold", pad=10)
    _save(fig, "table")

    plt.rcParams.update(plt.rcParamsDefault)


# ── Main visualiser ───────────────────────────────────────────────────────────

def visualize(model_path=None, epochs=10, phase_idx=VIZ_PHASE_IDX, vnorm_path=None,
              test_rng=None):
    """Run the animated 3D rollout visualiser and save split post-run figures."""
    if model_path is None:
        model_path = os.path.join(config.MODEL_DIR, "final_model")

    env   = PHASE_ENVS[phase_idx]()
    model = PPO.load(model_path, device="cpu")

    _cands = [vnorm_path, model_path + "_vnorm.pkl",
              os.path.join(os.path.dirname(model_path) or ".", "vecnormalize.pkl")]
    _vp = next((c for c in _cands if c and os.path.exists(c)), None)
    _vn = VecNormalize.load(_vp, DummyVecEnv([lambda: env])) if _vp else None
    print(f"  Obs normalisation: {_vp if _vp else 'NONE (stats file not found!)'}")

    def _norm(o):
        return _vn.normalize_obs(o) if _vn is not None else o

    _rounds = -(-epochs // 10)
    tests = sum((sample_test_set(None if test_rng is None else test_rng + r)
                 for r in range(_rounds)), [])[:epochs]
    print(f"  Test scenarios (test_rng={test_rng}): {describe(tests)}")
    env.set_next_scenario(tests[0])
    obs, _ = env.reset()
    obs = _norm(obs)

    replay_dir = os.path.join(VIZ_OUT_DIR, "replays", time.strftime("%Y%m%d_%H%M%S"))
    rec = ReplayRecorder()

    def _rmeta(epoch_idx):
        return dict(phase=phase_idx + 1, epoch=epoch_idx + 1, model=str(model_path),
                    vnorm=str(_vp), test_rng=test_rng)

    def _safe(fn, *a, **k):
        """A replay problem must never kill the animation (Tk silently stops it)."""
        if not SAVE_REPLAYS:
            return None
        try:
            return fn(*a, **k)
        except Exception as exc:
            print(f"  [WARNING] replay: {type(exc).__name__}: {exc}")
            return None

    _safe(rec.start, env, _rmeta(0))
    print(f"  Saving replays -> {os.path.abspath(replay_dir)}")

    colors_drones = plt.cm.jet(np.linspace(0, 1, env.num_drones))
    colors_epochs = plt.cm.tab10(np.linspace(0, 1, max(epochs, 1)))

    epoch_records = []

    state = {
        "obs":              obs,
        "current_epoch":    0,
        "coverage_history": [],
        "epoch_histories":  [],
        "drone_trails":     [deque(maxlen=MAX_TRAIL_LEN)
                             for _ in range(env.num_drones)],
        "total_reward":     0.0,
        "peak_rescued":     0,
        "collision_frames": 0,
        "collision_events": 0,
        "kind_drone": 0, "kind_bird": 0, "kind_building": 0,
        "active_collision": False,
        "collision_flash_frames": 0,
    }

    # ── Figure: 3 columns ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 9))
    fig.patch.set_facecolor("#ffffff")

    gs = GridSpec(
        2, 3, figure=fig,
        width_ratios=[1.3, 1.0, 0.55],
        height_ratios=[0.22, 1.0],
        hspace=0.10, wspace=0.28,
    )
    ax_wind = fig.add_subplot(gs[0, 0])
    ax3d    = fig.add_subplot(gs[1, 0], projection="3d")
    ax_cov  = fig.add_subplot(gs[:, 1])
    ax_fuel = fig.add_subplot(gs[:, 2])

    ax3d.set_facecolor("#ffffff")
    for ax in (ax_cov, ax_wind, ax_fuel):
        ax.set_facecolor("#f8f9fa")
    ax3d.tick_params(colors=C_GREY, labelsize=8)
    ax_cov.tick_params(colors=C_GREY, labelsize=8)
    ax_wind.tick_params(colors=C_GREY, labelsize=8)
    ax_fuel.tick_params(colors=C_GREY, labelsize=8)
    for sp in ax_cov.spines.values():  sp.set_edgecolor("#d0d7de")
    for sp in ax_wind.spines.values(): sp.set_edgecolor("#d0d7de")
    for sp in ax_fuel.spines.values(): sp.set_edgecolor("#d0d7de")

    # ── 3D scene ─────────────────────────────────────────────────────────────
    ax3d.set_xlim(0, env.map_size)
    ax3d.set_ylim(0, env.map_size)
    ax3d.set_zlim(0, env.max_height + 1.0)
    ax3d.set_xlabel("X", color="#24292f", fontsize=9)
    ax3d.set_ylabel("Y", color="#24292f", fontsize=9)
    ax3d.set_zlabel("Z", color="#24292f", fontsize=9)

    cids           = getattr(env, "building_class_ids", None)
    building_polys = _draw_buildings(ax3d, env.buildings, cids)

    gx = env.coverage_points[:, 0]
    gy = env.coverage_points[:, 1]
    gz = np.zeros_like(gx)
    ground_scatter = ax3d.scatter(
        gx, gy, gz, s=5, c="#b8c0cc", depthshade=False, alpha=0.85, zorder=1)

    cov_glow_outer = ax3d.scatter([], [], [], s=56, c="#2da44e",
                                  depthshade=False, alpha=0.18,
                                  edgecolors="none", zorder=2)
    cov_glow_mid   = ax3d.scatter([], [], [], s=34, c="#1a7f37",
                                  depthshade=False, alpha=0.45,
                                  edgecolors="none", zorder=3)
    cov_core       = ax3d.scatter([], [], [], s=16, c="#116329",
                                  depthshade=False, alpha=0.95,
                                  edgecolors="#ffffff", linewidths=0.45, zorder=4)

    def _checkpoint_artists():
        cp = env.checkpoints
        if len(cp) == 0:
            return None
        zcp = np.full(len(cp), 0.15)
        vis = env.checkpoint_visited
        col = np.where(vis, "#d4a017", "#e3b341")
        return ax3d.scatter(cp[:, 0], cp[:, 1], zcp, s=55, c=list(col),
                            marker="o", depthshade=False, zorder=5,
                            edgecolors="#7d5a00", linewidths=0.8)

    checkpoint_scatter = _checkpoint_artists()

    theta   = np.linspace(0, 2 * np.pi, 22)
    cx_ring = env.detection_radius * np.cos(theta)
    cy_ring = env.detection_radius * np.sin(theta)
    cz_ring = np.zeros_like(theta)
    detection_rings = [
        ax3d.plot([], [], [], lw=0.6, color=colors_drones[i],
                  alpha=0.45, linestyle="--")[0]
        for i in range(env.num_drones)
    ]
    trail_lines = [
        ax3d.plot([], [], [], lw=1.0, color=colors_drones[i], alpha=0.60)[0]
        for i in range(env.num_drones)
    ]
    drone_dots = [
        ax3d.scatter([], [], [], s=65, color=colors_drones[i],
                     depthshade=False, zorder=7)
        for i in range(env.num_drones)
    ]
    max_links  = env.num_drones * (env.num_drones - 1) // 2
    comm_lines = [
        ax3d.plot([], [], [], lw=1.15, color=C_BLUE, alpha=0.75)[0]
        for _ in range(max_links)
    ]
    collision_lines = [
        ax3d.plot([], [], [], lw=2.5, color=C_RED, alpha=0.0)[0]
        for _ in range(max_links)
    ]
    bird_scatter = ax3d.scatter([], [], [], s=40, c=C_RED,
                                marker="^", depthshade=False, zorder=6)

    title_text = ax3d.text2D(0.02, 0.98, "", transform=ax3d.transAxes,
                             color=C_TEXT, fontsize=9, va="top")
    collision_text = ax3d.text2D(
        0.5, 0.52, "", transform=ax3d.transAxes,
        color=C_RED, fontsize=20, fontweight="bold",
        ha="center", va="center",
        path_effects=[pe.withStroke(linewidth=4, foreground="#ffffff")])

    # ── Wind panel ────────────────────────────────────────────────────────────
    _wind_xlim = _WIND_CAP * 1.08
    ax_wind.set_xlim(-_wind_xlim, _wind_xlim)
    ax_wind.set_ylim(-0.6, 2.6)
    ax_wind.set_yticks([0, 1, 2])
    ax_wind.set_yticklabels(["Wx", "Wy", "|W|"], color="#24292f", fontsize=8)
    _tick_step = 2.0
    _pos_ticks = np.arange(0, _wind_xlim + _tick_step, _tick_step)
    _ticks     = np.unique(np.concatenate([-_pos_ticks[1:], _pos_ticks]))
    ax_wind.set_xticks(_ticks)
    ax_wind.set_xticklabels([f"{v:.0f}" for v in _ticks],
                            color=C_GREY, fontsize=7)
    ax_wind.axvline(0, color="#8b949e", linewidth=0.8)
    ax_wind.grid(axis="x", color="#d0d7de", linewidth=0.5, alpha=0.8)
    ax_wind.set_title("Wind (m/s)", color=C_YELLOW, fontsize=9, loc="left")
    wind_bars = ax_wind.barh([0, 1, 2], [0, 0, 0], height=0.48,
                             color=[C_BLUE, C_GREEN, C_YELLOW], alpha=0.85)

    w_ax     = np.linspace(env.map_size * 0.08, env.map_size * 0.92, WIND_GRID)
    wgx, wgy = np.meshgrid(w_ax, w_ax)
    wg_x     = wgx.ravel()
    wg_y     = wgy.ravel()
    wg_z     = np.zeros(WIND_GRID * WIND_GRID)
    wg_phase = (wg_x + wg_y) / (env.map_size + 1e-6)
    wind_quiver = [None]

    def _update_wind_field(frame_idx):
        if wind_quiver[0] is not None:
            wind_quiver[0].remove()
        wx, wy = float(env.wind_mean[0]), float(env.wind_mean[1])
        wmag   = float(np.hypot(wx, wy))
        if wmag < 1e-6:
            wind_quiver[0] = None
            for bar in wind_bars:
                bar.set_width(0); bar.set_x(0)
            return
        phase    = (wg_phase + frame_idx * 0.035) % 1.0
        pulse    = 0.45 + 0.55 * np.sin(phase * 2 * np.pi)
        base_len = (env.map_size / WIND_GRID) * 0.38
        ux = (wx / wmag) * base_len * (0.35 + 0.65 * pulse)
        uy = (wy / wmag) * base_len * (0.35 + 0.65 * pulse)
        uz = np.zeros_like(ux)
        col       = np.zeros((len(wg_x), 4))
        col[:, 0] = 0.80
        col[:, 1] = 0.35 + 0.25 * pulse
        col[:, 2] = 0.0
        col[:, 3] = 0.30 + 0.55 * pulse
        wind_quiver[0] = ax3d.quiver(
            wg_x, wg_y, wg_z, ux, uy, uz,
            colors=col, linewidth=0.8, arrow_length_ratio=0.35)
        for bar, val in zip(wind_bars, [wx, wy, wmag]):
            clipped = float(np.clip(val, -_wind_xlim, _wind_xlim))
            bar.set_width(abs(clipped))
            bar.set_x(clipped if clipped < 0 else 0)

    _update_wind_field(0)

    # ── Coverage panel ────────────────────────────────────────────────────────
    ax_cov.set_xlim(0, ROLLING_WINDOW)
    ax_cov.set_ylim(0, 100)
    ax_cov.set_xlabel("Step (rolling window)", color="#24292f", fontsize=9)
    ax_cov.set_ylabel("Coverage %", color="#24292f", fontsize=9)
    ax_cov.set_title("Coverage", color=C_TEXT, fontsize=10)
    ax_cov.grid(True, color="#d0d7de", linewidth=0.6, alpha=0.8)
    ax_cov.axhline(95, color=C_ORANGE, linestyle="--", linewidth=1.1,
                   alpha=0.85, label="95% target")
    MAX_LEGEND = 8
    live_line, = ax_cov.plot([], [], color=C_TEXT, lw=1.4,
                             label="current epoch")

    # ── Fuel gauge panel ──────────────────────────────────────────────────────
    _fuel_max    = float(config.DRONE_FUEL_STEPS)
    _rtb_fuel    = _fuel_max * config.FUEL_LOW_FRAC
    _drone_ys    = list(range(env.num_drones))
    _fuel_colors = [(*colors_drones[i][:3], 0.85)
                    for i in range(env.num_drones)]

    ax_fuel.set_xlim(0, _fuel_max)
    ax_fuel.set_ylim(-0.8, env.num_drones - 0.2)
    ax_fuel.set_yticks(_drone_ys)
    ax_fuel.set_yticklabels([f"D{i+1}" for i in range(env.num_drones)],
                            color="#24292f", fontsize=8)
    ax_fuel.set_xlabel("Fuel remaining", color="#24292f", fontsize=8)
    ax_fuel.set_title("Fuel", color=C_TEXT, fontsize=10)
    ax_fuel.grid(axis="x", color="#d0d7de", linewidth=0.5, alpha=0.8)
    ax_fuel.axvline(_rtb_fuel, color=C_RED, linewidth=0.9,
                    linestyle="--", alpha=0.85, label="RTB threshold")
    ax_fuel.legend(fontsize=7, frameon=False, labelcolor="#24292f",
                   loc="lower right")
    fuel_bars = ax_fuel.barh(_drone_ys, [_fuel_max] * env.num_drones,
                             height=0.55, color=_fuel_colors, edgecolor="none")
    fuel_texts = [
        ax_fuel.text(_fuel_max * 0.02, y, "100%",
                     va="center", ha="left",
                     color=C_TEXT, fontsize=7.5, fontweight="bold",
                     path_effects=[pe.withStroke(linewidth=2, foreground="#ffffff")])
        for y in _drone_ys
    ]

    def _update_fuel_bars():
        for i, (bar, txt) in enumerate(zip(fuel_bars, fuel_texts)):
            f      = float(env.fuel[i])
            pct    = f / _fuel_max
            landed = bool(env._landed[i])
            bar.set_width(f)
            if landed:
                bar.set_facecolor("#adb5bd")
                txt.set_text("RTB ✓")
                txt.set_color(C_GREY)
            elif pct <= config.FUEL_LOW_FRAC:
                bar.set_facecolor(C_RED)
                txt.set_text(f"{pct*100:.0f}% !")
                txt.set_color(C_RED)
            else:
                bar.set_facecolor(_fuel_colors[i])
                txt.set_text(f"{pct*100:.0f}%")
                txt.set_color(C_TEXT)

    # ── Helper functions ──────────────────────────────────────────────────────
    def _refresh_checkpoints():
        nonlocal checkpoint_scatter
        if checkpoint_scatter is not None:
            checkpoint_scatter.remove()
        checkpoint_scatter = _checkpoint_artists()

    def _sync_covered_scatter():
        pts = env.coverage_points[env.covered]
        if len(pts) == 0:
            cov_glow_outer._offsets3d = ([], [], [])
            cov_glow_mid._offsets3d   = ([], [], [])
            cov_core._offsets3d       = ([], [], [])
        else:
            z0 = np.zeros(len(pts))
            cov_glow_outer._offsets3d = (pts[:, 0], pts[:, 1], z0)
            cov_glow_mid._offsets3d   = (pts[:, 0], pts[:, 1], z0)
            cov_core._offsets3d       = (pts[:, 0], pts[:, 1], z0)

    def _start_new_epoch():
        nonlocal building_polys, checkpoint_scatter, ground_scatter
        env.set_next_scenario(tests[state["current_epoch"]])
        _obs, _ = env.reset()
        _safe(rec.start, env, _rmeta(state["current_epoch"]))
        state["obs"]               = _norm(_obs)
        state["coverage_history"]  = []
        state["total_reward"]      = 0.0
        state["peak_rescued"]      = 0
        state["collision_frames"]  = 0
        state["collision_events"]  = 0
        state["kind_drone"] = state["kind_bird"] = state["kind_building"] = 0
        state["active_collision"]  = False
        state["collision_flash_frames"] = 0
        collision_text.set_text("")
        for t in state["drone_trails"]:
            t.clear()
        for poly in building_polys:
            poly.remove()
        building_polys.clear()
        _cids = getattr(env, "building_class_ids", None)
        building_polys.extend(_draw_buildings(ax3d, env.buildings, _cids))
        _refresh_checkpoints()
        _sync_covered_scatter()
        ground_scatter.remove()
        gx2 = env.coverage_points[:, 0]
        gy2 = env.coverage_points[:, 1]
        gz2 = np.zeros_like(gx2)
        ground_scatter = ax3d.scatter(
            gx2, gy2, gz2, s=5, c="#b8c0cc", depthshade=False,
            alpha=0.85, zorder=1)
        for bar, txt, fc in zip(fuel_bars, fuel_texts, _fuel_colors):
            bar.set_width(_fuel_max)
            bar.set_facecolor(fc)
            txt.set_text("100%")
            txt.set_color(C_TEXT)
        for cl in collision_lines:
            cl.set_data([], [])
            cl.set_3d_properties([])
            cl.set_alpha(0.0)

    _sync_covered_scatter()

    # ── Animation frame ───────────────────────────────────────────────────────
    def update(_frame):
        ep = state["current_epoch"]

        action, _ = model.predict(state["obs"], deterministic=True)
        obs2, rew, terminated, truncated, info = env.step(action)
        _safe(rec.step, env, info, rew)
        state["obs"]          = _norm(obs2)
        state["total_reward"] += float(rew)
        done = terminated or truncated

        collision_terminated = False
        collision_hit = bool(info.get("collision_hit", False))
        if collision_hit:
            state["collision_frames"] += 1
            state["collision_flash_frames"] = 6
            for _k in ("drone", "bird", "building"):
                state[f"kind_{_k}"] += int(bool(info.get(f"hit_{_k}", False)))
        state["collision_events"] += int(info.get("collision_onsets", 0))

        if getattr(env, "rescued", None) is not None:
            state["peak_rescued"] = max(state["peak_rescued"],
                                        int(np.sum(env.rescued)))

        coll_pairs = _check_drone_collisions(env.positions,
                                             tol=COLLISION_VIZ_TOLERANCE)

        if state["collision_flash_frames"] > 0:
            state["collision_flash_frames"] -= 1
            state["active_collision"] = True
            _kinds = [k for k in ("drone", "bird", "building")
                      if info.get(f"hit_{k}", False)]
            collision_text.set_text(
                f"COLLISION {'/'.join(_kinds)}\n"
                f"({state['collision_frames']} steps, {state['collision_events']} events)")
            collision_text.set_color(C_RED)
        else:
            collision_text.set_text("")
            state["active_collision"] = False

        for k in range(max_links):
            if k < len(coll_pairs) and state["active_collision"]:
                i, j = coll_pairs[k]
                p1, p2 = env.positions[i], env.positions[j]
                collision_lines[k].set_data([p1[0], p2[0]], [p1[1], p2[1]])
                collision_lines[k].set_3d_properties([p1[2], p2[2]])
                collision_lines[k].set_alpha(0.85)
                collision_lines[k].set_color(C_RED)
            else:
                collision_lines[k].set_data([], [])
                collision_lines[k].set_3d_properties([])
                collision_lines[k].set_alpha(0.0)

        for i, pos in enumerate(env.positions):
            state["drone_trails"][i].append(pos.copy())
            tr = np.array(state["drone_trails"][i])
            if len(tr) > 1:
                trail_lines[i].set_data(tr[:, 0], tr[:, 1])
                trail_lines[i].set_3d_properties(tr[:, 2])
            drone_dots[i]._offsets3d = ([pos[0]], [pos[1]], [pos[2]])
            drone_dots[i].set_color("#8b949e" if (info.get("dead") and info["dead"][i])
                                    else colors_drones[i])
            detection_rings[i].set_data(pos[0] + cx_ring, pos[1] + cy_ring)
            detection_rings[i].set_3d_properties(cz_ring + pos[2] * 0.02)

        links = env.communication_links
        for k in range(max_links):
            if k < len(links):
                i, j = links[k]
                p1, p2 = env.positions[i], env.positions[j]
                comm_lines[k].set_data([p1[0], p2[0]], [p1[1], p2[1]])
                comm_lines[k].set_3d_properties([p1[2], p2[2]])
                comm_lines[k].set_visible(True)
            else:
                comm_lines[k].set_visible(False)

        if len(env.birds) > 0:
            bird_scatter._offsets3d = (
                env.birds[:, 0], env.birds[:, 1], env.birds[:, 2])
        else:
            bird_scatter._offsets3d = ([], [], [])

        _sync_covered_scatter()
        _refresh_checkpoints()
        _update_wind_field(env.step_count)
        _update_fuel_bars()

        coverage = env.get_coverage_percent()
        title_text.set_text(
            f"Epoch {ep+1}/{epochs}  |  {env.scenario_name}#{env.scenario_seed}"
            f"  |  Step {env.step_count}"
            f"  |  Coverage {coverage:.1f}%"
            f"  |  Colls {state['collision_frames']}"
            f"  |  Alive {env.num_drones - int(info.get('drones_dead', 0))}/{env.num_drones}"
            + ("  ⚡ COLLISION KILL" if collision_terminated else ""))

        state["coverage_history"].append(coverage)
        hist   = state["coverage_history"]
        window = hist[-ROLLING_WINDOW:]
        live_line.set_data(range(len(window)), window)
        ax_cov.set_xlim(0, max(len(window), 12))

        if done:
            final_cov = coverage
            _rp = os.path.join(
                replay_dir, f"epoch{ep+1:02d}_{env.scenario_name}_{env.scenario_seed}.npz")
            if _safe(rec.save, _rp, extra=dict(
                    **{"return": round(state["total_reward"], 1)},
                    collision_steps=state["collision_frames"],
                    collision_events=state["collision_events"],
                    drones_dead=int(info.get("drones_dead", 0)))):
                print(f"  Replay saved -> {_rp}")
                if RENDER_GIFS:
                    _safe(render_async, _rp, RENDER_MODE)
            epoch_records.append({
                "epoch":                ep,
                "scenario":             f"{env.scenario_name}#{env.scenario_seed}",
                "steps":                env.step_count,
                "final_coverage":       final_cov,
                "total_reward":         state["total_reward"],
                "peak_rescued":         state["peak_rescued"],
                "collisions":           state["collision_frames"],
                "collision_terminated": collision_terminated,
                "landed_count":         int(np.sum(env._landed)),
                "collision_events":     state["collision_events"],
                "dead_count":           int(info.get("drones_dead", 0)),
                "coll_drone":           state["kind_drone"],
                "coll_bird":            state["kind_bird"],
                "coll_building":        state["kind_building"],
            })

            print(
                f"Epoch {ep+1} ended [{env.scenario_name}#{env.scenario_seed}]"
                f" | Coverage {final_cov:.1f}%"
                f" | Steps {env.step_count}"
                f" | Return {state['total_reward']:.0f}"
                f" | Collision frames {state['collision_frames']}"
                f" | Events {state['collision_events']}"
                f" (d/b/B {state['kind_drone']}/{state['kind_bird']}/{state['kind_building']})"
                f" | Rescued {state['peak_rescued']}"
                f" | Landed {int(np.sum(env._landed))}"
                f" | Dead {int(info.get('drones_dead', 0))}"
                f" | FuelTerm={info.get('fuel_empty_terminated', False)}")

            color = colors_epochs[ep % len(colors_epochs)]
            ax_cov.plot(range(len(window)), window,
                        color=color, lw=0.85, alpha=0.65, label=f"Ep {ep+1}")
            handles, labels = ax_cov.get_legend_handles_labels()
            if len(handles) > MAX_LEGEND:
                handles = handles[-MAX_LEGEND:]
                labels  = labels[-MAX_LEGEND:]
            ax_cov.legend(handles, labels, fontsize=6.5,
                          loc="upper center",
                          bbox_to_anchor=(0.5, -0.18), ncol=3,
                          facecolor="#ffffff", labelcolor="#24292f",
                          edgecolor="#d0d7de")
            state["epoch_histories"].append(hist.copy())

            if ep + 1 >= epochs:
                ani.event_source.stop()
                os.makedirs(VIZ_OUT_DIR, exist_ok=True)

                # 1. Full live-window screenshot
                fig.savefig(
                    os.path.join(VIZ_OUT_DIR, "visualize_summary.png"),
                    dpi=150, bbox_inches="tight", facecolor="#ffffff")
                print("  Saved: visualize_summary.png")

                # 2. Clean per-epoch coverage graph
                _save_coverage_graph(state["epoch_histories"],
                                     colors_epochs, epochs)

                if SAVE_REPLAYS:
                    _safe(write_manifest, replay_dir)
                    if RENDER_GIFS:
                        _safe(launch_background_render, replay_dir, mode=RENDER_MODE)

                # 3. Research-grade per-epoch stats figures (3 split files)
                _save_epoch_stats_figure(
                    epoch_records,
                    os.path.join(VIZ_OUT_DIR, "visualize_epoch_stats.jpeg"))

                # 4. Three camera-angle map screenshots
                snap_dir = os.path.join(VIZ_OUT_DIR, "map_snapshots")
                os.makedirs(snap_dir, exist_ok=True)
                for angle_name, elev, azim in [
                    ("top_down",  90, -90),
                    ("isometric", 35,  45),
                    ("low_side",  15, 200),
                ]:
                    _save_map_snapshot(env, state, colors_drones,
                                       snap_dir, angle_name, elev, azim,
                                       final_cov)

                # 5. Network architecture diagram
                os.makedirs(config.NETWORKS_DIR, exist_ok=True)
                net_path = os.path.join(config.NETWORKS_DIR,
                                        "network_graph_final.jpeg")
                draw_uav_network(env, save_path=net_path,
                                 highlight="both", show=False)
                return

            state["current_epoch"] += 1
            _start_new_epoch()

    def _flush_partial(_evt=None):
        """Window closed / run aborted mid-epoch: keep what was recorded so far."""
        if SAVE_REPLAYS and rec.active:
            _pp = os.path.join(
                replay_dir,
                f"epoch{state['current_epoch'] + 1:02d}_{env.scenario_name}_"
                f"{env.scenario_seed}_PARTIAL.npz")
            if _safe(rec.snapshot, _pp):
                print(f"  Partial replay saved -> {_pp}")
                _safe(write_manifest, replay_dir)

    fig.canvas.mpl_connect("close_event", _flush_partial)

    plt.tight_layout()
    ani = FuncAnimation(fig, update, interval=INTERVAL_MS,
                        cache_frame_data=False)
    plt.show()
    return ani


# ── Post-run save helpers ─────────────────────────────────────────────────────

def _save_coverage_graph(epoch_histories, colors_epochs, epochs):
    fig_c, ax_c = plt.subplots(figsize=(10, 5), facecolor="#ffffff")
    ax_c.set_facecolor("#f8f9fa")
    ax_c.tick_params(colors=C_GREY, labelsize=9)
    for sp in ax_c.spines.values():
        sp.set_edgecolor("#d0d7de")

    for ep_idx, hist in enumerate(epoch_histories):
        col = colors_epochs[ep_idx % len(colors_epochs)]
        ax_c.plot(hist, color=col, lw=1.2, alpha=0.80, label=f"Ep {ep_idx+1}")

    ax_c.axhline(95, color=C_ORANGE, linestyle="--", lw=1.2,
                 alpha=0.85, label="95% target")
    ax_c.set_xlabel("Step", color="#24292f", fontsize=10)
    ax_c.set_ylabel("Coverage %", color="#24292f", fontsize=10)
    ax_c.set_title("Coverage per Epoch", color=C_TEXT, fontsize=11)
    ax_c.set_ylim(0, 105)
    ax_c.grid(True, color="#d0d7de", linewidth=0.6, alpha=0.8)

    handles, labels = ax_c.get_legend_handles_labels()
    ax_c.legend(handles, labels, fontsize=7, loc="lower right",
                facecolor="#ffffff", labelcolor="#24292f",
                edgecolor="#d0d7de", ncol=2)

    save_path = os.path.join(VIZ_OUT_DIR, "visualize_coverage_graph.png")
    fig_c.tight_layout()
    fig_c.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig_c)
    print("  Saved: visualize_coverage_graph.png")


def _save_map_snapshot(env, state, colors_drones,
                       snap_dir, angle_name, elev, azim, coverage):
    snap_fig = plt.figure(figsize=(10, 8), facecolor="#ffffff")
    snap_ax  = snap_fig.add_subplot(111, projection="3d")
    snap_ax.set_facecolor("#ffffff")
    snap_ax.tick_params(colors=C_GREY, labelsize=8)
    snap_ax.set_xlim(0, env.map_size)
    snap_ax.set_ylim(0, env.map_size)
    snap_ax.set_zlim(0, env.max_height + 1.0)
    snap_ax.set_xlabel("X", color="#24292f", fontsize=9)
    snap_ax.set_ylabel("Y", color="#24292f", fontsize=9)
    snap_ax.set_zlabel("Z", color="#24292f", fontsize=9)

    cids = getattr(env, "building_class_ids", None)
    _draw_buildings(snap_ax, env.buildings, cids)

    snap_ax.scatter(env.coverage_points[:, 0],
                    env.coverage_points[:, 1],
                    np.zeros(len(env.coverage_points)),
                    s=5, c="#b8c0cc", depthshade=False, alpha=0.85)

    covered_pts = env.coverage_points[env.covered]
    if len(covered_pts):
        snap_ax.scatter(covered_pts[:, 0], covered_pts[:, 1],
                        np.zeros(len(covered_pts)),
                        s=18, c="#1a7f37", depthshade=False, alpha=0.85,
                        edgecolors="#ffffff", linewidths=0.4)

    for di in range(env.num_drones):
        pos = env.positions[di]
        snap_ax.scatter([pos[0]], [pos[1]], [pos[2]],
                        s=70, color=colors_drones[di],
                        depthshade=False, zorder=7)
        tr = np.array(state["drone_trails"][di])
        if len(tr) > 1:
            snap_ax.plot(tr[:, 0], tr[:, 1], tr[:, 2],
                         lw=1.0, color=colors_drones[di], alpha=0.65)

    if len(env.birds) > 0:
        snap_ax.scatter(env.birds[:, 0], env.birds[:, 1], env.birds[:, 2],
                        s=35, c=C_RED, marker="^", depthshade=False)

    snap_ax.view_init(elev=elev, azim=azim)
    snap_ax.set_title(
        f"Final map — {angle_name.replace('_', ' ')}  "
        f"| Coverage {coverage:.1f}%",
        color=C_TEXT, fontsize=9, pad=6)

    snap_path = os.path.join(snap_dir, f"map_{angle_name}.jpeg")
    snap_fig.savefig(snap_path, dpi=150, format="jpeg",
                     bbox_inches="tight", facecolor="#ffffff")
    plt.close(snap_fig)
    print(f"  Map snapshot saved -> {snap_path}")


if __name__ == "__main__":
    visualize()