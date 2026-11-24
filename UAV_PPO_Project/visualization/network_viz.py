import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


# ── Colour palette — one per input feature group ──────────────────────────
GROUP_COLORS = {
    "pos":      "#378ADD",   # position xyz
    "vel":      "#D85A30",   # velocity xyz
    "scan_r":   "#E84393",   # scan radius norm
    "hazard":   "#3B6D11",   # hazard dist + hazard dir
    "team":     "#00A896",   # nearest-teammate dist + dir
    "uc_dir":   "#BA7517",   # direction to uncovered / RTB
    "walls":    "#8B5CF6",   # wall distances
    "fuel":     "#FF6B35",   # fuel fraction
    "global":   "#888780",   # coverage % + step fraction
    "wind":     "#2DA8D8",   # mean wind xy (OBS_INCLUDE_WIND)
    "bldg":     "#C2185B",   # nearest building dist/dir/roof clearance (Phase 4)
}

ACTOR_COLOR  = "#185FA5"
CRITIC_COLOR = "#3B6D11"
HIDDEN_COLOR = "#B4B2A9"
BG_COLOR     = "#FAFAF8"


def _node_positions(n_nodes, x, y_min, y_max):
    if n_nodes == 1:
        return [(x, (y_min + y_max) / 2)]
    ys = np.linspace(y_max, y_min, n_nodes)
    return [(x, y) for y in ys]


def _draw_connections(ax, src, dst, color, alpha, linewidth=0.35,
                      sample_src=None, sample_dst=None):
    s = src if sample_src is None else src[::max(1, len(src) // max(sample_src, 1))]
    d = dst if sample_dst is None else dst[::max(1, len(dst) // max(sample_dst, 1))]
    for (x1, y1) in s:
        for (x2, y2) in d:
            ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha,
                    linewidth=linewidth, zorder=1, solid_capstyle="round")


def draw_uav_network(env, save_path=None, highlight="both", show=True):
    num_drones    = env.num_drones
    obs_per_drone = getattr(env, "obs_per_drone", 20)   # must be 20

    # ── Build input metadata ───────────────────────────────────────────────
    # Each entry: (label, group_key)
    input_meta = []
    for d in range(num_drones):
        prefix = f"D{d+1}"
        input_meta += [(f"{prefix} px",      "pos"),
                       (f"{prefix} py",      "pos"),
                       (f"{prefix} pz",      "pos")]
        input_meta += [(f"{prefix} vx",      "vel"),
                       (f"{prefix} vy",      "vel"),
                       (f"{prefix} vz",      "vel")]
        input_meta += [(f"{prefix} scan_r",  "scan_r")]
        input_meta += [(f"{prefix} h_dist",  "hazard")]
        input_meta += [(f"{prefix} h_dx",    "hazard"),
                       (f"{prefix} h_dy",    "hazard")]
        input_meta += [(f"{prefix} team_d",  "team")]
        input_meta += [(f"{prefix} team_dx", "team"),
                       (f"{prefix} team_dy", "team")]
        input_meta += [(f"{prefix} uc_x",    "uc_dir"),
                       (f"{prefix} uc_y",    "uc_dir")]
        input_meta += [(f"{prefix} wL",      "walls"),
                       (f"{prefix} wR",      "walls"),
                       (f"{prefix} wB",      "walls"),
                       (f"{prefix} wT",      "walls")]
        input_meta += [(f"{prefix} fuel",    "fuel")]
    input_meta += [("Cov%",   "global"),
                   ("Step%",  "global")]
    has_wind = bool(getattr(env, "obs_include_wind", False))
    if has_wind:
        input_meta += [("Wx", "wind"), ("Wy", "wind")]
    if bool(getattr(env, "obs_building", False)):
        for d in range(num_drones):
            input_meta += [(f"D{d+1} b_dist", "bldg"), (f"D{d+1} b_dx", "bldg"),
                           (f"D{d+1} b_dy", "bldg"), (f"D{d+1} roof_clr", "bldg")]

    n_inputs = len(input_meta)   # num_drones*20 + 2 (+2 with wind)
    n_actor  = num_drones * 3
    n_critic = 1

    # Visual stand-in counts for hidden layers (proportional to 512/256/128)
    DRAW_H = [34, 24, 17]

    # ── Figure layout ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(24, 14), facecolor=BG_COLOR)
    ax  = fig.add_subplot(111)
    ax.set_facecolor(BG_COLOR)
    ax.set_aspect("equal")
    ax.axis("off")

    X = {
        "input":  0.04,
        "h1":     0.24,   # 512
        "h2":     0.42,   # 256
        "h3":     0.58,   # 128 (shared trunk ends here)
        "actor":  0.76,
        "critic": 0.92,
    }
    Y_MIN, Y_MAX = 0.02, 0.97
    R_IN  = 0.004
    R_H   = 0.006
    R_OUT = 0.008

    # ── Node positions ────────────────────────────────────────────────────
    input_pos  = _node_positions(n_inputs,   X["input"],  Y_MIN, Y_MAX)
    h1_pos     = _node_positions(DRAW_H[0],  X["h1"],     Y_MIN+0.02, Y_MAX-0.02)
    h2_pos     = _node_positions(DRAW_H[1],  X["h2"],     Y_MIN+0.05, Y_MAX-0.05)
    h3_pos     = _node_positions(DRAW_H[2],  X["h3"],     Y_MIN+0.08, Y_MAX-0.08)
    actor_pos  = _node_positions(n_actor,    X["actor"],  Y_MIN+0.05, Y_MAX-0.05)
    critic_pos = [(X["critic"], (Y_MIN + Y_MAX) / 2)]

    # ── Connections ───────────────────────────────────────────────────────
    # Input → H1  coloured by feature group
    groups = list(dict.fromkeys(m[1] for m in input_meta))
    for gk in groups:
        g_nodes = [input_pos[i] for i, m in enumerate(input_meta) if m[1] == gk]
        _draw_connections(ax, g_nodes, h1_pos,
                          color=GROUP_COLORS[gk], alpha=0.09,
                          sample_src=max(1, len(g_nodes) // 2), sample_dst=8)

    # H1 → H2 → H3  (shared trunk, grey)
    _draw_connections(ax, h1_pos, h2_pos, "#AAAAAA", 0.07, sample_src=12, sample_dst=9)
    _draw_connections(ax, h2_pos, h3_pos, "#AAAAAA", 0.09, sample_src=9,  sample_dst=7)

    # H3 → actor
    if highlight in ("actor", "both"):
        _draw_connections(ax, h3_pos, actor_pos, ACTOR_COLOR, 0.18,
                          linewidth=0.55, sample_src=9, sample_dst=None)

    # H3 → critic
    if highlight in ("critic", "both"):
        _draw_connections(ax, h3_pos, critic_pos, CRITIC_COLOR, 0.35,
                          linewidth=0.8, sample_src=None, sample_dst=None)

    # ── Draw input nodes ──────────────────────────────────────────────────
    for pos, (_, gk) in zip(input_pos, input_meta):
        col = GROUP_COLORS[gk]
        ax.add_patch(plt.Circle(pos, R_IN, color=col, ec=col, lw=0.3, zorder=3))

    # ── Draw hidden nodes ─────────────────────────────────────────────────
    for pos in h1_pos:
        ax.add_patch(plt.Circle(pos, R_H, color=HIDDEN_COLOR, ec="#888", lw=0.4, zorder=3))
    for pos in h2_pos:
        ax.add_patch(plt.Circle(pos, R_H, color=HIDDEN_COLOR, ec="#888", lw=0.4, zorder=3))
    for pos in h3_pos:
        ax.add_patch(plt.Circle(pos, R_H, color=HIDDEN_COLOR, ec="#888", lw=0.4, zorder=3))

    # ── Draw output nodes ─────────────────────────────────────────────────
    for pos in actor_pos:
        ax.add_patch(plt.Circle(pos, R_OUT, color=ACTOR_COLOR,
                                ec="#0C447C", lw=0.5, zorder=3))
    ax.add_patch(plt.Circle(critic_pos[0], R_OUT * 1.8,
                             color=CRITIC_COLOR, ec="#085041", lw=0.6, zorder=3))

    # ── Input group bracket labels ─────────────────────────────────────────
    # One label per group, vertically centred on that group's nodes
    seen_groups: dict[str, list] = {}
    for i, (_, gk) in enumerate(input_meta):
        seen_groups.setdefault(gk, []).append(input_pos[i])

    group_display = {
        "pos":    f"D1-D{num_drones} pos xyz (3)",
        "vel":    f"D1-D{num_drones} vel xyz (3)",
        "scan_r": f"D1-D{num_drones} scan_r (1)",
        "hazard": f"D1-D{num_drones} hazard dist+dir (3)",
        "team":   f"D1-D{num_drones} teammate dist+dir (3)",
        "uc_dir": f"D1-D{num_drones} uc/RTB dir (2)",
        "walls":  f"D1-D{num_drones} walls (4)",
        "fuel":   f"D1-D{num_drones} fuel_frac (1)",
        "global": "global cov+step (2)",
        "wind":   "mean wind xy (2)",
        "bldg":   f"D1-D{num_drones} bldg dist+dir+roof clr (4)",
    }
    drawn_groups: set = set()
    for _, gk in input_meta:
        if gk not in drawn_groups:
            drawn_groups.add(gk)
            nodes = seen_groups[gk]
            mid_y = float(np.mean([p[1] for p in nodes]))
            ax.text(X["input"] + 0.012, mid_y, group_display[gk],
                    ha="left", va="center", fontsize=6.0,
                    color=GROUP_COLORS[gk])

    # ── Actor output labels ───────────────────────────────────────────────
    actor_labels = [f"D{d+1} {ax_}"
                    for d in range(num_drones) for ax_ in ["ax", "ay", "az"]]
    for pos, lbl in zip(actor_pos, actor_labels):
        ax.text(X["actor"] + 0.012, pos[1], lbl,
                ha="left", va="center", fontsize=6.0, color=ACTOR_COLOR)

    ax.text(X["critic"] + 0.014, critic_pos[0][1], "V(s)",
            ha="left", va="center", fontsize=10,
            color=CRITIC_COLOR, fontweight="bold")

    # ── Hidden layer size annotations ─────────────────────────────────────
    for x_key, label in [("h1", "512\nReLU"), ("h2", "256\nReLU"), ("h3", "128\nReLU")]:
        ax.text(X[x_key], Y_MIN - 0.05, label,
                ha="center", va="top", fontsize=8, color="#5F5E5A")

    # Shared trunk dashed connector
    ax.annotate("", xy=(X["h3"] + 0.01, 0.5), xytext=(X["h1"] - 0.01, 0.5),
                arrowprops=dict(arrowstyle="-", color="#AAAAAA",
                                lw=1.5, linestyle="dashed"))
    ax.text((X["h1"] + X["h3"]) / 2, 0.52,
            "shared trunk  [512 \u2192 256 \u2192 128]",
            ha="center", fontsize=8, color="#888780", style="italic")

    ax.text((X["h3"] + X["actor"]) / 2, Y_MAX - 0.01, "actor head",
            ha="center", fontsize=7.5, color=ACTOR_COLOR, style="italic")
    ax.text((X["h3"] + X["critic"]) / 2, Y_MIN + 0.03, "critic head",
            ha="center", fontsize=7.5, color=CRITIC_COLOR, style="italic")

    # ── Column headers ────────────────────────────────────────────────────
    for x, lbl, bold in [
        (X["input"],  f"Observations\n({n_inputs})",       False),
        (X["h1"],     "",                                   False),
        (X["h2"],     "",                                   False),
        (X["h3"],     "",                                   False),
        (X["actor"],  f"Actor \u03c0(a|s)\n({n_actor})",   True),
        (X["critic"], "Critic V(s)\n(1)",                   True),
    ]:
        if lbl:
            ax.text(x, Y_MAX + 0.025, lbl,
                    ha="center", va="bottom", fontsize=9, color="#2C2C2A",
                    fontweight="bold" if bold else "normal")

    # ── Legend ────────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(color=GROUP_COLORS["pos"],    label="Position xyz (3)"),
        mpatches.Patch(color=GROUP_COLORS["vel"],    label="Velocity xyz (3)"),
        mpatches.Patch(color=GROUP_COLORS["scan_r"], label="Scan radius norm (1)"),
        mpatches.Patch(color=GROUP_COLORS["hazard"], label="Hazard dist+dir (3)"),
        mpatches.Patch(color=GROUP_COLORS["team"],   label="Teammate dist+dir (3)"),
        mpatches.Patch(color=GROUP_COLORS["uc_dir"], label="Uncov/RTB dir (2)"),
        mpatches.Patch(color=GROUP_COLORS["walls"],  label="Wall dists (4)"),
        mpatches.Patch(color=GROUP_COLORS["fuel"],   label="Fuel frac (1)"),
        mpatches.Patch(color=GROUP_COLORS["global"], label="Global state (2)"),
        mpatches.Patch(color=GROUP_COLORS["wind"],   label="Mean wind xy (2)"),
        mpatches.Patch(color=GROUP_COLORS["bldg"],   label="Building dist/dir/roof (4)"),
        Line2D([0], [0], color=ACTOR_COLOR,  lw=1.5, label="Actor path"),
        Line2D([0], [0], color=CRITIC_COLOR, lw=1.5, label="Critic path"),
    ]
    ax.legend(handles=legend_elements, loc="lower center",
              bbox_to_anchor=(0.5, -0.08), ncol=5,
              fontsize=8, frameon=False, handlelength=1.2)

    # ── Figure title ──────────────────────────────────────────────────────
    fig.suptitle(
        f"PPO Actor-Critic  |  {num_drones} UAV swarm  "
        f"|  Input: {n_inputs}  ({num_drones}\u00d720 + {n_inputs - num_drones * 20})  "
        f"\u2192  [512 \u2192 256 \u2192 128]  "
        f"\u2192  Actor: {n_actor} / Critic: 1",
        fontsize=9, y=1.01, color="#2C2C2A",
    )

    ax.set_xlim(-0.02, 1.10)
    ax.set_ylim(-0.14, 1.07)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
        print(f"  Network diagram saved -> {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig