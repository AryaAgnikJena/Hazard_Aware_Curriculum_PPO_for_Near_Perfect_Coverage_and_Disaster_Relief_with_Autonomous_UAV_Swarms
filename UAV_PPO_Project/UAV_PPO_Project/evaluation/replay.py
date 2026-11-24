"""
evaluation/replay.py

Save every test episode as a compact .npz and re-render it any time, without
the model or the env.

    rec = ReplayRecorder()
    rec.start(env, meta)              # right after env.reset()
    rec.step(env, info, reward)       # right after every env.step()
    rec.save("ep01.npz", extra={...}) # at episode end
    write_manifest(dir)               # manifest.csv: one row per replay

    python -m evaluation.replay PATH_OR_DIR [--mode top|3d] [--format gif|mp4]
                                [--fps 15] [--stride 3] [--out DIR]

What is stored (≈ 0.3 MB per 1000-step episode):
  static : buildings, building class, coverage points, checkpoints, victims,
           spawn zone, map size
  per step: drone xyz, bird xyz, fuel, dead, landed, lives, mean wind,
            coverage %, reward, collision flags (any/drone/bird/building), onsets
  first-covered step of every coverage cell / checkpoint / victim, so the
  covered mask at ANY step is rebuilt exactly without storing 1000 masks.
"""
import os
import sys
import csv
import time
import glob
import json
import math
import argparse
import subprocess
import numpy as np

_TAN15 = math.tan(math.radians(15.0))
BG, PANEL, EDGE = "#0d1117", "#161b22", "#30363d"
TXT, GREY = "#c9d1d9", "#8b949e"
GREEN, ORANGE, RED, YELLOW, TEAL = "#3fb950", "#f0883e", "#f85149", "#e3b341", "#39d353"
CLASS_COLORS = {0: "#5a8a6a", 1: "#5a7a9a", 2: "#9a5a7a"}


# ══════════════════════════════════════════════════════════════════════════
#  Recording
# ══════════════════════════════════════════════════════════════════════════

class ReplayRecorder:
    ROWS = ("pos", "birds", "fuel", "dead", "landed", "lives", "wind",
            "coverage", "reward", "hit", "onsets")

    def __init__(self):
        self._active = False

    def start(self, env, meta=None):
        try:
            import configs.config as config
            low = float(config.FUEL_LOW_FRAC)
            opt = float(config.OPTIMAL_SCAN_HEIGHT)
        except Exception:
            low, opt = 0.15, 20.0
        self.meta = dict(meta or {})
        self.meta.setdefault("scenario_type", getattr(env, "scenario_name", "random"))
        self.meta.setdefault("scenario_seed", getattr(env, "scenario_seed", None))
        self.static = dict(
            buildings=np.asarray(env.buildings, np.float32),
            building_class=np.asarray(getattr(env, "building_class_ids", []), np.int16),
            coverage_points=np.asarray(env.coverage_points, np.float32),
            checkpoints=np.asarray(env.checkpoints, np.float32),
            victims=np.asarray(env.victims, np.float32).reshape(-1, 2),
            spawn_zone=np.asarray(env._spawn_zone, np.float32),
            map_size=np.float32(env.map_size),
            max_height=np.float32(env.max_height),
            fuel_low_frac=np.float32(low),
            optimal_height=np.float32(opt),
        )
        self.cov_first = np.full(len(env.coverage_points), -1, np.int32)
        self.cp_first = np.full(len(env.checkpoints), -1, np.int32)
        self.resc_first = np.full(len(env.victims), -1, np.int32)
        self._rows = {k: [] for k in self.ROWS}
        self._t = -1
        self._active = True
        self._grab(env, None, 0.0)                    # frame 0 = initial state

    def step(self, env, info=None, reward=0.0):
        if self._active:
            self._grab(env, info, reward)

    def _grab(self, env, info, reward):
        self._t += 1
        t = self._t
        self.cov_first[env.covered & (self.cov_first < 0)] = t
        cpv = np.asarray(env.checkpoint_visited_mask, bool)
        self.cp_first[cpv & (self.cp_first < 0)] = t
        rs = np.asarray(env.rescued, bool)
        self.resc_first[rs & (self.resc_first < 0)] = t
        info = info or {}
        r = self._rows
        r["pos"].append(np.asarray(env.positions, np.float32).copy())
        r["birds"].append(np.asarray(env.birds, np.float32).reshape(-1, 3).copy())
        r["fuel"].append(np.asarray(env.fuel, np.float32).copy())
        r["dead"].append(np.asarray(env._dead, bool).copy())
        r["landed"].append(np.asarray(env._landed, bool).copy())
        r["lives"].append(np.minimum(env._lives, 99).astype(np.int16))
        r["wind"].append(np.asarray(env.wind_mean[:2], np.float32).copy())
        r["coverage"].append(float(env.get_coverage_percent()))
        r["reward"].append(float(reward))
        r["hit"].append([bool(info.get("collision_hit", False)),
                         bool(info.get("hit_drone", False)),
                         bool(info.get("hit_bird", False)),
                         bool(info.get("hit_building", False))])
        r["onsets"].append(int(info.get("collision_onsets", 0)))

    @property
    def active(self):
        return self._active

    def snapshot(self, path, extra=None):
        """Save the episode SO FAR (marked incomplete) and keep recording."""
        return self.save(path, dict(extra or {}, incomplete=True), finish=False)

    def save(self, path, extra=None, finish=True):
        if not self._active:
            return None
        arr = {}
        for k, v in self._rows.items():
            dt = {"coverage": np.float32, "reward": np.float32, "onsets": np.int16,
                  "hit": bool}.get(k)
            arr[k] = np.asarray(v, dtype=dt) if dt else np.stack(v)
        meta = dict(self.meta)
        meta.update(
            frames=int(len(arr["pos"])),
            steps=int(len(arr["pos"]) - 1),
            final_coverage=round(float(arr["coverage"][-1]), 2),
            **{"return": round(float(arr["reward"].sum()), 1)},
            drones_dead=int(arr["dead"][-1].sum()),
            drones_landed=int(arr["landed"][-1].sum()),
            collision_steps=int(arr["hit"][:, 0].sum()),
            collision_events=int(arr["onsets"].sum()),
            rescued=int((self.resc_first >= 0).sum()),
        )
        if extra:
            meta.update(extra)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(path, **self.static, **arr,
                            cov_first=self.cov_first, cp_first=self.cp_first,
                            resc_first=self.resc_first,
                            meta=np.array(json.dumps(meta, default=str)))
        if finish:
            self._active = False
        return path


def load_replay(path):
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(str(d["meta"]))
    return d


def write_manifest(replay_dir):
    """manifest.csv: one row per replay file (reads only the metadata)."""
    files = sorted(glob.glob(os.path.join(replay_dir, "*.npz")))
    cols = ["file", "scenario_type", "scenario_seed", "steps", "final_coverage",
            "return", "drones_dead", "drones_landed", "collision_steps",
            "collision_events", "rescued"]
    out = os.path.join(replay_dir, "manifest.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for p in files:
            with np.load(p, allow_pickle=False) as z:
                m = json.loads(str(z["meta"]))
            w.writerow([os.path.basename(p)] + [m.get(c, "") for c in cols[1:]])
    print(f"  Replay manifest → {out}  ({len(files)} replays)")
    return out


# ══════════════════════════════════════════════════════════════════════════
#  Rendering  (independent of the global matplotlib backend)
# ══════════════════════════════════════════════════════════════════════════

def _render_impl(path, out, mode, fps, stride, use_mp4, dpi, trail):
    """Draw and write one replay (paths / skipping / locking are handled by render_replay)."""
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
    import matplotlib.patches as mpatches
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    d = load_replay(path)
    meta = d["meta"]
    pos, birds, fuel = d["pos"], d["birds"], d["fuel"]
    dead, landed, cov, hit = d["dead"], d["landed"], d["coverage"], d["hit"]
    T, n = len(pos), pos.shape[1]
    ms, maxh = float(d["map_size"]), float(d["max_height"])
    B, cls = d["buildings"], d["building_class"]
    cpts, cov_first = d["coverage_points"], d["cov_first"]
    chk, chk_first = d["checkpoints"], d["cp_first"]
    vic, vic_first = d["victims"], d["resc_first"]
    sz = d["spawn_zone"]
    fuel0 = float(max(fuel[0].max(), 1.0))
    low = float(d["fuel_low_frac"])
    has_birds = birds.shape[1] > 0
    frames = list(range(0, T, max(1, int(stride))))
    if frames[-1] != T - 1:
        frames.append(T - 1)
    cols = matplotlib.colormaps["jet"](np.linspace(0, 1, n))
    name = f"{meta.get('scenario_type', '?')}#{meta.get('scenario_seed', '?')}"
    is3d = (mode == "3d")

    fig = Figure(figsize=(12, 6.4), dpi=dpi, facecolor=BG)
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1], hspace=0.32, wspace=0.12,
                          left=0.04, right=0.97, top=0.92, bottom=0.08)
    ax = fig.add_subplot(gs[:, 0], projection="3d" if is3d else None)
    ax_c, ax_f = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    for a in (ax_c, ax_f):
        a.set_facecolor(PANEL)
        a.tick_params(colors=GREY, labelsize=7)
        for sp in a.spines.values():
            sp.set_edgecolor(EDGE)
    ax.set_facecolor(BG if not is3d else PANEL)
    ax.tick_params(colors=GREY, labelsize=7)

    # ── right-hand panels (shared) ───────────────────────────────────────
    xs = np.arange(T)
    ax_c.plot(xs, cov, color=EDGE, lw=1.2)
    rev, = ax_c.plot([], [], color=GREEN, lw=1.8)
    cur = ax_c.axvline(0, color="#f0f6fc", lw=0.8, alpha=0.6)
    ax_c.axhline(95, color=ORANGE, ls="--", lw=1)
    ax_c.set_xlim(0, max(T - 1, 1)); ax_c.set_ylim(0, 105)
    ax_c.set_title("Coverage %", color="#f0f6fc", fontsize=9)
    ax_c.set_xlabel("Step", color=GREY, fontsize=8)
    bars = ax_f.barh(range(n), fuel[0], color=cols, height=0.6)
    btxt = [ax_f.text(fuel0 * 0.02, i, "", va="center", ha="left", fontsize=8,
                      color="#ffffff", fontweight="bold") for i in range(n)]
    ax_f.axvline(fuel0 * low, color=RED, ls="--", lw=0.8)
    ax_f.set_xlim(0, fuel0); ax_f.set_ylim(n - 0.4, -0.6)
    ax_f.set_yticks(range(n)); ax_f.set_yticklabels([f"D{i + 1}" for i in range(n)],
                                                    color=TXT, fontsize=8)
    ax_f.set_title("Fuel / status", color="#f0f6fc", fontsize=9)

    def upd_side(t):
        rev.set_data(xs[:t + 1], cov[:t + 1])
        cur.set_xdata([t, t])
        for i in range(n):
            f = float(fuel[t, i])
            bars[i].set_width(f)
            if dead[t, i]:
                bars[i].set_facecolor(RED); btxt[i].set_text("DEAD")
            elif landed[t, i]:
                bars[i].set_facecolor("#555555"); btxt[i].set_text("RTB")
            else:
                bars[i].set_facecolor(cols[i])
                btxt[i].set_text(f"{100 * f / fuel0:.0f}%")

    def status_color(t, i):
        if dead[t, i]:
            return RED
        if landed[t, i]:
            return "#8b949e"
        return cols[i]

    def header(t):
        alive = int(n - dead[t].sum())
        flash = "   ⚠ COLLISION" if hit[max(t - 1, 0):t + 1, 0].any() else ""
        return (f"{name}   step {t}/{T - 1}   coverage {cov[t]:.1f}%   "
                f"alive {alive}/{n}{flash}")

    unc_rgba = matplotlib.colors.to_rgba("#3a3f47", 0.8)
    cov_rgba = matplotlib.colors.to_rgba(TEAL, 0.95)

    # ── left scene ───────────────────────────────────────────────────────
    if not is3d:
        ax.set_xlim(0, ms); ax.set_ylim(0, ms); ax.set_aspect("equal")
        for sp in ax.spines.values():
            sp.set_edgecolor(EDGE)
        ax.add_patch(mpatches.Rectangle((sz[0], sz[1]), sz[2] - sz[0], sz[3] - sz[1],
                                        fill=True, fc="#3fb95015", ec=GREEN, lw=1.5))
        for k, b in enumerate(B):
            ax.add_patch(mpatches.Rectangle((b[0], b[1]), b[3], b[3], lw=0.5,
                                            ec="#aaaaaa55", alpha=0.75,
                                            fc=CLASS_COLORS.get(int(cls[k]) if k < len(cls) else 0)))
            if b[2] >= 9.0:
                ax.text(b[0] + b[3] / 2, b[1] + b[3] / 2, f"{b[2]:.0f}m", ha="center",
                        va="center", color="white", fontsize=6, alpha=0.9)
        sc_cov = ax.scatter(cpts[:, 0], cpts[:, 1], s=7, c=[unc_rgba] * len(cpts),
                            edgecolors="none", zorder=2)
        sc_chk = ax.scatter(chk[:, 0], chk[:, 1], s=26, marker="D", zorder=3,
                            c=[YELLOW] * len(chk), edgecolors="none") if len(chk) else None
        sc_vic = ax.scatter(vic[:, 0], vic[:, 1], s=60, marker="*", zorder=4,
                            c=[RED] * len(vic), edgecolors="none") if len(vic) else None
        sc_bird = ax.scatter([], [], s=34, marker="^", c=RED, zorder=5) if has_birds else None
        sc_dr = ax.scatter(pos[0, :, 0], pos[0, :, 1], s=70, zorder=7, c=cols,
                           edgecolors="#f0f6fc", linewidths=0.8)
        rings = [mpatches.Circle((0, 0), 1.0, fill=False, ec=cols[i], lw=0.8, alpha=0.6, ls="--")
                 for i in range(n)]
        for r_ in rings:
            ax.add_patch(r_)
        trails = [ax.plot([], [], color=cols[i], lw=1.0, alpha=0.55)[0] for i in range(n)]
        labels = [ax.text(0, 0, f"D{i + 1}", color="#f0f6fc", fontsize=7, zorder=8) for i in range(n)]
        ax.set_xlabel("X (m)", color=GREY, fontsize=8); ax.set_ylabel("Y (m)", color=GREY, fontsize=8)

        def upd_scene(t):
            m = (cov_first >= 0) & (cov_first <= t)
            sc_cov.set_facecolor(np.where(m[:, None], cov_rgba, unc_rgba))
            if sc_chk is not None:
                v = (chk_first >= 0) & (chk_first <= t)
                sc_chk.set_facecolor([("#6b5a1f" if x else YELLOW) for x in v])
            if sc_vic is not None:
                v = (vic_first >= 0) & (vic_first <= t)
                sc_vic.set_facecolor([(GREEN if x else RED) for x in v])
            if sc_bird is not None:
                sc_bird.set_offsets(birds[t][:, :2])
            sc_dr.set_offsets(pos[t][:, :2])
            sc_dr.set_facecolor([status_color(t, i) for i in range(n)])
            for i in range(n):
                p = pos[t, i]
                rings[i].center = (p[0], p[1])
                rings[i].set_radius(max(float(p[2]) * _TAN15, 0.5))
                rings[i].set_visible(not (dead[t, i] or landed[t, i]))
                lo = max(0, t - trail)
                trails[i].set_data(pos[lo:t + 1, i, 0], pos[lo:t + 1, i, 1])
                labels[i].set_position((p[0] + 1.2, p[1] + 1.2))
                labels[i].set_text(f"D{i + 1} z={p[2]:.0f}" + (" ✗" if dead[t, i] else ""))
    else:
        ax.set_xlim(0, ms); ax.set_ylim(0, ms); ax.set_zlim(0, maxh + 1)
        ax.view_init(elev=35, azim=-60)
        for k, b in enumerate(B):
            x, y, h, w = [float(v) for v in b]
            v = np.array([[x, y, 0], [x + w, y, 0], [x + w, y + w, 0], [x, y + w, 0],
                          [x, y, h], [x + w, y, h], [x + w, y + w, h], [x, y + w, h]])
            faces = [[v[0], v[1], v[2], v[3]], [v[4], v[5], v[6], v[7]],
                     [v[0], v[1], v[5], v[4]], [v[1], v[2], v[6], v[5]],
                     [v[2], v[3], v[7], v[6]], [v[3], v[0], v[4], v[7]]]
            ax.add_collection3d(Poly3DCollection(
                faces, facecolors=CLASS_COLORS.get(int(cls[k]) if k < len(cls) else 0),
                edgecolors="#6a6a6a", alpha=0.6, linewidth=0.3))
        ax.scatter(cpts[:, 0], cpts[:, 1], np.zeros(len(cpts)), s=5, c="#5c5c5c",
                   depthshade=False, alpha=0.8)
        sc_cov = ax.scatter([], [], [], s=14, c=TEAL, depthshade=False, alpha=0.95)
        sc_bird = ax.scatter([], [], [], s=34, c=RED, marker="^", depthshade=False) if has_birds else None
        dots = [ax.scatter([], [], [], s=60, color=cols[i], depthshade=False) for i in range(n)]
        trails = [ax.plot([], [], [], color=cols[i], lw=1.0, alpha=0.55)[0] for i in range(n)]

        def upd_scene(t):
            m = (cov_first >= 0) & (cov_first <= t)
            pts = cpts[m]
            sc_cov._offsets3d = (pts[:, 0], pts[:, 1], np.zeros(len(pts)))
            if sc_bird is not None:
                sc_bird._offsets3d = (birds[t][:, 0], birds[t][:, 1], birds[t][:, 2])
            for i in range(n):
                p = pos[t, i]
                dots[i]._offsets3d = ([p[0]], [p[1]], [p[2]])
                dots[i].set_color(status_color(t, i))
                lo = max(0, t - trail)
                trails[i].set_data(pos[lo:t + 1, i, 0], pos[lo:t + 1, i, 1])
                trails[i].set_3d_properties(pos[lo:t + 1, i, 2])

    title = ax.set_title("", color="#f0f6fc", fontsize=10)

    def update(i):
        t = frames[i]
        upd_scene(t); upd_side(t)
        title.set_text(header(t))
        if i % 50 == 0:
            print(f"    frame {i + 1}/{len(frames)}")

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)
    writer = FFMpegWriter(fps=fps, bitrate=2500) if use_mp4 else PillowWriter(fps=fps)
    root, ext = os.path.splitext(out)
    tmp = root + ".part" + ext                      # write aside, publish atomically
    print(f"  Rendering {name} ({len(frames)} frames, {mode}) -> {out}", flush=True)
    anim.save(tmp, writer=writer, dpi=dpi)
    os.replace(tmp, out)
    return out


def render_replay(path, out=None, mode="top", fps=15, stride=3, fmt="gif",
                  dpi=70, trail=50, skip_existing=False):
    """Render one replay to GIF/MP4.  mode: "top" (fast, 2-D) or "3d".
    skip_existing: leave a finished file alone, and one another process is writing now."""
    from matplotlib.animation import FFMpegWriter
    base = os.path.splitext(os.path.basename(path))[0] + f"_{mode}"
    if out is None:
        out = os.path.dirname(os.path.abspath(path))
    if os.path.isdir(out) or not os.path.splitext(out)[1]:
        os.makedirs(out, exist_ok=True)
        out = os.path.join(out, base + "." + fmt)
    want_mp4 = (fmt == "mp4" or out.endswith(".mp4"))
    use_mp4 = want_mp4 and FFMpegWriter.isAvailable()
    if want_mp4 and not use_mp4:
        print("  ffmpeg not found -> writing GIF instead")
        out = os.path.splitext(out)[0] + ".gif"
    lock = out + ".lock"
    if skip_existing:
        if os.path.exists(out) and os.path.getsize(out) > 0:
            print(f"  skip (exists): {os.path.basename(out)}", flush=True)
            return out
        if os.path.exists(lock) and time.time() - os.path.getmtime(lock) < 900:
            print(f"  skip (being rendered): {os.path.basename(out)}", flush=True)
            return out
    open(lock, "w").close()
    try:
        return _render_impl(path, out, mode, fps, stride, use_mp4, dpi, trail)
    finally:
        try:
            os.remove(lock)
        except OSError:
            pass


def _render_one(args):
    path, kw = args
    try:
        return path, render_replay(path, **kw), None
    except Exception as exc:                      # one bad replay must not stop the rest
        return path, None, f"{type(exc).__name__}: {exc}"


def render_many(paths, workers=3, **kw):
    """Render several replays, `workers` at a time (separate processes)."""
    from concurrent.futures import ProcessPoolExecutor
    paths = list(paths)
    workers = max(1, min(int(workers), len(paths)))
    jobs = [(p, kw) for p in paths]
    done = 0
    if workers == 1:
        results = map(_render_one, jobs)
        for p, out, err in results:
            done += 1
            print(f"  [{done}/{len(paths)}] {os.path.basename(p)} -> {out or 'FAILED: ' + str(err)}")
        return
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for p, out, err in ex.map(_render_one, jobs):
            done += 1
            print(f"  [{done}/{len(paths)}] {os.path.basename(p)} -> {out or 'FAILED: ' + str(err)}",
                  flush=True)


def launch_background_render(replay_dir, mode="top", workers=2, stride=4):
    """Render every replay in `replay_dir` in a SEPARATE process so the caller
    (e.g. the live visualiser window) is never blocked.  Progress -> render.log."""
    _spawn([replay_dir, "--mode", mode, "--workers", str(workers), "--stride", str(stride)],
           os.path.join(replay_dir, "render.log"))
    print(f"  GIF catch-up render started in the background -> "
          f"{os.path.join(replay_dir, 'render.log')}")


MAX_ASYNC_RENDERS = 2        # concurrent background renders (each one is CPU/RAM heavy)


def _active_renders(folder):
    """Renders currently in progress in `folder` (fresh .lock files)."""
    now = time.time()
    return sum(1 for f in glob.glob(os.path.join(folder, "*.lock"))
               if now - os.path.getmtime(f) < 900)


def render_async(path, mode="top", stride=4):
    """Render ONE replay in its own background process (call it right after the
    episode is saved, so GIFs appear while the next epoch is still running).
    If MAX_ASYNC_RENDERS are already busy it does nothing: the end-of-run
    catch-up (launch_background_render) renders whatever is still missing."""
    folder = os.path.dirname(os.path.abspath(path))
    if _active_renders(folder) >= MAX_ASYNC_RENDERS:
        print("  (renderer busy - this GIF will be rendered by the end-of-run catch-up)")
        return False
    _spawn([path, "--mode", mode, "--workers", "1", "--stride", str(stride)],
           os.path.join(folder, "render.log"))
    return True


def _spawn(cli_args, log_path):
    log = open(log_path, "a")
    cmd = [sys.executable, "-m", "evaluation.replay"] + cli_args
    flags = 0x00000008 if os.name == "nt" else 0          # DETACHED_PROCESS on Windows
    subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, creationflags=flags,
                     cwd=os.getcwd())
    log.close()


def main():
    ap = argparse.ArgumentParser(description="Render saved replays to GIF/MP4")
    ap.add_argument("path", help="a .npz replay or a folder of them")
    ap.add_argument("--mode", choices=["top", "3d"], default="top")
    ap.add_argument("--format", choices=["gif", "mp4"], default="gif")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--stride", type=int, default=4, help="render every Nth step")
    ap.add_argument("--workers", type=int, default=3, help="replays rendered in parallel")
    ap.add_argument("--force", action="store_true", help="re-render even if the GIF already exists")
    ap.add_argument("--dpi", type=int, default=70)
    ap.add_argument("--out", default=None, help="output file or folder")
    a = ap.parse_args()
    paths = (sorted(glob.glob(os.path.join(a.path, "*.npz")))
             if os.path.isdir(a.path) else [a.path])
    render_many(paths, a.workers, out=a.out, mode=a.mode, fps=a.fps, stride=a.stride,
                fmt=a.format, dpi=a.dpi, skip_existing=not a.force)


if __name__ == "__main__":
    main()
