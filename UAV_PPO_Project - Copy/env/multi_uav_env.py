"""
env/multi_uav_env.py   (v2)

Changes vs v1
─────────────
• ONE shared step() for all phases (phases only differ by what they spawn).
• Collision accounting is event-based:
    onset penalty (new contact) + small per-step penalty, per KIND
    (drone-drone / bird / building).  info reports hit flags per kind,
    `collision_onsets`, and a per-step reward breakdown `r_comp`.
• Reward is no longer saturated at -500 when drones touch.
• Wind: relative-airspeed drag (config.WIND_MODEL) and mean wind is part of
  the observation (config.OBS_INCLUDE_WIND).
• Landed drones are frozen in place.
• RTB: optional "must be inside spawn zone" mode; `landed_in_zone` reported.
• Coverage is read AFTER the coverage update (was one step stale).
• Uses self.np_random everywhere → seeding works.

v3: hover-centred vertical action (a_z = 0 holds altitude) and the
    piecewise height profile y(h) as a bounded penalty / coverage multiplier.

Observation per drone (20):
  pos xyz(3) · vel xyz(3) · scan_r(1) · hazard dist(1)+dir(2) ·
  teammate dist(1)+dir(2) · uncovered/RTB dir(2) · walls(4) · fuel(1)
Global: coverage frac, step frac, [wind x, wind y]   → 5·20 + 2 (+2)
"""
import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import configs.config as config

# ── Physics constants ──────────────────────────────────────────────────────
MASS       = 1.5
MAX_THRUST = 20.0
GRAVITY    = 9.81
DT         = 0.1
DRAG       = 0.1

_TAN15 = math.tan(math.radians(15.0))

REWARD_COMPONENTS = (
    "coverage", "milestone", "checkpoint", "rescue", "progress",
    "height", "wall", "roof", "fuel", "oob",
    "drone_hit", "bird_hit", "building_hit",
    "ceiling", "sustained_height", "lazy",
    "rtb", "mission", "death", "clip",
)


def scan_radius(height: float) -> float:
    """r(h) = h × tan(15°), minimum 0.5 m."""
    return max(float(height) * _TAN15, 0.5)


def height_reward(h: float, h_opt: float, tol: float, sigma: float = None) -> float:
    if sigma is None:
        sigma = tol * 1.5
    dz = abs(h - h_opt)
    if dz <= tol:
        return 1.0
    return float(math.exp(-0.5 * ((dz - tol) / sigma) ** 2))


def height_profile(h: float) -> float:
    """
    y(h): 1 at the optimal height, 0 on the ground and at HEIGHT_PROFILE_ZERO_HIGH,
    negative above that (floored at HEIGHT_SHAPE_MIN).
    """
    peak = float(getattr(config, "HEIGHT_PROFILE_PEAK", config.OPTIMAL_SCAN_HEIGHT))
    zero = float(getattr(config, "HEIGHT_PROFILE_ZERO_HIGH", peak + 5.0))
    dh = float(h) - peak
    width = peak if dh <= 0.0 else (zero - peak)
    y = 1.0 - (dh / max(width, 1e-6)) ** 2
    return max(y, float(getattr(config, "HEIGHT_SHAPE_MIN", -3.0)))


# ══════════════════════════════════════════════════════════════════════════════
class MultiUAVEnv(gym.Env):
    metadata        = {"render_modes": []}
    SPAWN_ZONE_SIZE = 36

    def __init__(self, building_count=0, num_birds=0, num_victims=0,
                 lethal=False, building_obs=False):
        super().__init__()

        self.num_drones       = config.NUM_DRONES
        self.map_size         = config.MAP_SIZE
        self.max_steps        = config.MAX_STEPS
        self.detection_radius = config.DETECTION_RADIUS
        self.rescue_radius    = config.RESCUE_RADIUS
        self.min_height       = config.MIN_HEIGHT
        self.max_height       = config.MAX_HEIGHT
        self.building_count   = building_count
        self.num_birds        = num_birds
        self.num_victims      = num_victims
        # scenario support (env/scenarios.py): defaults restored on every reset
        self._base_building_count = building_count
        self._base_birds          = num_birds
        self._pending_scenario    = None
        self._sc                  = {}
        self.scenario_name        = "random"
        self.scenario_seed        = None
        self._wind_speed          = float(getattr(config, "WIND_MEAN_SPEED", 6.0))
        self._bird_mult           = 1.0

        self.obs_include_wind = bool(getattr(config, "OBS_INCLUDE_WIND", False))
        self.obs_per_drone    = 20
        self.obs_building     = bool(building_obs)
        self.obs_bldg_per_drone = 4
        self._lethal          = bool(lethal)
        self._near_bonus      = float(getattr(
            config, "PHASE4_NEAR_BUILDING_BONUS" if lethal else "NEAR_BUILDING_COVERAGE_BONUS",
            2.5))
        self._episode_ends    = (str(getattr(config, "PHASE4_COLLISION_MODE",
                                             "drone_dies")) == "episode_ends")
        obs_dim = (self.num_drones * self.obs_per_drone + 2
                   + (2 if self.obs_include_wind else 0)
                   + (self.num_drones * self.obs_bldg_per_drone
                      if self.obs_building else 0))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.num_drones * 3,), dtype=np.float32)

        self._max_scan_r = config.MAX_HEIGHT * _TAN15
        self._wind_cap   = max(1.5 * float(getattr(config, "WIND_MEAN_SPEED", 6.0)), 1e-6)

        self._spawn_zone = (
            self.map_size * 0.3, self.map_size * 0.3,
            self.map_size * 0.7, self.map_size * 0.7,
        )

    # ── Scenario hook ──────────────────────────────────────────────────────

    def set_next_scenario(self, scenario):
        """Make the NEXT reset() build exactly this scenario (seed + generator
        overrides).  One-shot: later resets, e.g. VecEnv auto-resets, are random
        again.  Works through VecNormalize/DummyVecEnv because it is set on the
        raw env.  Pass None to clear."""
        self._pending_scenario = scenario

    # ── Spawn zone ─────────────────────────────────────────────────────────

    def _choose_spawn_zone(self):
        sz, m = self.SPAWN_ZONE_SIZE, 2
        corners = [
            (m, m),
            (self.map_size - m - sz, m),
            (m, self.map_size - m - sz),
            (self.map_size - m - sz, self.map_size - m - sz),
        ]
        cx, cy = corners[int(self.np_random.integers(4))]
        return float(cx), float(cy), float(cx + sz), float(cy + sz)

    def _spawn_centre(self) -> np.ndarray:
        x0, y0, x1, y1 = self._spawn_zone
        return np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float32)

    # ── Buildings ──────────────────────────────────────────────────────────

    def _make_buildings(self):
        rng = self.np_random
        if self.building_count == 0 and "bldg_mix" not in self._sc:
            self.building_class_ids = np.zeros(0, dtype=int)
            return np.zeros((0, 4), dtype=np.float32)

        x0, y0, x1, y1 = self._spawn_zone
        classes = [
            (config.BUILDING_SMALL_COUNT,
             config.BUILDING_SMALL_H_MIN,  config.BUILDING_SMALL_H_MAX,
             config.BUILDING_SMALL_W_MIN,  config.BUILDING_SMALL_W_MAX),
            (config.BUILDING_MEDIUM_COUNT,
             config.BUILDING_MEDIUM_H_MIN, config.BUILDING_MEDIUM_H_MAX,
             config.BUILDING_MEDIUM_W_MIN, config.BUILDING_MEDIUM_W_MAX),
            (config.BUILDING_LARGE_COUNT,
             config.BUILDING_LARGE_H_MIN,  config.BUILDING_LARGE_H_MAX,
             config.BUILDING_LARGE_W_MIN,  config.BUILDING_LARGE_W_MAX),
        ]
        sc = self._sc
        mix = sc.get("bldg_mix")
        if mix is not None:
            counts = [int(m) for m in mix]
        else:
            total_cfg = max(sum(c[0] for c in classes), 1)
            counts = [max(0, round(c[0] / total_cfg * self.building_count))
                      for c in classes]
            counts[0] = max(0, counts[0] + self.building_count - sum(counts))

        layout = sc.get("layout", "random")
        ms = float(self.map_size)
        centres = []
        if layout == "clustered":                   # 4 cluster centres away from spawn
            tries = 0
            while len(centres) < 4 and tries < 200:
                tries += 1
                c = rng.uniform(15.0, ms - 15.0, 2)
                if not (x0 - 12 < c[0] < x1 + 12 and y0 - 12 < c[1] < y1 + 12):
                    centres.append(c)
            if not centres:
                centres = [np.array([ms * 0.2, ms * 0.8])]
        PITCH = 18.0
        lattice = []
        if layout == "canyon":                      # lattice → streets of PITCH − w metres
            n_ax = int((ms - 2.0) // PITCH)
            off  = rng.uniform(1.0, 3.0)
            lattice = [(off + i * PITCH, off + j * PITCH)
                       for i in range(n_ax) for j in range(n_ax)]
            rng.shuffle(lattice)

        def _xy(bw):
            if layout == "clustered":
                c = centres[int(rng.integers(len(centres)))]
                bx, by = rng.normal(c, 7.0)
            elif layout == "canyon" and lattice:
                bx, by = lattice.pop()
                bx += rng.uniform(0.0, 1.0)
                by += rng.uniform(0.0, 1.0)
            elif layout == "perimeter":
                inset = rng.uniform(1.0, 5.0)
                along = rng.uniform(1.0, ms - bw - 1.0)
                side = int(rng.integers(4))
                bx, by = ((inset, along) if side == 0 else
                          (ms - bw - inset, along) if side == 1 else
                          (along, inset) if side == 2 else
                          (along, ms - bw - inset))
            else:
                bx = rng.uniform(1.0, max(ms - bw - 1.0, 1.5))
                by = rng.uniform(1.0, max(ms - bw - 1.0, 1.5))
            hi = max(ms - bw - 1.0, 1.5)
            return float(np.clip(bx, 1.0, hi)), float(np.clip(by, 1.0, hi))

        accepted, class_ids = [], []
        for cls_idx, (cnt, (_, h_lo, h_hi, w_lo, w_hi)) in enumerate(
                zip(counts, classes)):
            placed, attempts, max_att = 0, 0, max(cnt * 40, 80)
            while placed < cnt and attempts < max_att:
                attempts += 1
                bh = rng.uniform(h_lo, h_hi)
                bw = rng.uniform(w_lo, w_hi)
                bx, by = _xy(bw)
                if bx - 1 < x1 and bx + bw + 1 > x0 \
                        and by - 1 < y1 and by + bw + 1 > y0:
                    continue                                # overlaps spawn zone
                accepted.append([bx, by, bh, bw])
                class_ids.append(cls_idx)
                placed += 1

        if accepted:
            self.building_class_ids = np.array(class_ids, dtype=int)
            return np.array(accepted, dtype=np.float32)
        self.building_class_ids = np.zeros(0, dtype=int)
        return np.zeros((0, 4), dtype=np.float32)

    # ── Safe spawn ─────────────────────────────────────────────────────────

    def _safe_spawn(self):
        x0, y0, x1, y1 = self._spawn_zone
        grid_side = int(np.ceil(np.sqrt(self.num_drones)))
        sx = (x1 - x0) / (grid_side + 1)
        sy = (y1 - y0) / (grid_side + 1)
        kill_ceiling = getattr(self, "_kill_alt", self.max_height)
        safe_z = np.clip(
            min(config.OPTIMAL_SCAN_HEIGHT, kill_ceiling - 3.0),
            self.min_height, self.max_height - 1.0)
        positions = np.zeros((self.num_drones, 3))
        for i in range(self.num_drones):
            row, col = i // grid_side, i % grid_side
            positions[i] = [x0 + (col + 1) * sx,
                            y0 + (row + 1) * sy,
                            safe_z + i * 0.3]
        return positions

    # ── Reset ──────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        pend, self._pending_scenario = self._pending_scenario, None
        if pend is None and options and options.get("scenario"):
            pend = options["scenario"]
        if pend is not None:
            seed = pend["seed"]
            self._sc           = dict(pend.get("params", {}))
            self.scenario_name = pend.get("type", "custom")
            self.scenario_seed = pend["seed"]
        else:                                    # ordinary random episode
            self._sc, self.scenario_name, self.scenario_seed = {}, "random", seed
        self.building_count = int(self._sc.get("n_buildings", self._base_building_count))
        self.num_birds      = int(self._sc.get("n_birds", self._base_birds))
        self._wind_speed    = float(self._sc.get(
            "wind_speed", getattr(config, "WIND_MEAN_SPEED", 6.0)))
        self._bird_mult     = float(self._sc.get("bird_speed_mult", 1.0))

        super().reset(seed=seed)
        rng = self.np_random
        n   = self.num_drones

        self.step_count = 0
        self._spawn_zone = self._choose_spawn_zone()
        self.buildings   = self._make_buildings()

        if len(self.buildings) > 0:
            self._max_bh = float(np.max(self.buildings[:, 2]))
            bx, by, bw = (self.buildings[:, 0], self.buildings[:, 1],
                          self.buildings[:, 3])
            self._bldg_centres = np.column_stack(
                [bx + bw / 2, by + bw / 2]).astype(np.float32)
        else:
            self._max_bh = 0.0
            self._bldg_centres = np.zeros((0, 2), dtype=np.float32)

        self._kill_alt = max(
            self._max_bh + config.HEIGHT_KILL_MARGIN,
            config.OPTIMAL_SCAN_HEIGHT + config.SCAN_HEIGHT_TOL + 2.0)
        self._max_bird_z = max(self._max_bh, config.MIN_BIRD_Z + 1.0)

        self.positions  = self._safe_spawn()
        self.velocities = np.zeros((n, 3), dtype=np.float32)

        wdir = rng.uniform(0, 2 * math.pi)
        wspd = self._wind_speed
        self.wind      = np.array([math.cos(wdir) * wspd,
                                   math.sin(wdir) * wspd, 0.0], dtype=np.float32)
        self.wind_mean = self.wind[:2].copy()

        self.fuel            = np.full(n, float(config.DRONE_FUEL_STEPS))
        self._landed         = np.zeros(n, dtype=bool)
        self._landed_in_zone = np.zeros(n, dtype=bool)
        self._dead           = np.zeros(n, dtype=bool)
        if not self._lethal:
            lives0 = 10 ** 9
        elif self._episode_ends:
            lives0 = 1
        else:
            lives0 = max(int(getattr(config, "COLLISION_LIVES", 1)), 1)
        self._lives = np.full(n, lives0, dtype=np.int64)

        if self.num_birds > 0:
            nb = self.num_birds
            self.birds = rng.uniform(0, self.map_size, (nb, 3)).astype(np.float32)
            self.birds[:, 2] = rng.uniform(
                config.MIN_BIRD_Z, self._max_bird_z, nb)
            spd = config.BIRD_SPEED * self._bird_mult
            self.bird_vel = rng.uniform(-spd, spd, (nb, 3)).astype(np.float32)
            self.bird_vel[:, 2] *= 0.05
        else:
            self.birds    = np.zeros((0, 3), dtype=np.float32)
            self.bird_vel = np.zeros((0, 3), dtype=np.float32)

        if self.num_victims > 0:
            self.victims = rng.uniform(10, self.map_size - 10,
                                       (self.num_victims, 2))
            self.rescued = np.zeros(self.num_victims, dtype=bool)
        else:
            self.victims = np.zeros((0, 2))
            self.rescued = np.zeros(0, dtype=bool)

        grid = config.GRID_SIZE
        axis = (np.linspace(0, self.map_size, grid, endpoint=False)
                + self.map_size / grid / 2)
        gx, gy = np.meshgrid(axis, axis)
        self.coverage_points = np.column_stack(
            [gx.ravel(), gy.ravel()]).astype(np.float32)
        self.covered = np.zeros(len(self.coverage_points), dtype=bool)

        n_cp    = int(getattr(config, "NUM_CHECKPOINTS", 25))
        cp_side = max(2, int(np.ceil(np.sqrt(n_cp))))
        cp_ax   = np.linspace(5, self.map_size - 5, cp_side)
        cpx, cpy = np.meshgrid(cp_ax, cp_ax)
        self.checkpoints = np.column_stack(
            [cpx.ravel(), cpy.ravel()])[:n_cp].astype(np.float32)
        self.checkpoint_visited_mask = np.zeros(len(self.checkpoints), dtype=bool)

        self._milestones_paid = set()
        self._mission_paid    = False

        self._height_ticks = np.zeros(n, dtype=int)
        self._lazy_ticks   = np.zeros(n, dtype=int)
        self._prev_xy      = self.positions[:, :2].copy()

        # contact state (for onset detection)
        self._pair_contact = np.zeros((n, n), dtype=bool)
        self._bird_contact = np.zeros(n, dtype=bool)
        self._bldg_contact = np.zeros(n, dtype=bool)
        self._wall_hits    = 0

        return self._get_obs(), {}

    # ── Physics ────────────────────────────────────────────────────────────

    def _tick_wind(self):
        rng        = self.np_random
        drift      = float(getattr(config, "WIND_DRIFT",      0.97))
        gust_amp   = float(getattr(config, "WIND_GUST_AMP",   0.3))
        mean_speed = self._wind_speed

        mean_gust = rng.uniform(-mean_speed * 0.06, mean_speed * 0.06,
                                2).astype(np.float32)
        self.wind_mean = (self.wind_mean + mean_gust).astype(np.float32)
        cur = float(np.linalg.norm(self.wind_mean))
        lo, hi = mean_speed * 0.3, mean_speed * 1.5
        if cur > hi:
            self.wind_mean *= hi / (cur + 1e-6)
        elif 1e-6 < cur < lo:
            self.wind_mean *= lo / cur

        gust = rng.uniform(-gust_amp, gust_amp, 3).astype(np.float32)
        gust[2] *= 0.15
        mean_wind = np.array([self.wind_mean[0], self.wind_mean[1], 0.0],
                             dtype=np.float32)
        self.wind = (drift * self.wind
                     + (1 - drift) * (mean_wind + gust)).astype(np.float32)

    def _physics_step(self, action):
        n = self.num_drones
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(n, 3),
                         -1.0, 1.0)
        thrust  = action * MAX_THRUST
        if getattr(config, "ACTION_Z_HOVER_CENTERED", False):
            hover = MASS * GRAVITY
            thrust[:, 2] = hover + action[:, 2] * (MAX_THRUST - hover)
        gravity = np.array([0.0, 0.0, -MASS * GRAVITY])
        vel     = self.velocities.astype(np.float64)

        if str(getattr(config, "WIND_MODEL", "relative_drag")) == "legacy_force":
            drag   = -DRAG * np.abs(vel) * vel
            wind_f = self.wind.astype(np.float64) * MASS
        else:
            rel    = vel - self.wind.astype(np.float64)[None, :]
            drag   = -DRAG * np.abs(rel) * rel
            wind_f = 0.0

        # drone-drone repulsion (active drones only)
        rep_r = config.DRONE_COLLISION_RADIUS * 3.0
        gain  = float(getattr(config, "DRONE_REPULSION_GAIN", 4.0))
        diff  = self.positions[:, None, :] - self.positions[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        active  = ~self._off
        pair_ok = active[:, None] & active[None, :]
        mask    = (dists < rep_r) & (dists > 0) & pair_ok
        safe_d  = np.where(np.isfinite(dists) & (dists > 0), dists, 1.0)
        rv      = diff / safe_d[:, :, None]
        st      = np.where(mask, gain * np.exp(-dists / (rep_r * 0.25)), 0.0)
        rep_f   = (rv * st[:, :, None]).sum(axis=1) * MASS
        rep_f[:, 2] *= float(getattr(config, "DRONE_REPULSION_Z_SCALE", 1.0))

        accel = (thrust + gravity + drag + wind_f + rep_f) / MASS
        prev_pos = self.positions.copy()
        self.velocities += (accel * DT).astype(np.float32)
        self.positions  += self.velocities * DT
        self.positions[:, :2] = np.clip(self.positions[:, :2], 0, self.map_size)
        self.positions[:,  2] = np.clip(self.positions[:,  2],
                                        self.min_height, self.max_height)

        # landed drones stay where they are
        off = self._off
        if off.any():
            self.positions[off]  = prev_pos[off]
            self.velocities[off] = 0.0

        wall_hit = np.zeros(n, dtype=bool)
        for ax in range(2):
            lo = self.positions[:, ax] <= 0
            hi = self.positions[:, ax] >= self.map_size
            self.velocities[lo | hi, ax] *= -0.5
            wall_hit |= (lo | hi)
        self._wall_hits = int(np.sum(wall_hit & ~self._off))

    # ── Birds ──────────────────────────────────────────────────────────────

    def _update_birds(self):
        if len(self.birds) == 0:
            return
        rng  = self.np_random
        mask = rng.random(self.num_birds) < 0.02
        if mask.any():
            self.bird_vel[mask] += rng.uniform(
                -0.1, 0.1, (int(mask.sum()), 3)).astype(np.float32)
        spd = np.linalg.norm(self.bird_vel, axis=1, keepdims=True)
        spd = np.where(spd < 1e-6, 1.0, spd)
        self.bird_vel = (self.bird_vel / spd * config.BIRD_SPEED
                         * self._bird_mult).astype(np.float32)
        scale = float(getattr(config, "BIRD_TIME_SCALE", 1.0))
        self.birds = self.birds + self.bird_vel * DT * scale
        lo = self.birds < 0
        hi = self.birds > self.map_size
        self.birds    = np.where(lo, 0, np.where(hi, self.map_size,
                                                 self.birds)).astype(np.float32)
        self.bird_vel = np.where(lo | hi, -self.bird_vel,
                                 self.bird_vel).astype(np.float32)
        self.birds[:, 2] = np.clip(self.birds[:, 2],
                                   config.MIN_BIRD_Z, self._max_bird_z)

    # ── Fuel / RTB / mission ───────────────────────────────────────────────

    def _step_fuel(self) -> tuple:
        active   = ~self._off
        n_active = int(active.sum())
        self.fuel[active] -= 1.0
        empty = active & (self.fuel <= 0.0)
        self.fuel = np.maximum(self.fuel, 0.0)
        n_empty = int(empty.sum())
        penalty = -float(config.FUEL_EMPTY_PENALTY) * n_empty
        all_empty = (n_active > 0) and (n_empty == n_active)
        return penalty, all_empty

    def _check_rtb_landings(self, coverage: float) -> float:
        bonus   = 0.0
        thr     = float(getattr(config, "RTB_COVERAGE_THRESHOLD", 95.0))
        cov_high = coverage >= thr
        require = bool(getattr(config, "RTB_REQUIRES_POSITION", False))
        centre  = self._spawn_centre()
        for i in range(self.num_drones):
            if self._off[i]:
                continue
            fuel_frac = self.fuel[i] / config.DRONE_FUEL_STEPS
            if not (fuel_frac <= config.FUEL_LOW_FRAC or cov_high):
                continue
            in_zone = float(np.linalg.norm(
                self.positions[i, :2] - centre)) <= config.LANDING_RADIUS
            if require and not in_zone:
                continue
            self._landed[i]         = True
            self._landed_in_zone[i] = in_zone
            bonus += config.RETURN_REWARD + config.FUEL_RETURN_REWARD * fuel_frac
        return bonus

    @property
    def _off(self) -> np.ndarray:
        """Drones that no longer fly: landed OR dead."""
        return self._landed | self._dead

    def _all_off(self) -> bool:
        return bool(np.all(self._off))

    def _mission_check(self, coverage: float) -> tuple:
        if coverage >= 95.0 and not self._mission_paid:
            self._mission_paid = True
            base = float(getattr(config, "MISSION_COMPLETE_BASE", 5000.0))
            return base / max(self.step_count, 1), True
        return 0.0, False

    def _episode_end_flags(self, all_empty: bool, episode_ends: bool = False) -> tuple:
        fuel_term  = bool(all_empty
                          and getattr(config, "FUEL_EMPTY_TERMINATE", False))
        terminated = fuel_term or self._all_off() or episode_ends
        truncated  = self.step_count >= self.max_steps
        return terminated, truncated, fuel_term

    # ── Rewards ────────────────────────────────────────────────────────────

    def _coverage_reward(self) -> float:
        reward = 0.0
        for i in range(self.num_drones):
            if self._off[i]:
                continue
            drone = self.positions[i]
            r  = scan_radius(drone[2])
            if getattr(config, "HEIGHT_PROFILE_FOR_COVERAGE", False):
                hr = max(height_profile(drone[2]), 0.0)
            else:
                hr = height_reward(drone[2], config.OPTIMAL_SCAN_HEIGHT,
                                   config.SCAN_HEIGHT_TOL)
            dists  = np.linalg.norm(self.coverage_points - drone[:2], axis=1)
            inside = dists < r
            new_mask = inside & ~self.covered
            self.covered |= inside
            n_new = int(new_mask.sum())
            if n_new == 0:
                continue
            if len(self._bldg_centres) > 0:
                new_pts = self.coverage_points[new_mask]
                d = np.min(np.linalg.norm(
                    new_pts[:, None, :] - self._bldg_centres[None, :, :],
                    axis=2), axis=1)
                near = float(np.sum(d < 15.0))
                reward += ((n_new - near) + near * self._near_bonus) * hr
            else:
                reward += n_new * hr
        return config.COVERAGE_REWARD * reward

    def _milestone_reward(self, cov: float) -> float:
        r = 0.0
        for thr, val in ((30, 50), (50, 75), (70, 100), (80, 125), (95, 200)):
            if cov > thr and thr not in self._milestones_paid:
                self._milestones_paid.add(thr)
                r += val
        return r

    def _rescue_reward(self) -> float:
        if len(self.victims) == 0:
            return 0.0
        r = 0.0
        dists = np.linalg.norm(
            self.positions[:, None, :2] - self.victims[None, :, :], axis=2)
        for i in range(len(self.victims)):
            if not self.rescued[i] and np.any(dists[:, i] < self.rescue_radius):
                self.rescued[i] = True
                r += config.NORMAL_REWARD
        return r

    def _update_checkpoint_hits(self) -> float:
        if len(self.checkpoints) == 0:
            return 0.0
        cp_r   = float(getattr(config, "CHECKPOINT_REWARD", 10.0))
        reward = 0.0
        dists  = np.linalg.norm(
            self.positions[:, None, :2] - self.checkpoints[None, :, :], axis=2)
        for i in range(len(self.checkpoints)):
            if (not self.checkpoint_visited_mask[i]
                    and np.any(dists[:, i] < self.detection_radius)):
                self.checkpoint_visited_mask[i] = True
                reward += cp_r
        return reward

    def _height_penalty(self) -> float:
        """−SCALE·(1 − y(h)) per active drone  (0 at the optimal height)."""
        scale = float(getattr(config, "HEIGHT_SHAPE_SCALE", 6.0))
        total = 0.0
        for i in range(self.num_drones):
            if self._off[i]:
                continue
            total -= scale * (1.0 - height_profile(float(self.positions[i, 2])))
        return total

    def _wall_hug_penalty(self) -> float:
        margin  = float(getattr(config, "WALL_HUG_MARGIN", 5.0))
        pen_max = float(getattr(config, "WALL_HUG_PENALTY", 2.0))
        ms = float(self.map_size)
        x, y = self.positions[:, 0], self.positions[:, 1]
        d   = np.stack([x, ms - x, y, ms - y], axis=1)
        pen = pen_max * np.clip(1.0 - d / margin, 0.0, 1.0)
        return -float(pen[~self._off].sum())

    def _oob_penalty(self) -> float:
        return -float(getattr(config, "OOB_PENALTY", 3.0)) * self._wall_hits

    def _roof_proximity_penalty(self) -> float:
        if len(self.buildings) == 0:
            return 0.0
        margin  = float(getattr(config, "P2_ROOF_MARGIN",  2.0))
        pen_max = float(getattr(config, "P2_ROOF_PENALTY", 1.5))
        bx, by, bh, bw = (self.buildings[:, k] for k in range(4))
        total = 0.0
        for i in range(self.num_drones):
            if self._off[i]:
                continue
            d  = self.positions[i]
            cx = np.clip(d[0], bx, bx + bw)
            cy = np.clip(d[1], by, by + bw)
            dxy  = np.sqrt((d[0] - cx) ** 2 + (d[1] - cy) ** 2)
            near = dxy < config.BUILDING_RADIUS * 1.5
            for k in np.where(near)[0]:
                above = float(d[2]) - float(bh[k])
                if 0.0 <= above < margin:
                    total -= pen_max * (1.0 - above / margin)
        return total

    def _ceiling_penalty(self) -> tuple:
        """Per-step penalty for each active drone above the kill altitude."""
        ceiling = getattr(self, "_kill_alt", self.max_height)
        c_pen   = float(getattr(config, "P2_CEILING_PENALTY", 15.0))
        above   = (self.positions[:, 2] > ceiling) & ~self._off
        return -c_pen * int(above.sum()), bool(above.any())

    def _sustained_checks(self) -> tuple:
        """
        Sustained-height and lazy penalties (non-fatal, re-armed after firing).
        Returns (height_pen, lazy_pen, height_event, lazy_event).
        """
        h_ticks = int(getattr(config, "HEIGHT_KILL_TICKS", 100))
        h_pen   = float(getattr(config, "HEIGHT_SUSTAINED_PENALTY", 50.0))
        l_ticks = int(getattr(config, "LAZY_KILL_TICKS", 300))
        l_disp  = float(getattr(config, "LAZY_MIN_DISP", 0.05))
        l_pen   = float(getattr(config, "LAZY_KILL_PENALTY", 30.0))
        l_grace = int(getattr(config, "LAZY_GRACE_STEPS", 0))
        can_lazy = self.step_count >= l_grace
        ceiling  = getattr(self, "_kill_alt", self.max_height)

        hp = lp = 0.0
        h_event = l_event = False
        for i in range(self.num_drones):
            if self._off[i]:
                self._height_ticks[i] = 0
                self._lazy_ticks[i]   = 0
                self._prev_xy[i]      = self.positions[i, :2].copy()
                continue

            if float(self.positions[i, 2]) > ceiling:
                self._height_ticks[i] += 1
                if self._height_ticks[i] >= h_ticks:
                    hp -= h_pen
                    h_event = True
                    self._height_ticks[i] = 0
            else:
                self._height_ticks[i] = 0

            disp = float(np.linalg.norm(self.positions[i, :2] - self._prev_xy[i]))
            if disp < l_disp:
                self._lazy_ticks[i] += 1
                if can_lazy and self._lazy_ticks[i] >= l_ticks:
                    lp -= l_pen
                    l_event = True
                    self._lazy_ticks[i] = 0
            else:
                self._lazy_ticks[i] = 0
            self._prev_xy[i] = self.positions[i, :2].copy()
        return hp, lp, h_event, l_event

    # ── Collisions (onset + per-step, per kind) ────────────────────────────

    def _drone_drone_hits(self) -> tuple:
        """Returns (penalty, any_contact, n_onsets)."""
        act  = ~self._off
        pos  = self.positions
        dist = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
        pair_ok = np.triu(act[:, None] & act[None, :], k=1)

        radius  = config.DRONE_COLLISION_RADIUS
        contact = pair_ok & (dist < radius)
        onsets  = contact & ~self._pair_contact
        self._pair_contact = contact
        self._on_i_drone = onsets.any(axis=0) | onsets.any(axis=1)
        n_on = int(onsets.sum())

        penalty = -float(config.DRONE_HIT_ONSET_PENALTY) * n_on
        if contact.any():
            overlap = (radius - dist[contact]) / radius          # (0, 1]
            penalty -= float(config.DRONE_HIT_STEP_PENALTY) * float(
                np.sum(0.5 + overlap))

        soft_range = float(getattr(config, "DRONE_SOFT_BARRIER_RANGE", radius * 2))
        bk = float(getattr(config, "DRONE_SOFT_BARRIER_K", 8.0))
        bs = float(getattr(config, "DRONE_SOFT_BARRIER_STEEPNESS", 3.0))
        soft = pair_ok & (dist >= radius) & (dist < radius + soft_range)
        if soft.any():
            t = np.clip(1.0 - (dist[soft] - radius) / soft_range, 0, 1)
            penalty -= float(np.sum(bk * (np.exp(bs * t) - 1) / (np.exp(bs) - 1)))
        return penalty, bool(contact.any()), n_on

    def _bird_hits(self) -> tuple:
        n = self.num_drones
        if len(self.birds) == 0:
            self._bird_contact = np.zeros(n, dtype=bool)
            return 0.0, False, 0
        act = ~self._off
        R   = float(config.BIRD_RADIUS)
        rng_ = float(getattr(config, "BIRD_SOFT_RANGE", 4.0))
        bk   = float(getattr(config, "BIRD_SOFT_K", 6.0))
        bs   = float(getattr(config, "SOFT_BARRIER_STEEPNESS", 3.0))

        bd = np.linalg.norm(self.positions[:, None, :] - self.birds[None, :, :],
                            axis=2)                                   # (n, nb)
        contact = (bd < R).any(axis=1) & act
        onsets  = contact & ~self._bird_contact
        self._bird_contact = contact
        self._on_i_bird = onsets
        n_on = int(onsets.sum())

        penalty = (-float(config.BIRD_HIT_ONSET_PENALTY) * n_on
                   - float(config.BIRD_HIT_STEP_PENALTY) * int(contact.sum()))
        zone = (bd >= R) & (bd < R + rng_) & act[:, None]
        if zone.any():
            t = np.clip(1.0 - (bd[zone] - R) / rng_, 0, 1)
            penalty -= float(np.sum(bk * (np.exp(bs * t) - 1) / (np.exp(bs) - 1)))
        return penalty, bool(contact.any()), n_on

    def _building_hits(self) -> tuple:
        n = self.num_drones
        if len(self.buildings) == 0:
            self._bldg_contact = np.zeros(n, dtype=bool)
            return 0.0, False, 0
        act = ~self._off
        bx, by, bh, bw = (self.buildings[:, k] for k in range(4))
        px = self.positions[:, 0:1]
        py = self.positions[:, 1:2]
        pz = self.positions[:, 2:3]

        inside  = ((bx[None, :] <= px) & (px <= (bx + bw)[None, :])
                   & (by[None, :] <= py) & (py <= (by + bw)[None, :])
                   & (pz <= bh[None, :]))
        contact = inside.any(axis=1) & act
        onsets  = contact & ~self._bldg_contact
        self._bldg_contact = contact
        self._on_i_bldg = onsets
        n_on = int(onsets.sum())

        penalty = (-float(config.BUILDING_HIT_ONSET_PENALTY) * n_on
                   - float(config.BUILDING_HIT_STEP_PENALTY) * int(contact.sum()))

        # soft barrier — only buildings tall enough to matter (roof + 2 m)
        R_b  = float(config.BUILDING_RADIUS)
        rng_ = float(getattr(config, "BUILDING_SOFT_RANGE", 4.0))
        bk   = float(getattr(config, "BUILDING_SOFT_K", 6.0))
        bs   = float(getattr(config, "SOFT_BARRIER_STEEPNESS", 3.0))
        cxs = np.clip(px, bx[None, :], (bx + bw)[None, :])
        cys = np.clip(py, by[None, :], (by + bw)[None, :])
        dxy = np.sqrt((px - cxs) ** 2 + (py - cys) ** 2)
        relevant = (pz <= bh[None, :] + 2.0) & act[:, None]
        near = np.where(relevant, dxy, np.inf).min(axis=1)
        zone = (near < rng_ + R_b) & ~contact
        if zone.any():
            eff = np.maximum(near[zone], R_b)
            tt  = np.clip(1.0 - (eff - R_b) / rng_, 0, 1)
            penalty -= float(np.sum(bk * (np.exp(bs * tt) - 1) / (np.exp(bs) - 1)))
        return penalty, bool(contact.any()), n_on

    def _collision_step(self) -> dict:
        n = self.num_drones
        self._on_i_drone = np.zeros(n, dtype=bool)
        self._on_i_bird  = np.zeros(n, dtype=bool)
        self._on_i_bldg  = np.zeros(n, dtype=bool)
        p_d, h_d, o_d = self._drone_drone_hits()
        p_b, h_b, o_b = self._bird_hits()
        p_g, h_g, o_g = self._building_hits()
        return dict(pen_drone=p_d, pen_bird=p_b, pen_building=p_g,
                    hit_drone=h_d, hit_bird=h_b, hit_building=h_g,
                    onsets=o_d + o_b + o_g,
                    onset_i=self._on_i_drone | self._on_i_bird | self._on_i_bldg)

    def _apply_lives(self, onset_i: np.ndarray) -> int:
        """Each new contact costs the drone one life; 0 lives = dead.  Returns #new deaths."""
        if not self._lethal:
            return 0
        hit = onset_i & ~self._off
        if not hit.any():
            return 0
        self._lives[hit] -= 1
        newly = hit & (self._lives <= 0) & ~self._dead
        if newly.any():
            self._dead |= newly
            self.velocities[newly] = 0.0
        return int(newly.sum())

    # ── Targets / progress shaping ─────────────────────────────────────────

    def _clip_total(self, r: float) -> float:
        lim = float(getattr(config, "TOTAL_REWARD_CLIP", 300.0))
        return float(np.clip(r, -lim, lim))

    def _compute_targets(self) -> tuple:
        """Per-drone XY target (Voronoi-split nearest uncovered cell, or spawn
        centre when low on fuel).  Returns (targets[n,2], valid[n])."""
        n       = self.num_drones
        spawn_c = self._spawn_centre()
        tgt     = np.tile(spawn_c, (n, 1)).astype(np.float32)
        valid   = np.zeros(n, dtype=bool)

        uncovered = self.coverage_points[~self.covered]
        cand = uncovered
        if len(uncovered) > 0 and len(self._bldg_centres) > 0:
            safe_margin = float(getattr(config, "BUILDING_RADIUS", 3.0)) * 2.5
            c2b = np.min(np.linalg.norm(
                uncovered[:, None, :] - self._bldg_centres[None, :, :],
                axis=2), axis=1)
            safe = c2b > safe_margin
            if safe.any():
                cand = uncovered[safe]

        have_cells = len(cand) > 0
        active     = ~self._off
        xy         = self.positions[:, :2]
        use_vor    = bool(getattr(config, "USE_VORONOI_TARGETS", True))

        owner = d_all = None
        if have_cells:
            d_all = np.linalg.norm(cand[:, None, :] - xy[None, :, :], axis=2)
            exploring = active & (self.fuel / config.DRONE_FUEL_STEPS
                                  > config.FUEL_LOW_FRAC)
            if use_vor and int(exploring.sum()) > 1:
                d_exp = np.where(exploring[None, :], d_all, np.inf)
                owner = np.argmin(d_exp, axis=1)

        for i in range(n):
            if self._off[i]:
                continue
            if self.fuel[i] / config.DRONE_FUEL_STEPS <= config.FUEL_LOW_FRAC:
                valid[i] = True                      # heading home
                continue
            if not have_cells:
                continue
            if owner is not None:
                mine = np.where(owner == i)[0]
                j = (int(mine[np.argmin(d_all[mine, i])]) if len(mine)
                     else int(np.argmin(d_all[:, i])))
            else:
                j = int(np.argmin(d_all[:, i]))
            tgt[i]   = cand[j]
            valid[i] = True
        return tgt, valid

    def _progress_reward(self) -> float:
        scale = float(getattr(config, "PROGRESS_REWARD_SCALE", 0.0))
        if scale <= 0.0 or not hasattr(self, "_tgt_xy"):
            return 0.0
        total = 0.0
        for i in range(self.num_drones):
            if self._off[i] or not self._tgt_valid[i]:
                continue
            d_now = float(np.linalg.norm(self._tgt_xy[i] - self.positions[i, :2]))
            gain  = float(self._tgt_prev_dist[i]) - d_now
            total += scale * float(np.clip(gain, -1.0, 1.0))
        return total

    # ── Observation ────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        tgt, tgt_valid = self._compute_targets()
        self._tgt_xy        = tgt.copy()
        self._tgt_valid     = tgt_valid.copy()
        self._tgt_prev_dist = np.linalg.norm(
            tgt - self.positions[:, :2], axis=1).astype(np.float32)

        has_b = len(self.buildings) > 0
        if has_b:
            bx, by, bw = (self.buildings[:, 0], self.buildings[:, 1],
                          self.buildings[:, 3])
        team_detect_r = max(config.DRONE_COLLISION_RADIUS * 8.0,
                            self.detection_radius)
        obs = []
        for i in range(self.num_drones):
            if self._dead[i]:                      # dead drone: all-zero block
                obs.append(np.zeros(self.obs_per_drone, dtype=np.float32))
                continue
            drone  = self.positions[i]
            r_now  = scan_radius(drone[2])
            r_norm = r_now / max(self._max_scan_r, 1.0)
            detect_r = max(r_now * 2.0, self.detection_radius)

            hd = []
            if len(self.birds) > 0:
                bd  = np.linalg.norm(self.birds - drone, axis=1)
                ins = bd[bd <= detect_r]
                if len(ins):
                    hd.append(float(np.min(ins)))
            if has_b:
                cx = np.clip(drone[0], bx, bx + bw)
                cy = np.clip(drone[1], by, by + bw)
                cd = np.sqrt((drone[0] - cx) ** 2 + (drone[1] - cy) ** 2)
                ins = cd[cd <= detect_r]
                if len(ins):
                    hd.append(float(np.min(ins)))
            nearest_h   = min(hd) if hd else float(detect_r)
            hazard_norm = nearest_h / max(detect_r, 1.0)

            hazard_dir = np.zeros(2, dtype=np.float32)
            best = float(detect_r)
            if len(self.birds) > 0:
                bdx = np.linalg.norm(self.birds[:, :2] - drone[:2], axis=1)
                ib  = int(np.argmin(bdx))
                if bdx[ib] < detect_r:
                    best  = float(bdx[ib])
                    delta = self.birds[ib, :2] - drone[:2]
                    hazard_dir = (delta / (np.linalg.norm(delta) + 1e-6)
                                  ).astype(np.float32)
            if has_b:
                cd2 = np.sqrt((drone[0] - cx) ** 2 + (drone[1] - cy) ** 2)
                ib2 = int(np.argmin(cd2))
                if cd2[ib2] < best:
                    delta = np.array([cx[ib2] - drone[0], cy[ib2] - drone[1]])
                    hazard_dir = (delta / (np.linalg.norm(delta) + 1e-6)
                                  ).astype(np.float32)

            other_idx = [j for j in range(self.num_drones)
                         if j != i and not self._dead[j]]
            if other_idx:
                others = self.positions[other_idx]
                team_d = np.linalg.norm(others[:, :2] - drone[:2], axis=1)
                jn = int(np.argmin(team_d))
                nearest_team_d = float(team_d[jn])
                team_dist_norm = float(np.clip(nearest_team_d / team_detect_r, 0, 1))
                if nearest_team_d < team_detect_r:
                    delta = others[jn, :2] - drone[:2]
                    team_dir = (delta / (np.linalg.norm(delta) + 1e-6)
                                ).astype(np.float32)
                else:
                    team_dir = np.zeros(2, dtype=np.float32)
            else:
                team_dist_norm, team_dir = 1.0, np.zeros(2, dtype=np.float32)

            if tgt_valid[i] or self._landed[i]:
                delta  = tgt[i] - drone[:2]
                uc_dir = (delta / (np.linalg.norm(delta) + 1e-6)).astype(np.float32)
            else:
                uc_dir = np.zeros(2, dtype=np.float32)

            fuel_frac = self.fuel[i] / config.DRONE_FUEL_STEPS
            wall_dists = np.array([
                drone[0] / self.map_size,
                (self.map_size - drone[0]) / self.map_size,
                drone[1] / self.map_size,
                (self.map_size - drone[1]) / self.map_size,
            ], dtype=np.float32)

            obs.append(np.concatenate([
                (drone / self.map_size).astype(np.float32),
                self.velocities[i].astype(np.float32),
                np.array([r_norm], dtype=np.float32),
                np.array([hazard_norm], dtype=np.float32),
                hazard_dir,
                np.array([team_dist_norm], dtype=np.float32),
                team_dir,
                uc_dir,
                wall_dists,
                np.array([fuel_frac], dtype=np.float32),
            ]))

        glob = [self.get_coverage_percent() / 100.0,
                self.step_count / self.max_steps]
        if self.obs_include_wind:
            glob += list(np.clip(self.wind_mean / self._wind_cap, -1.0, 1.0))
        if self.obs_building:
            glob += self._building_obs()
        return np.concatenate([np.array(obs).flatten(), glob]).astype(np.float32)

    def _building_obs(self) -> list:
        """Per drone: nearest building dist/R, unit dir xy, roof clearance."""
        R      = float(getattr(config, "BUILDING_OBS_RADIUS", 20.0))
        margin = float(getattr(config, "ROOF_CLEARANCE_MARGIN", 2.0))
        has_b  = len(self.buildings) > 0
        if has_b:
            bx, by, bh, bw = (self.buildings[:, k] for k in range(4))
        out = []
        for i in range(self.num_drones):
            if self._dead[i] or not has_b:
                out += [1.0, 0.0, 0.0, 1.0]
                continue
            d  = self.positions[i]
            cx = np.clip(d[0], bx, bx + bw)
            cy = np.clip(d[1], by, by + bw)
            dist = np.sqrt((d[0] - cx) ** 2 + (d[1] - cy) ** 2)
            k = int(np.argmin(dist))
            if dist[k] > R:
                out += [1.0, 0.0, 0.0, 1.0]
                continue
            vec = np.array([cx[k] - d[0], cy[k] - d[1]])
            direction = vec / (np.linalg.norm(vec) + 1e-6)
            clear = float(np.clip((d[2] - (bh[k] + margin)) / 20.0, -1.5, 1.5))
            out += [float(dist[k] / R), float(direction[0]), float(direction[1]), clear]
        return out

    # ── Step (shared by all phases) ────────────────────────────────────────

    def step(self, action):
        self.step_count += 1
        self._physics_step(action)
        if self.num_birds > 0:
            self._update_birds()
        self._tick_wind()

        comp = dict.fromkeys(REWARD_COMPONENTS, 0.0)

        comp["coverage"] = self._coverage_reward()
        coverage         = self.get_coverage_percent()      # AFTER the update
        comp["milestone"]  = self._milestone_reward(coverage)
        comp["checkpoint"] = self._update_checkpoint_hits()
        comp["rescue"]     = self._rescue_reward()
        comp["progress"]   = self._progress_reward()        # before _get_obs
        comp["fuel"], all_empty = self._step_fuel()
        comp["rtb"]        = self._check_rtb_landings(coverage)
        comp["height"]     = self._height_penalty()
        comp["wall"]       = self._wall_hug_penalty()
        comp["roof"]       = self._roof_proximity_penalty()
        comp["oob"]        = self._oob_penalty()

        col = self._collision_step()
        comp["drone_hit"]    = col["pen_drone"]
        comp["bird_hit"]     = col["pen_bird"]
        comp["building_hit"] = col["pen_building"]

        n_dead_new = self._apply_lives(col["onset_i"])
        death_pen  = -float(getattr(config, "DEATH_PENALTY", 400.0)) * n_dead_new

        comp["ceiling"], above_ceiling = self._ceiling_penalty()
        (comp["sustained_height"], comp["lazy"],
         h_event, l_event) = self._sustained_checks()

        comp["mission"], mission_done = self._mission_check(coverage)

        total   = float(sum(comp.values()))          # (death not in here yet)
        clipped = self._clip_total(total)
        reward  = clipped + death_pen                # deaths are never clipped
        comp["clip"]  = clipped - total
        comp["death"] = death_pen

        terminated, truncated, fuel_term = self._episode_end_flags(
            all_empty, episode_ends=(self._episode_ends and n_dead_new > 0))
        obs = self._get_obs()

        act = ~self._off
        if act.any():
            z = self.positions[act, 2]
            alt_mean = float(np.mean(z))
            alt_band = float(np.mean(np.abs(z - config.OPTIMAL_SCAN_HEIGHT)
                                     <= config.SCAN_HEIGHT_TOL))
        else:
            alt_mean, alt_band = float("nan"), float("nan")

        info = {
            "coverage":              coverage,
            "step":                  self.step_count,
            "collision_hit":         bool(col["hit_drone"] or col["hit_bird"]
                                          or col["hit_building"]),
            "hit_drone":             bool(col["hit_drone"]),
            "hit_bird":              bool(col["hit_bird"]),
            "hit_building":          bool(col["hit_building"]),
            "collision_onsets":      int(col["onsets"]),
            "height_killed":         bool(h_event),
            "above_ceiling":         bool(above_ceiling),
            "lazy_killed":           bool(l_event),
            "mission_complete":      mission_done,
            "fuel_empty_terminated": fuel_term,
            "fuel":                  self.fuel.tolist(),
            "landed":                self._landed.tolist(),
            "landed_in_zone":        self._landed_in_zone.tolist(),
            "dead":                  self._dead.tolist(),
            "drones_dead":           int(self._dead.sum()),
            "lives":                 self._lives.tolist() if self._lethal else [],
            "deaths_this_step":      int(n_dead_new),
            "rescued":               int(np.sum(self.rescued)),
            "alt_mean":              alt_mean,
            "alt_in_band":           alt_band,
            "r_comp":                dict(comp),
        }
        return obs, reward, terminated, truncated, info

    # ── Coverage / visualiser compatibility ────────────────────────────────

    def get_coverage_percent(self) -> float:
        total = len(self.covered)
        return 0.0 if total == 0 else float(np.sum(self.covered)) / total * 100.0

    @property
    def covered_mask(self): return self.covered

    @property
    def checkpoint_visited(self): return self.checkpoint_visited_mask

    @property
    def communication_links(self):
        links = []
        for i in range(self.num_drones):
            for j in range(i + 1, self.num_drones):
                if np.linalg.norm(self.positions[i] - self.positions[j]) \
                        < config.DETECTION_RADIUS * 2:
                    links.append((i, j))
        return links


# ══════════════════════════════════════════════════════════════════════════════
#  Phases only differ by what they spawn (step() is shared).
# ══════════════════════════════════════════════════════════════════════════════

class Phase1Env(MultiUAVEnv):
    """No obstacles — height discipline, scanning, coverage, fuel."""
    def __init__(self):
        super().__init__(building_count=0, num_birds=0, num_victims=0)


class Phase2Env(MultiUAVEnv):
    """Buildings only."""
    def __init__(self):
        super().__init__(building_count=config.PHASE2_BUILDINGS,
                         num_birds=0, num_victims=0)


class Phase3Env(MultiUAVEnv):
    """Buildings + birds."""
    def __init__(self):
        super().__init__(building_count=config.PHASE3_BUILDINGS,
                         num_birds=config.PHASE3_BIRDS, num_victims=0)


class Phase4Env(MultiUAVEnv):
    """Buildings + birds + victims.  Lethal collisions (lives) + building obs."""
    def __init__(self):
        super().__init__(building_count=config.NUM_BUILDINGS,
                         num_birds=config.NUM_BIRDS,
                         num_victims=config.NUM_VICTIMS,
                         lethal=bool(getattr(config, "PHASE4_LETHAL", True)),
                         building_obs=bool(getattr(config, "PHASE4_BUILDING_OBS", True)))


PHASE_ENVS = [Phase1Env, Phase2Env, Phase3Env, Phase4Env]