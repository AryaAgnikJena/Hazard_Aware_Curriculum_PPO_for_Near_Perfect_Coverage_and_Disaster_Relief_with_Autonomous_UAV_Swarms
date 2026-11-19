# configs/config.py   (v2 — rebalanced rewards)
# ──────────────────────────────────────────────────────────────────
# WHY v2?
#   The old penalties (100-200 PER STEP, ×4 escalation, clipped at -500)
#   made one episode cost ≈ -195 000 against ≈ +10 000 available from
#   coverage.  Once drones touched, every action returned the same -500,
#   so PPO had no gradient to separate them.  v2 uses:
#     • a ONE-TIME onset penalty when a new contact starts, plus
#     • a SMALL per-step penalty that shrinks as overlap shrinks
#   so escaping a collision is always rewarded.
#   Typical clean episode ≈ +7 000, a 100-step collision costs ≈ -1 500.
#
#   Bump REWARD_VERSION whenever the reward changes: the trainer then
#   archives the old run (policy + VecNormalize stats are meaningless
#   under a different reward) and starts fresh.
# ──────────────────────────────────────────────────────────────────
import math as _math

REWARD_VERSION      = 4
# Versions whose trained policy is still a good starting point (same observations
# and dynamics, only reward weights changed).  The trainer keeps such a run and
# just resets the reward-normalisation statistics instead of archiving it.
REWARD_COMPATIBLE_WITH = [3]

# ──────────────────────────────────────────────────────────────────
# Map + drones
# ──────────────────────────────────────────────────────────────────
NUM_DRONES          = 5
MAP_SIZE            = 100
MAX_STEPS           = 1000
GRID_SIZE           = 20          # 20×20 = 400 coverage cells

# ──────────────────────────────────────────────────────────────────
# Physics
# ──────────────────────────────────────────────────────────────────
DT                  = 0.1
MIN_HEIGHT          = 2.0
MAX_HEIGHT          = 40.0

# Wind
#   "relative_drag" : drag acts on (v - wind).  A 6 m/s wind pushes a
#                     hovering drone with 0.1·6² = 3.6 N  (of 20 N thrust).
#   "legacy_force"  : old behaviour, wind speed used as a constant
#                     acceleration (6 m/s² ≈ 9 N, 45 % of max thrust).
WIND_MODEL          = "relative_drag"
WIND_MEAN_SPEED     = 6.0
WIND_GUST_AMP       = 3.0
WIND_DRIFT          = 0.97
# Policy cannot counter wind it cannot see → expose mean wind (2 values).
# NOTE: changes obs size 102 → 104, so old checkpoints will not load.
OBS_INCLUDE_WIND    = True

# Vertical action mapping.
#   False (old): thrust_z = a_z·20 N.  Hovering needs a_z = +0.735, a fresh
#                policy outputs ≈0 → every drone free-falls to the floor, and
#                with exploration noise (σ≈0.6) it NEVER holds altitude
#                (simulated: 70–92 % of the time on the floor).
#   True  (new): thrust_z = m·g + a_z·(20 − m·g) N.  a_z = 0 means "hold
#                altitude", ±1 = ±3.5 m/s² of climb/descent.  Symmetric, so
#                noise does not bias the drone downward.
ACTION_Z_HOVER_CENTERED = True

# Stronger drone-drone repulsion (acceleration, m/s², at zero distance)
DRONE_REPULSION_GAIN = 8.0
# The vertical part of that repulsion is scaled down: stacked drones (same XY,
# 0.3 m apart) used to be hurled up/down at 8 m/s², far beyond the ±3.5 m/s²
# of vertical control the policy has → floor / ceiling slams.
DRONE_REPULSION_Z_SCALE = 0.15

# ──────────────────────────────────────────────────────────────────
# Height-dependent scanning
#   r(h) = h × tan(15°),  optimal scan height 20 m (r = 5.36 m)
# ──────────────────────────────────────────────────────────────────
SCAN_HALF_ANGLE_DEG = 15.0
SCAN_TAN            = 0.267949
OPTIMAL_SCAN_HEIGHT = 20.0
SCAN_HEIGHT_TOL     = 3.0
HEIGHT_KILL_MARGIN  = 10.0
DETECTION_RADIUS    = 8

# ──────────────────────────────────────────────────────────────────
# Coverage
# ──────────────────────────────────────────────────────────────────
COVERAGE_REWARD     = 35.0
# Cells within 15 m of a building centre pay NEAR_BUILDING_COVERAGE_BONUS× (72 % of
# the map!).  Useful while learning to navigate (Phases 2-3); in the lethal phase it
# makes a cell next to a building worth 87 vs 35, so risking a contact looks cheap.
NEAR_BUILDING_COVERAGE_BONUS = 2.5
PHASE4_NEAR_BUILDING_BONUS   = 1.0

# ──────────────────────────────────────────────────────────────────
# Buildings — three size classes
# ──────────────────────────────────────────────────────────────────
BUILDING_SMALL_COUNT  = 12
BUILDING_MEDIUM_COUNT =  9
BUILDING_LARGE_COUNT  =  4

BUILDING_SMALL_H_MIN  =  4.0
BUILDING_SMALL_H_MAX  =  8.0
BUILDING_SMALL_W_MIN  =  4.0
BUILDING_SMALL_W_MAX  =  8.0

BUILDING_MEDIUM_H_MIN =  9.0
BUILDING_MEDIUM_H_MAX = 18.0
BUILDING_MEDIUM_W_MIN =  6.0
BUILDING_MEDIUM_W_MAX = 14.0

BUILDING_LARGE_H_MIN  = 19.0
BUILDING_LARGE_H_MAX  = 30.0
BUILDING_LARGE_W_MIN  =  8.0
BUILDING_LARGE_W_MAX  = 16.0

NUM_BUILDINGS         = (BUILDING_SMALL_COUNT
                         + BUILDING_MEDIUM_COUNT
                         + BUILDING_LARGE_COUNT)   # = 25

BUILDING_RADIUS       = 3         # proximity barrier reference (m)

PHASE2_BUILDINGS      = 10
PHASE3_BUILDINGS      = 25

BUILDING_MIN_SIZE     = BUILDING_SMALL_W_MIN      # legacy aliases
BUILDING_MAX_SIZE     = BUILDING_LARGE_W_MAX

# ──────────────────────────────────────────────────────────────────
# Birds
# ──────────────────────────────────────────────────────────────────
NUM_BIRDS           = 10
BIRD_SPEED          = 3.0         # m/s
BIRD_TIME_SCALE     = 2.0         # was a hard-coded ×10 (=30 m/s!).  2 → 6 m/s
BIRD_RADIUS         = 2
MIN_BIRD_Z          = 2.0
PHASE3_BIRDS        = 10

# ──────────────────────────────────────────────────────────────────
# Collision penalties (v2: onset + gentle per-step)
#   per-step terms are multiplied by (0.5 + overlap_fraction) for
#   drone-drone, so separating always reduces the penalty.
# ──────────────────────────────────────────────────────────────────
DRONE_COLLISION_RADIUS      = 3.0
DRONE_HIT_ONSET_PENALTY     = 40.0    # once per new drone-drone contact
DRONE_HIT_STEP_PENALTY      = 8.0     # per pair per step in contact
BUILDING_HIT_ONSET_PENALTY  = 150.0   # v4: was 60  (one contact must cost clearly more than a cell is worth)
BUILDING_HIT_STEP_PENALTY   = 25.0    # v4: was 15  per drone per step inside a building
BIRD_HIT_ONSET_PENALTY      = 30.0
BIRD_HIT_STEP_PENALTY       = 10.0

# Soft proximity barriers (smooth gradient BEFORE contact)
DRONE_SOFT_BARRIER_RANGE     = DRONE_COLLISION_RADIUS * 2.0
DRONE_SOFT_BARRIER_K         = 8.0
DRONE_SOFT_BARRIER_STEEPNESS = 3.0
BIRD_SOFT_RANGE              = 4.0
BIRD_SOFT_K                  = 6.0
BUILDING_SOFT_RANGE          = 6.0    # v4: was 4  (earlier warning)
BUILDING_SOFT_K              = 15.0   # v4: was 6  (stronger gradient before contact)
SOFT_BARRIER_STEEPNESS       = 3.0

# Legacy names (other scripts may still import them)
DRONE_COLLISION_PENALTY = DRONE_HIT_STEP_PENALTY
BUILDING_PENALTY        = BUILDING_HIT_ONSET_PENALTY
BIRD_PENALTY            = BIRD_HIT_ONSET_PENALTY
MIN_STEPS_BEFORE_DEATH  = 100          # unused (collisions are non-fatal)

# Evaluation: an episode is "heavy" if ≥ this fraction of its steps had a hit
COLLISION_HEAVY_FRAC    = 0.05

# ──────────────────────────────────────────────────────────────────
# Fuel / return-to-base
# ──────────────────────────────────────────────────────────────────
OPTIMAL_R           = OPTIMAL_SCAN_HEIGHT * _math.tan(_math.radians(15.0))
_SCAN_STRIP         = 2.0 * OPTIMAL_R * 0.3
FUEL_COVERAGE_FACTOR = 1.5
DRONE_FUEL_STEPS    = max(200, int(
    FUEL_COVERAGE_FACTOR * MAP_SIZE**2 * 0.95
    / (NUM_DRONES * _SCAN_STRIP)
))                                 # ≈ 886
FUEL_LOW_FRAC        = 0.15
FUEL_EMPTY_PENALTY   = 10.0        # per empty drone per step (was 30)
FUEL_EMPTY_TERMINATE = True
FUEL_RETURN_REWARD   = 50.0
RETURN_REWARD        = 80.0
LANDING_RADIUS       = 12.0        # m from spawn-zone centre
LANDING_HEIGHT       = 8.0

# RTB_REQUIRES_POSITION = False (legacy): a drone is flagged "landed" the
#   instant fuel ≤ FUEL_LOW_FRAC, wherever it is → every episode is exactly
#   ~754 steps and "RTB 5/5" means nothing.  Evaluation now also reports
#   how many drones landed INSIDE the spawn zone.
# RTB_REQUIRES_POSITION = True : a drone only lands when it is inside
#   LANDING_RADIUS of the spawn centre.  Raise FUEL_LOW_FRAC to ~0.30 when
#   you switch this on so there is fuel left to fly home.
RTB_REQUIRES_POSITION   = False
RTB_COVERAGE_THRESHOLD  = 95.0

MISSION_COMPLETE_BASE = 5000.0

# ──────────────────────────────────────────────────────────────────
# Height profile  y(h)   (your 3-part curve)
#   0 ≤ h ≤ 20 :  y = 1 − ((h−20)/20)²   rises gently 0 → 1   (−0.0025(h−20)²+1)
#   20 < h     :  y = 1 − ((h−20)/5)²    falls sharply, 0 at 25 m, <0 above
#                                        (−0.04(h−20)²+1)
#   floored at HEIGHT_SHAPE_MIN so 40 m is not −15.
#
#   penalty per drone per step = −HEIGHT_SHAPE_SCALE · (1 − y(h))
#       → 0 at 20 m, mild below (7 m: −2.5, floor: −4.9), −6 at 25 m,
#         capped at −4·SCALE (=−24) far above.   (old: −8.5 at 7 m, ×5 drones)
#   Coverage credit uses max(y, 0) instead of the narrow Gaussian, so a drone
#   at 15 m still earns 94 % per newly covered cell (its scan circle is
#   already physically smaller there).
# ──────────────────────────────────────────────────────────────────
HEIGHT_PROFILE_PEAK       = OPTIMAL_SCAN_HEIGHT     # 20 m
HEIGHT_PROFILE_ZERO_HIGH  = 25.0                    # y = 0 here
HEIGHT_SHAPE_MIN          = -3.0
HEIGHT_SHAPE_SCALE        = 6.0
HEIGHT_PROFILE_FOR_COVERAGE = True
HEIGHT_PENALTY_SCALE      = HEIGHT_SHAPE_SCALE      # legacy alias

HEIGHT_KILL_TICKS         = 100
HEIGHT_SUSTAINED_PENALTY  = 50.0    # was 200
P2_CEILING_PENALTY        = 15.0    # per drone per step above ceiling (was 150)
P2_ROOF_MARGIN            = 2.0
P2_ROOF_PENALTY           = 1.5

# Legacy (Phase 1 ceiling was unreachable: z is clipped to MAX_HEIGHT)
P1_CEILING          = BUILDING_LARGE_H_MAX + HEIGHT_KILL_MARGIN
P1_CEILING_PENALTY  = 100.0

# ──────────────────────────────────────────────────────────────────
# Lazy / stationary penalty (non-fatal)
# ──────────────────────────────────────────────────────────────────
LAZY_GRACE_STEPS         = 150
LAZY_KILL_TICKS          = 300
LAZY_KILL_TICKS_FLOOR    = 100
LAZY_MIN_DISP            = 0.05
LAZY_KILL_PENALTY        = 30.0     # was 100


def lazy_ticks_for(num_milestones_hit: int, num_milestones_total: int) -> int:
    """Lazy-tick curriculum: loose at first, tighter as milestones are banked."""
    total = max(num_milestones_total, 1)
    decay = (LAZY_KILL_TICKS - LAZY_KILL_TICKS_FLOOR) / total
    val = LAZY_KILL_TICKS - decay * num_milestones_hit
    return max(int(round(val)), LAZY_KILL_TICKS_FLOOR)

# ──────────────────────────────────────────────────────────────────
# Walls
# ──────────────────────────────────────────────────────────────────
OOB_PENALTY         = 3.0           # per drone per step touching a wall (was 20 flat)
OOB_MISTAKE_PENALTY = OOB_PENALTY   # legacy alias
WALL_HUG_MARGIN     = 4.0           # edge cells are scannable from ≥5 m, so no penalty there
WALL_HUG_PENALTY    = 3.0           # max per drone per step at the wall

# ──────────────────────────────────────────────────────────────────
# Disaster relief / comms
# ──────────────────────────────────────────────────────────────────
NUM_VICTIMS         = 10
RESCUE_RADIUS       = 6
NORMAL_REWARD       = 40
CRITICAL_REWARD     = 100
COMMUNICATION_RANGE = 45

# ──────────────────────────────────────────────────────────────────
# Training (defaults for train.py; threshold trainer has its own block)
# ──────────────────────────────────────────────────────────────────
TOTAL_TIMESTEPS     = 500_000
N_STEPS_TRAIN       = 1024          # was 256: rollouts were 1/3 of an episode
CHECKPOINT_FREQ     = 100_000
PPO_GAMMA           = 0.99

# ──────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────
LOG_DIR             = "logs/tensorboard/"
CSV_LOG_DIR         = "logs/csv/"
CHECKPOINT_DIR      = "logs/checkpoints/"
MODEL_DIR           = "models/"
NETWORKS_DIR        = "logs/networks/"

# ──────────────────────────────────────────────────────────────────
# Reward shaping
# ──────────────────────────────────────────────────────────────────
CHECKPOINT_REWARD   = 10.0
NUM_CHECKPOINTS     = 25
CRUISE_ALT          = 20.0
CRUISE_TOL          = 3.0

# The WHOLE per-step reward is clipped to ±this.  With v2 penalties this
# almost never binds (it used to saturate at -500 whenever drones touched).
TOTAL_REWARD_CLIP     = 300.0
USE_VORONOI_TARGETS   = True
PROGRESS_REWARD_SCALE = 2.0         # was 3.0


# ──────────────────────────────────────────────────────────────────
# Phase 4 — lethal collisions
#
#   PHASE4_COLLISION_MODE
#     "drone_dies"   : a drone that runs out of LIVES is removed from the
#                      episode (frozen, no more reward); the rest keep flying.
#     "episode_ends" : the first collision of any drone ends the episode.
#
#   COLLISION_LIVES  = contacts a drone survives before it dies.
#     1 = die on the first contact (the FINAL setting, used by evaluate/visualize).
#     The trainer walks LIVES_SCHEDULE downwards as the policy proves it can
#     cope (6 → 4 → 3 → 2 → 1), because switching a policy that makes ~12
#     contacts per episode straight to "one contact = death" just collapses it.
#
#   DEATH_PENALTY is paid per dead drone and is NOT clipped by
#   TOTAL_REWARD_CLIP, so two simultaneous deaths cost twice as much.
# ──────────────────────────────────────────────────────────────────
PHASE4_LETHAL          = True
PHASE4_COLLISION_MODE  = "drone_dies"
COLLISION_LIVES        = 1
LIVES_SCHEDULE         = [6, 4, 3, 2, 1]
DEATH_PENALTY          = 400.0

# Phase 4 also gets building observations (the policy could not see building
# HEIGHTS before, so it could not tell "fly over" from "fly around").
# 4 extra values per drone appended at the END of the observation:
#   nearest-building distance/R, unit direction xy, roof clearance
#   (z − (roof + margin)) / 20   → positive = safe to fly over, negative = avoid
PHASE4_BUILDING_OBS    = True
BUILDING_OBS_RADIUS    = 20.0
ROOF_CLEARANCE_MARGIN  = 2.0