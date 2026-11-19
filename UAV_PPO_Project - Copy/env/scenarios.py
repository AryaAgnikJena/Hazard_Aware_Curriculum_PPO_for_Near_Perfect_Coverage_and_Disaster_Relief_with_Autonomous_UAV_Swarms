"""
env/scenarios.py

Reproducible evaluation suite for the multi-UAV env.

    10 scenario TYPES  x  5 SEEDS  =  50 fixed episodes

A scenario is a plain dict:
    {"type": "tight_corners", "seed": 10402, "params": {...generator overrides...}}

* `type`   decides WHAT is generated (building mix, layout, birds, wind ...).
* `seed`   decides WHICH instance of that type (exact positions, wind direction,
           bird tracks).  The env draws everything from self.np_random, so the
           same (type, seed) reproduces the same episode for every method.

Seed ranges (keep them disjoint so nothing leaks between splits):
    TEST suite        10_000 + 100*type_idx + k     (final evaluation, never trained on)
    VALIDATION suite  20_000 + 100*type_idx + k     (per-burst checkpoint selection)
    training          unseeded / seeds < 1_000
"""
import random

TEST_SEED_BASE = 10_000
VAL_SEED_BASE = 20_000
SEEDS_PER_TYPE = 5

# bldg_mix = (small, medium, large) building counts.  Anything omitted falls
# back to the Phase-4 defaults in configs/config.py.
SCENARIO_TYPES = {
    # control: the training distribution
    "baseline":      dict(),
    # only 19-30 m towers: you cannot fly over them, must route around
    "large_only":    dict(bldg_mix=(0, 0, 10), layout="random"),
    # many 4-8 m blocks: lots of low obstacles, flying over is possible
    "small_dense":   dict(bldg_mix=(40, 0, 0), layout="random"),
    # 45 mixed buildings: high clutter, little free space
    "crowded":       dict(bldg_mix=(20, 15, 10), layout="random"),
    # buildings grouped into a few clusters -> L-shapes, dead ends, tight corners
    "tight_corners": dict(bldg_mix=(8, 10, 6), layout="clustered"),
    # lattice of medium/large blocks with ~2-12 m streets between them
    "street_canyon": dict(bldg_mix=(0, 12, 8), layout="canyon"),
    # buildings along the borders -> corner traps that also trigger wall penalties
    "perimeter":     dict(bldg_mix=(6, 8, 6), layout="perimeter"),
    # 3x the birds, 1.5x faster
    "bird_swarm":    dict(n_birds=30, bird_speed_mult=1.5),
    # stronger mean wind (default 6 m/s), same obstacles
    "high_wind":     dict(wind_speed=10.0),
    # everything at once
    "stress":        dict(bldg_mix=(20, 15, 10), layout="clustered",
                          n_birds=20, bird_speed_mult=1.3, wind_speed=9.0),
}
TYPE_NAMES = list(SCENARIO_TYPES)
assert len(TYPE_NAMES) == 10


def build_suite(seed_base: int = TEST_SEED_BASE, seeds_per_type: int = SEEDS_PER_TYPE):
    """All 50 scenarios, ordered type by type."""
    suite = []
    for ti, name in enumerate(TYPE_NAMES):
        for k in range(seeds_per_type):
            suite.append(dict(type=name,
                              seed=seed_base + 100 * ti + k,
                              params=dict(SCENARIO_TYPES[name])))
    return suite


def sample_test_set(rng_seed=None, seed_base: int = TEST_SEED_BASE,
                    seeds_per_type: int = SEEDS_PER_TYPE):
    """One randomly chosen seed from EACH type -> 10 scenarios.
    Pass rng_seed to make the draw itself reproducible (and print/log it)."""
    rnd = random.Random(rng_seed)
    suite = build_suite(seed_base, seeds_per_type)
    picks = []
    for name in TYPE_NAMES:
        picks.append(rnd.choice([s for s in suite if s["type"] == name]))
    return picks


def describe(scenarios) -> str:
    return ", ".join(f"{s['type']}#{s['seed']}" for s in scenarios)
