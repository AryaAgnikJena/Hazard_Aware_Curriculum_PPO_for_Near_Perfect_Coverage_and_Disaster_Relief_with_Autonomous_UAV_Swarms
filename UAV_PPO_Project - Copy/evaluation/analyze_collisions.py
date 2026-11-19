"""
evaluation/analyze_collisions.py

Answer "WHY do the collisions plateau?" from saved replays (no model/env needed):

    python -m evaluation.analyze_collisions results/visualization_res/replays/<stamp>
    python -m evaluation.analyze_collisions logs/results/final_eval/replays

For every NEW building contact (a drone going from outside to inside a footprint
below the roof) it records
  * how far into the episode it happened,
  * whether the "safe" target list was already EMPTY at that moment.  The env drops
    cells within BUILDING_RADIUS*2.5 of a building centre from the targets while any
    other cell is left; when none is left it falls back to ALL uncovered cells, i.e.
    cells under/next to buildings, and the obs direction + progress reward point there.
  * how far below the roof the drone was (a clipped roof edge vs a deep fly-through).
The key number is the ENRICHMENT: share of contacts that happen while the safe list
is empty  vs  share of all drone-steps spent in that state.  ~1x = unrelated,
>>1x = collisions are driven by chasing the unsafe leftover cells.
"""
import os
import sys
import glob
import numpy as np

from evaluation.replay import load_replay

SAFE_MARGIN = 7.5          # = config.BUILDING_RADIUS * 2.5


def analyse(path):
    d = load_replay(path)
    pos, dead = d["pos"], d["dead"]                 # (T, n, 3), (T, n)
    B, cp, cf = d["buildings"], d["coverage_points"], d["cov_first"]
    T, n = pos.shape[0], pos.shape[1]
    bx, by, bh, bw = B[:, 0], B[:, 1], B[:, 2], B[:, 3]

    # inside[t, i, k]: drone i inside footprint k and below its roof (env's own test)
    px, py, pz = pos[..., 0:1], pos[..., 1:2], pos[..., 2:3]
    inside_k = ((bx <= px) & (px <= bx + bw) & (by <= py) & (py <= by + bw) & (pz <= bh))
    inside = inside_k.any(axis=2)
    onset = inside[1:] & ~inside[:-1] & ~dead[:-1]                  # (T-1, n)

    # safe-target list empty? (uncovered cells farther than SAFE_MARGIN from every centre)
    ctr = np.column_stack([bx + bw / 2, by + bw / 2])
    far = (np.min(np.linalg.norm(cp[:, None] - ctr[None], axis=2), axis=1) > SAFE_MARGIN
           if len(B) else np.ones(len(cp), bool))
    unc = ~((cf[None, :] >= 0) & (cf[None, :] <= np.arange(T)[:, None]))   # (T, cells)
    safe_empty = ~(unc & far[None, :]).any(axis=1)                         # (T,)
    all_done = ~unc.any(axis=1)

    ts, ds = np.nonzero(onset)
    ts = ts + 1
    gaps = []
    for t, i in zip(ts, ds):
        ks = np.nonzero(inside_k[t, i])[0]
        gaps.append(float(np.max(bh[ks]) - pos[t, i, 2]))   # >0: metres below the roof
    alive_steps = (~dead).sum(axis=1)                       # drones flying at each t
    return dict(T=T, n_on=len(ts), t_on=ts, gap=np.array(gaps), safe_empty=safe_empty,
                alive=alive_steps, all_done=all_done, meta=d["meta"])


def main(paths):
    on_total = on_empty = 0
    steps_total = steps_empty = 0.0
    prog, gaps = [], []
    rows = []
    for p in paths:
        r = analyse(p)
        se, al = r["safe_empty"], r["alive"]
        e = int(se[r["t_on"]].sum()) if r["n_on"] else 0
        on_total += r["n_on"]; on_empty += e
        steps_total += al.sum(); steps_empty += (al * se).sum()
        prog += list(r["t_on"] / max(r["T"] - 1, 1)); gaps += list(r["gap"])
        rows.append((os.path.basename(p), r["meta"].get("scenario_type", "?"), r["n_on"], e,
                     float(se.mean()) * 100))
    print(f"\n{'replay':44s} {'type':14s} {'contacts':>8s} {'safe-empty':>10s}  {'%steps safe-empty':>17s}")
    for name, typ, n_on, e, pct in rows:
        print(f"{name[:44]:44s} {typ:14s} {n_on:8d} {e:10d}  {pct:16.0f}%")
    if on_total == 0:
        print("\nNo building contacts in these replays."); return
    p_on = on_empty / on_total
    p_st = steps_empty / max(steps_total, 1)
    gaps = np.array(gaps); prog = np.array(prog)
    print(f"\nbuilding contacts analysed        : {on_total}")
    print(f"  happen while safe list is EMPTY : {p_on * 100:5.1f}%   (that state covers {p_st * 100:.1f}% of drone-steps)")
    print(f"  enrichment                      : {p_on / max(p_st, 1e-9):5.2f}x   (>>1 → drones chasing leftover building cells)")
    print(f"  median point in episode         : {np.median(prog) * 100:5.0f}%   (late = leftover-cell regime)")
    print(f"  clipped roof edge (<2 m below)  : {np.mean(gaps < 2) * 100:5.1f}%   deep fly-through (>=5 m below): {np.mean(gaps >= 5) * 100:.1f}%")


if __name__ == "__main__":
    args = sys.argv[1:] or ["."]
    files = []
    for a in args:
        files += (sorted(glob.glob(os.path.join(a, "*.npz"))) if os.path.isdir(a) else [a])
    main(files)
