import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KINDS = (("drone", "#f85149"), ("bird", "#f0883e"), ("building", "#bc8cff"))


def evaluate_model(model, env, episodes=10, save_dir="logs/evaluation"):
    """
    Evaluate a trained PPO model on a VecEnv (VecNormalize → DummyVecEnv → env).

    Per episode: coverage, return, steps, collision STEPS, collision EVENTS and
    collision steps per kind (needs the v2 env's info keys).  Writes a CSV plus
    coverage_vs_time / reward_vs_time / coverage_distribution /
    collisions_per_episode PNGs.
    """
    os.makedirs(save_dir, exist_ok=True)
    data = []

    for ep in range(episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        done, steps, total = False, 0, 0.0
        cov_hist, rew_hist = [], []
        coll_steps = coll_events = 0
        kind = {k: 0 for k, _ in KINDS}
        coverage = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            done = bool(dones[0])
            total += float(np.mean(rewards)); steps += 1
            info = infos[0]
            coverage = float(info.get("coverage", coverage))   # survives auto-reset
            cov_hist.append(coverage); rew_hist.append(total)
            if info.get("collision_hit"):
                coll_steps += 1
            coll_events += int(info.get("collision_onsets", 0))
            for k, _ in KINDS:
                kind[k] += int(bool(info.get(f"hit_{k}", False)))

        data.append(dict(coverage=cov_hist, reward=rew_hist, steps=steps,
                         final_coverage=cov_hist[-1] if cov_hist else 0.0,
                         coll_steps=coll_steps, coll_events=coll_events, kind=kind))
        print(f"  Episode {ep:2d}: coverage={data[-1]['final_coverage']:.1f}%  "
              f"steps={steps}  collision_steps={coll_steps}  events={coll_events}")

    csv_path = os.path.join(save_dir, "evaluation_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "steps", "final_coverage", "collision_steps",
                    "collision_events", "coll_drone", "coll_bird", "coll_building"])
        for i, d in enumerate(data):
            w.writerow([i, d["steps"], d["final_coverage"], d["coll_steps"],
                        d["coll_events"], d["kind"]["drone"], d["kind"]["bird"],
                        d["kind"]["building"]])
    print(f"Metrics saved → {csv_path}")

    plt.figure()
    for d in data: plt.plot(d["coverage"], alpha=0.7)
    plt.xlabel("Steps"); plt.ylabel("Coverage %"); plt.title("Coverage vs Time"); plt.grid()
    plt.savefig(os.path.join(save_dir, "coverage_vs_time.png")); plt.close()

    plt.figure()
    for d in data: plt.plot(d["reward"], alpha=0.7)
    plt.xlabel("Steps"); plt.ylabel("Cumulative Reward"); plt.title("Reward vs Time"); plt.grid()
    plt.savefig(os.path.join(save_dir, "reward_vs_time.png")); plt.close()

    plt.figure()
    plt.hist([d["final_coverage"] for d in data], bins=10, edgecolor="black")
    plt.xlabel("Coverage %"); plt.ylabel("Frequency"); plt.title("Final Coverage Distribution")
    plt.savefig(os.path.join(save_dir, "coverage_distribution.png")); plt.close()

    # collisions per episode, stacked by kind
    x = np.arange(len(data))
    bottom = np.zeros(len(data))
    plt.figure(figsize=(9, 4.5))
    for k, col in KINDS:
        vals = np.array([d["kind"][k] for d in data], float)
        plt.bar(x, vals, bottom=bottom, color=col, label=k)
        bottom += vals
    plt.plot(x, [d["coll_steps"] for d in data], "k.-", label="steps with ≥1 hit")
    plt.xlabel("Episode"); plt.ylabel("Collision steps")
    plt.title("Collision steps per episode"); plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_dir, "collisions_per_episode.png"), dpi=130); plt.close()

    print(f"Evaluation graphs saved → {save_dir}")
    return data