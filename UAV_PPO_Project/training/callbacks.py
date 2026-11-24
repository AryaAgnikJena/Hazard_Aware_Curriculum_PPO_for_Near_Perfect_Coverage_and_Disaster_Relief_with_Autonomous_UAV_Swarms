import os
import csv
import math
from stable_baselines3.common.callbacks import BaseCallback


class CheckpointCallback(BaseCallback):
    """
    Save a model checkpoint every save_freq TOTAL steps.
    Temporarily detaches the TensorBoard logger before saving to avoid
    the cloudpickle thread-lock error that occurs when _update_learning_rate
    is monkey-patched (the bound method holds a reference to the model which
    holds the logger which holds an unpicklable threading.Lock).
    """

    def __init__(self, save_freq, save_path, n_envs=1):
        super().__init__()
        self.save_freq  = save_freq
        self.save_path  = save_path
        self.n_envs     = n_envs
        self._adj_freq  = max(save_freq // n_envs, 1) * n_envs

    def _on_step(self):
        if self.num_timesteps % self._adj_freq < self.n_envs:
            path = os.path.join(
                self.save_path,
                f"model_{self.num_timesteps}"
            )
            saved_logger = self.model._logger
            self.model._logger = None
            try:
                self.model.save(path)
                print(f"  Checkpoint saved at step {self.num_timesteps}")
            finally:
                self.model._logger = saved_logger
        return True


class CSVLoggerCallback(BaseCallback):
    """
    Buffered CSV logger — writes every flush_freq steps.

    The output file is NOT created until the first flush (i.e. the first time
    real data exists).  This prevents empty header-only files appearing in
    logs/ when training is set up but never actually starts, or when a
    callback is constructed speculatively.

    episode_reward_mean uses SB3 logger when available; falls back to
    finished-ep returns from VecEnv infos, then forward-fills the last
    known mean.
    """

    def __init__(self, csv_path, flush_freq=1000):
        super().__init__()
        self.csv_path   = csv_path
        self.flush_freq = flush_freq
        self._buffer    = []
        self._last_ep_mean = float("nan")
        self._header_written = False   # write header lazily on first flush

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_header(self, f):
        """Write the CSV header the first time we open the file for real."""
        if not self._header_written:
            writer = csv.writer(f)
            writer.writerow(["timesteps", "mean_reward", "episode_reward_mean"])
            self._header_written = True

    @staticmethod
    def _lookup_rollout_ep_mean(model):
        """Read rollout mean reward from the model's training logger output."""
        log = getattr(model, "logger", None)
        if log is None:
            return float("nan")
        nv = getattr(log, "name_to_value", None)
        if not isinstance(nv, dict):
            return float("nan")
        for key in (
            "rollout/ep_rew_mean",
            "rollout/ep_gt_rew_mean",
        ):
            if key in nv:
                v = nv[key]
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
        return float("nan")

    def _ep_reward_from_infos(self):
        """Latest finished-episode return from any env (Gymnasium VecEnv)."""
        infos = self.locals.get("infos")
        if not infos:
            return float("nan")
        last = float("nan")
        for info in infos:
            if not isinstance(info, dict):
                continue
            ep = info.get("episode")
            if ep is None:
                continue
            if isinstance(ep, dict) and "r" in ep:
                try:
                    last = float(ep["r"])
                except (TypeError, ValueError):
                    pass
        return last

    # ------------------------------------------------------------------
    # SB3 callback hooks
    # ------------------------------------------------------------------

    def _on_step(self):
        import numpy as np

        rewards = self.locals.get("rewards", [0])
        mean_reward = float(np.mean(rewards))

        ep_mean = self._lookup_rollout_ep_mean(self.model)
        if math.isnan(ep_mean):
            ep_from_info = self._ep_reward_from_infos()
            if not math.isnan(ep_from_info):
                self._last_ep_mean = ep_from_info
                ep_mean = ep_from_info
            elif not math.isnan(self._last_ep_mean):
                ep_mean = self._last_ep_mean
        else:
            self._last_ep_mean = ep_mean

        self._buffer.append([
            self.num_timesteps,
            round(mean_reward, 4),
            ep_mean if not math.isnan(ep_mean) else "",
        ])

        if len(self._buffer) >= self.flush_freq:
            self._flush()

        return True

    def _on_rollout_end(self) -> None:
        """SB3 writes rollout/ep_rew_mean right before this — refresh cache."""
        ep_mean = self._lookup_rollout_ep_mean(self.model)
        if not math.isnan(ep_mean):
            self._last_ep_mean = ep_mean

    def _flush(self):
        """Write buffered rows to disk.  File + header created on first call."""
        if not self._buffer:
            return

        # Decide open mode: 'w' the very first time (creates file + header),
        # 'a' on every subsequent flush so we append to existing content.
        mode = "w" if not self._header_written else "a"

        os.makedirs(os.path.dirname(os.path.abspath(self.csv_path)),
                    exist_ok=True)

        with open(self.csv_path, mode=mode, newline="") as f:
            self._ensure_header(f)
            writer = csv.writer(f)
            writer.writerows(self._buffer)

        self._buffer.clear()

    def _on_training_end(self):
        self._flush()

class RewardBreakdownCallback(BaseCallback):
    """
    Logs, once per rollout, to TensorBoard / the SB3 logger:
      reward_components/<name>   mean reward per env-step for each component
      collisions/<kind>_rate     fraction of env-steps with a hit (any/drone/bird/building)
      collisions/onsets_per_step new contacts per env-step
    Requires the v2 env (info["r_comp"], info["hit_*"], info["collision_onsets"]).
    """

    def __init__(self):
        super().__init__()
        self._reset()

    def _reset(self):
        self._sum = {}
        self._n = 0
        self._hits = {"collision_hit": 0, "hit_drone": 0,
                      "hit_bird": 0, "hit_building": 0}
        self._onsets = 0

    def _on_step(self):
        for info in (self.locals.get("infos") or []):
            if not isinstance(info, dict):
                continue
            comp = info.get("r_comp")
            if comp is None:
                continue
            self._n += 1
            for k, v in comp.items():
                self._sum[k] = self._sum.get(k, 0.0) + float(v)
            for k in self._hits:
                self._hits[k] += int(bool(info.get(k, False)))
            self._onsets += int(info.get("collision_onsets", 0))
        return True

    def _on_rollout_end(self) -> None:
        if self._n == 0:
            return
        for k, v in self._sum.items():
            self.logger.record(f"reward_components/{k}", v / self._n)
        for k, v in self._hits.items():
            self.logger.record(f"collisions/{k}_rate", v / self._n)
        self.logger.record("collisions/onsets_per_step", self._onsets / self._n)
        self._reset()