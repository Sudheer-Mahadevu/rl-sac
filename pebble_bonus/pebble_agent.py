import numpy as np
import torch

from sac_continuous import SACContinuous
from reward_model import RewardModel, PreferenceBuffer
from teacher import SimulatedTeacher

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PEBBLE:
    """
    PEBBLE: Preference-Based Reward Learning on top of SAC.
    Reference: Lee et al. (2021), https://arxiv.org/abs/2106.05091

    Steps per env interaction:
      1. Collect trajectory segments into a pool.
      2. Periodically query teacher for preference labels on sampled segment pairs.
      3. Train ensemble reward model on preference data (Bradley-Terry loss).
      4. Relabel entire replay buffer with predicted rewards.
      5. Run standard SAC update on relabeled buffer.
    """

    def __init__(
        self,
        obs_dim,
        act_dim,
        reward_model,
        teacher,
        seg_len=50,
        query_freq=5_000,
        queries_per_round=10,
        budget=500,
        reward_update_epochs=10,
        pref_buffer_capacity=3_000,
        sac_kwargs=None,
    ):
        self.sac        = SACContinuous(obs_dim, act_dim, **(sac_kwargs or {}))
        self.reward_mdl = reward_model
        self.teacher    = teacher

        self.seg_len              = seg_len
        self.query_freq           = query_freq
        self.queries_per_round    = queries_per_round
        self.budget               = budget
        self.reward_update_epochs = reward_update_epochs
        self.queries_used         = 0

        self.pref_buffer = PreferenceBuffer(capacity=pref_buffer_capacity)
        self._seg_pool: list = []
        self._steps_since_query = 0

    def _record_segment(self, obs_list, act_list, gt_r_list):
        self._seg_pool.append({
            "obs":  np.array(obs_list,  dtype=np.float32),
            "act":  np.array(act_list,  dtype=np.float32),
            "gt_r": np.array(gt_r_list, dtype=np.float32),
        })
        if len(self._seg_pool) > 2000:
            self._seg_pool = self._seg_pool[-2000:]

    def _sample_segment(self):
        seg = self._seg_pool[np.random.randint(len(self._seg_pool))]
        L   = len(seg["obs"])
        if L > self.seg_len:
            start = np.random.randint(0, L - self.seg_len)
            return {k: seg[k][start:start + self.seg_len] for k in seg}
        return seg

    def _query_teacher(self):
        if len(self._seg_pool) < 2 or self.queries_used >= self.budget:
            return
        n = min(self.queries_per_round, self.budget - self.queries_used)
        for _ in range(n):
            if len(self._seg_pool) < 2:
                break
            s1    = self._sample_segment()
            s2    = self._sample_segment()
            label = self.teacher.label(s1["gt_r"], s2["gt_r"])
            self.pref_buffer.add(s1["obs"], s1["act"], s2["obs"], s2["act"], label)
            self.queries_used += 1
        print(f"  [PEBBLE] Queried {n} prefs | total={self.queries_used}/{self.budget}")

    def _update_reward_model(self):
        self.reward_mdl.update(
            self.pref_buffer,
            batch_size=64,
            n_epochs=self.reward_update_epochs,
        )

    def _relabel_buffer(self):
        buf = self.sac.buffer
        if buf.size == 0:
            return
        r_hat = self.reward_mdl.predict_batch(buf.obs[:buf.size], buf.actions[:buf.size])
        buf.rewards[:buf.size, 0] = r_hat

    def select_action(self, obs, evaluate=False):
        return self.sac.select_action(obs, evaluate=evaluate)

    def store(self, obs, next_obs, action, reward_gt, terminated, truncated):
        self.sac.store(obs, next_obs, action, reward_gt, terminated, truncated)

    def update(self):
        self.sac.update()

    def end_of_step(self, cur_seg_obs, cur_seg_act, cur_seg_gt, episode_done):
        """
        Call once per env step with the running segment buffers.
        Handles segment flushing, teacher querying, reward model update, relabeling.
        """
        self._steps_since_query += 1

        if len(cur_seg_obs) >= self.seg_len:
            self._record_segment(cur_seg_obs, cur_seg_act, cur_seg_gt)
        elif episode_done and len(cur_seg_obs) >= self.seg_len // 2:
            self._record_segment(cur_seg_obs, cur_seg_act, cur_seg_gt)

        if self._steps_since_query >= self.query_freq and self.queries_used < self.budget:
            self._steps_since_query = 0
            self._query_teacher()
            self._update_reward_model()
            self._relabel_buffer()
