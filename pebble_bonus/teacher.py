import numpy as np


class SimulatedTeacher:
    """
    Simulated teacher that labels preference pairs using ground-truth reward sums.
    label = 1.0 if sigma1 preferred, 0.0 if sigma2 preferred, 0.5 if indifferent.
    """

    def __init__(self, teacher_noise=0.0, indiff_tol=0.0):
        self.noise      = teacher_noise
        self.indiff_tol = indiff_tol

    def label(self, gt_rewards_seg1, gt_rewards_seg2):
        r1   = float(np.sum(gt_rewards_seg1))
        r2   = float(np.sum(gt_rewards_seg2))
        diff = r1 - r2

        if abs(diff) <= self.indiff_tol:
            label = 0.5
        elif diff > 0:
            label = 1.0
        else:
            label = 0.0

        if self.noise > 0 and label != 0.5 and np.random.rand() < self.noise:
            label = 1.0 - label

        return label
