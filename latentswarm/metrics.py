"""Pluggable metrics plus the analytic capacity-1 (Hungarian) oracle ceiling.

  earned_skill       : mean per-round reward, normalized to (policy - random)/(oracle - random)
  unseen_pair_skill  : decision quality on NEVER-engaged tasks from the learned model,
                       self-normalized so oracle=1, random~0 (the categorical generalization metric)
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from .registry import metric


def hungarian_oracle_per_step(P, U):
    """Capacity-1 ceiling: optimal one-to-one (Hungarian) matching of the m robots to m distinct
    tasks maximizing total <p_i,u_j>; same every round for static traits. Returns per-robot value."""
    R = P @ U.T
    ri, ci = linear_sum_assignment(-R)
    return float(R[ri, ci].sum() / P.shape[0])


class Metric:
    name = "base"

    def compute(self, **kw):
        raise NotImplementedError


@metric("earned_skill")
class EarnedSkill(Metric):
    name = "earned_skill"

    def compute(self, mean_reward, random_mean, oracle_mean, **kw):
        return (mean_reward - random_mean) / max(oracle_mean - random_mean, 1e-9)


@metric("unseen_pair_skill")
class UnseenPairSkill(Metric):
    name = "unseen_pair_skill"

    def compute(self, P, U, pred_rows, engaged, rng, **kw):
        R = P @ U.T
        m, n = R.shape
        sks = []
        for i in range(m):
            unseen = np.array([j for j in range(n) if j not in engaged[i]])
            if unseen.size < 2:
                continue
            r = R[i, unseen]
            denom = r.max() - r.mean()
            if denom < 1e-9:
                continue
            if pred_rows is not None:
                jrel = int(np.argmax(pred_rows[i, unseen]))
            else:
                jrel = int(rng.randint(unseen.size))   # no model -> random pick (structure-free floor)
            sks.append(float((R[i, unseen[jrel]] - r.mean()) / denom))
        return float(np.mean(sks)) if sks else None


def bootstrap_ci(xs, B=5000, seed=0):
    """(mean, lo, hi) bootstrap 95% CI over the per-seed values."""
    a = np.asarray([x for x in xs if x is not None], float)
    if a.size == 0:
        return (None, None, None)
    if a.size == 1:
        return (float(a[0]), float(a[0]), float(a[0]))
    rng = np.random.RandomState(seed)
    boot = a[rng.randint(0, a.size, size=(B, a.size))].mean(axis=1)
    return float(a.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
