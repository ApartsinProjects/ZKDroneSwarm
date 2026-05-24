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


def best_in_subset_oracle_per_step(P, U, rng, offer_size, reps=200):
    """No-contention ceiling: the per-round best-in-OFFERED-subset value, averaged. For each of
    `reps` repetitions, draw a fresh size-`offer_size` menu per robot and take that robot's best
    offered reward; also returns the random (mean-in-offer) floor. This is the oracle/random pair the
    analytical "skill" normalization uses when there is NO capacity-1 contention (targets do not
    deplete, drones may overlap), as in experiments/core._oracle_rand and run_masked's overall eval.
    Returns (oracle_mean, random_mean) as per-robot averages."""
    R = P @ U.T
    m, n = R.shape
    c = n if (not offer_size or offer_size <= 0 or offer_size >= n) else int(offer_size)
    oc, rd = [], []
    for _ in range(reps):
        for k in range(m):
            off = rng.choice(n, size=c, replace=False)
            oc.append(R[k, off].max()); rd.append(R[k, off].mean())
    return float(np.mean(oc)), float(np.mean(rd))


@metric("earned_skill")
class EarnedSkill(Metric):
    """Mean per-round reward normalized to (policy - random)/(oracle - random).

    By default `oracle_mean`/`random_mean` are passed in (the runner uses the capacity-1 Hungarian
    ceiling). For a NO-CONTENTION sweep, pass `oracle_mode="best_in_subset"` together with
    `P, U, rng, offer_size` and the metric computes the best-in-offered-subset oracle/random pair
    instead (the ceiling appropriate when targets do not deplete)."""
    name = "earned_skill"

    def compute(self, mean_reward, random_mean=None, oracle_mean=None,
                oracle_mode="hungarian", P=None, U=None, rng=None, offer_size=0, **kw):
        if oracle_mode == "best_in_subset":
            oracle_mean, random_mean = best_in_subset_oracle_per_step(P, U, rng, offer_size)
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


@metric("unseen_pair_skill_heldout")
class UnseenPairSkillHeldout(Metric):
    """Held-out unseen-pair skill matching the analytical protocol (pilot_c11_masking.run_masked):
    over `reps` repetitions, draw a fresh size-min(c, 20) menu PER ROBOT from that robot's
    NEVER-ENGAGED tasks, take the robot's greedy pick under its model, and normalize the realized
    reward against the best-in-menu (oracle) and mean-in-menu (random). Self-normalized so oracle=1,
    random~0; structure-free policies (pred_rows=None) pick randomly -> the floor.

    kwargs: P, U, pred_rows ([m,n] or None), engaged (list[set]), rng (RandomState),
            offer_size (training c; eval menu = min(c,20), or 20 when c<=0=all-tasks),
            reps (default 120, as run_masked)."""
    name = "unseen_pair_skill_heldout"

    def compute(self, P, U, pred_rows, engaged, rng, offer_size=0, reps=120, **kw):
        R = P @ U.T
        m, n = R.shape
        c = n if (not offer_size or offer_size <= 0) else int(offer_size)
        ev = min(c, 20)                          # eval offer size, decoupled from training c (so c=n works)
        ung, uno, unr = [], [], []
        for _ in range(reps):
            for k in range(m):
                unseen = np.array([j for j in range(n) if j not in engaged[k]])
                if unseen.size < ev:
                    continue
                offu = rng.choice(unseen, size=ev, replace=False)
                if pred_rows is not None:
                    pick = offu[int(np.argmax(pred_rows[k, offu]))]
                else:
                    pick = offu[int(rng.randint(ev))]      # no model -> random pick (floor)
                ung.append(R[k, pick]); uno.append(R[k, offu].max()); unr.append(R[k, offu].mean())
        if not ung:
            return None
        gm, om, rm = float(np.mean(ung)), float(np.mean(uno)), float(np.mean(unr))
        return (gm - rm) / max(om - rm, 1e-6)


@metric("anytime_trajectory")
class AnytimeTrajectory(Metric):
    """Per-round cumulative-normalized skill trajectory and its final value (the anytime AUC summary).
    Given the per-round summed realized / oracle / random reward arrays (length T), returns
    traj[t] = (sum_{<=t} real - sum_{<=t} random) / (sum_{<=t} oracle - random); traj[-1] is the
    whole-episode anytime skill (charges any probe-phase cost in full). Mirrors
    pilot_anytime.run_anytime_clshp.

    kwargs: real, oracle, random (each length-T array of per-round SUMS over robots).
    Returns {"trajectory": [...T...], "final": float}."""
    name = "anytime_trajectory"

    def compute(self, real, oracle, random, **kw):
        real = np.asarray(real, float); oracle = np.asarray(oracle, float); random = np.asarray(random, float)
        cr, co, cd = np.cumsum(real), np.cumsum(oracle), np.cumsum(random)
        traj = (cr - cd) / np.maximum(co - cd, 1e-9)
        return {"trajectory": traj.tolist(), "final": float(traj[-1])}


@metric("cumulative_regret")
class CumulativeRegret(Metric):
    """Cumulative regret = sum_t (1 - skill_t), where the per-round skill_t = (real_t - random_t) /
    (oracle_t - random_t) is the round's reward normalized to [random=0, oracle=1]. Lower is better;
    it is the total normalized mission value left on the table over the run.

    kwargs: real, oracle, random (each length-T array of per-round SUMS over robots).
    Returns float (sum of per-round regret)."""
    name = "cumulative_regret"

    def compute(self, real, oracle, random, **kw):
        real = np.asarray(real, float); oracle = np.asarray(oracle, float); random = np.asarray(random, float)
        skill_t = (real - random) / np.maximum(oracle - random, 1e-9)
        return float(np.sum(1.0 - skill_t))


@metric("time_to_competence")
class TimeToCompetence(Metric):
    """Rounds to reach a fraction `frac` of the oracle dispatch, measured on the CUMULATIVE-normalized
    skill trajectory (the same trajectory the anytime metric reports). Returns the 1-based round index
    at which traj[t] first reaches `frac`, or None if it never does (e.g. a structure-free learner
    stuck at the floor). Mirrors the operational time-to-competence read (opmetrics / F19a).

    kwargs: real, oracle, random (length-T per-round SUMS) OR a precomputed `trajectory`; frac (0.25)."""
    name = "time_to_competence"

    def compute(self, real=None, oracle=None, random=None, trajectory=None, frac=0.25, **kw):
        if trajectory is None:
            real = np.asarray(real, float); oracle = np.asarray(oracle, float); random = np.asarray(random, float)
            cr, co, cd = np.cumsum(real), np.cumsum(oracle), np.cumsum(random)
            trajectory = (cr - cd) / np.maximum(co - cd, 1e-9)
        trajectory = np.asarray(trajectory, float)
        hit = np.where(trajectory >= frac)[0]
        return int(hit[0] + 1) if hit.size else None


@metric("state_uniqueness")
class StateUniqueness(Metric):
    """Do robots learn DIFFERENT models? 1 - mean pairwise cosine of the per-robot predicted reward
    matrices R_hat^(i) = P_i @ U_i^T (mean-centered, frame-invariant). ~0 when every robot fits the
    same data (rho=1, broadcast shared -> cosmetic decentralization); grows as the mask makes states
    genuinely unique. Mirrors pilot_c11_masking.run_masked's `uniq`.

    kwargs: full_mats (list of m predicted [m,n] matrices, e.g. policy.per_robot_full()); or `policy`
    (a policy exposing per_robot_full()). Returns float, or nan for a structure-free policy."""
    name = "state_uniqueness"

    def compute(self, full_mats=None, policy=None, **kw):
        if full_mats is None:
            if policy is None or not hasattr(policy, "per_robot_full"):
                return float("nan")
            full_mats = policy.per_robot_full()
        m = len(full_mats)
        if m < 2:
            return float("nan")
        Rh = np.array([np.asarray(M, float).ravel() for M in full_mats])
        Rh = Rh - Rh.mean(1, keepdims=True)
        nrm = np.linalg.norm(Rh, axis=1, keepdims=True) + 1e-9
        cos = (Rh / nrm) @ (Rh / nrm).T
        return 1.0 - float(cos[np.triu_indices(m, 1)].mean())


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
