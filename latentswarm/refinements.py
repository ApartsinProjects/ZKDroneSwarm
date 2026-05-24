"""The SwarmCF-* refinement family (the follow-up paper) as @algorithm drop-ins.

Each refinement is a member of the same decentralized, communication-free online estimator that
``algorithms.SwarmCF`` implements; it changes one or two axes (signal channel, exploration rule,
confidence handling, contention handling, rank). All are faithful ports of the prototypes in
``experiments/pilot_noise.py`` (the update rules are mined from there, not invented), re-expressed
in the package's Policy contract:

    act(obs)        -> int[m]          action per robot (a task index in its offer, or NO_OP)
    observe(obs)    -> None            update from the post-step broadcast
    predict_rows()  -> [m, n] | None   each robot's predicted reward row (None => structure-free)

Like the native policies, each class manages all ``m`` robots internally (one independent per-robot
estimator each). The prototypes were fed a per-drone ``(t, choices, revealed, cand_sets, rvar)``
tuple; here that tuple is reconstructed per robot ``i`` from its observation dict ``obs[i]`` exactly
as ``baselines.py`` does: a teammate ``k`` is observed iff ``obs[i]["sel"][k] != NO_OP``, the
reading is ``obs[i]["rew"][k]``, and the per-observer variance is ``sigma_own^2`` for ``k == i``
else ``sigma_obs^2`` (the persistent / per-round masking case; under ``line_of_sight`` the env's
distance-dependent noise is approximated by ``sigma_obs^2`` for the precision-weighted methods).

Registered names (display name from the paper in parentheses):
  em_cf            SwarmCF-B     variational Bayesian PMF + predictive-variance UCB exploration
  ard_em_cf        SwarmCF-B-ARD em_cf + automatic relevance determination (learns the rank)
  active_cf        SwarmCF-X     own-count latent-UCB exploration (probe where own coverage is low)
  coord_cf         SwarmCF-Xc    negative-correlated exploration (probe where the SWARM probed little)
  contention_cf    SwarmCF-D     fixed private per-task offset (capacity-1 de-confliction)
  contention_ada_cf SwarmCF-D+   scarcity-gated, loss-self-tuning private offset
  choice_cf        SwarmCF-Ch    learns from teammates' CHOICES only (noise-immune channel)
  both_cf          SwarmCF-RC    fuses reward + competence-weighted choice
  unified_cf       SwarmCF-U     em_cf + loss-gated offset + loss-gated exploration + abundance gate

Loss signal (de-confliction). The prototypes detect a lost contest with ``choices[idx] == -1`` (a
separate contention harness sets losers' choices to -1). The package env keeps a colliding robot's
own ``sel`` set and zeroes its reward, so that exact flag is not in ``obs``. The communication-free,
env-native equivalent used here is a VISIBLE-CONTENTION proxy: robot ``i`` counts itself as having
lost when it engaged a task this round and at least one teammate it can see engaged the SAME task
(the broadcast already carries who-engaged-what). This is exact for detecting contention on the
robot's own pick and is monotone in the true loss rate; the dedicated contention sweep
(``sweeps.sweep_contention``) instead supplies the exact won/lost flag. See FOLLOW_UP notes in the
developer guide.
"""
import numpy as np

from .registry import algorithm
from .algorithms import Policy, NO_OP
from .baselines import _rvar_row


# =================================================================================================
# SwarmCF-B and SwarmCF-B-ARD: variational Bayesian PMF (EMCF)
# =================================================================================================
class _EMCF(Policy):
    """Variational-EM / Bayesian probabilistic matrix factorization with predictive confidence
    intervals (the principled way to use confidence: put uncertainty in the MODEL, not in ad-hoc
    observation weights). Mean-field VI for PMF:

        r_ij ~ N(p_i . u_j, sigma_ij^2),  p_i ~ N(0, Lam^-1),  u_j ~ N(0, Lam^-1),

    with a per-robot posterior q(p_i)=N(muP_i, SP_i) and q(u_j)=N(muU_j, SU_j). The M-step
    propagates uncertainty via the second moment E[u u^T]=S_j + mu_j mu_j^T (a poorly-covered factor
    contributes less automatically). The prediction has a real interval:
        Var(p_i . u_j) = mu_i^T S_j mu_i + mu_j^T S_i mu_j + tr(S_i S_j).
    ``em_beta>0`` uses the predictive sd as a UCB exploration bonus (confidence-directed probing);
    ``em_collective`` restricts the bonus to the COLLECTIVE term p_i^T Sigma_{u_j} p_i (shared-factor
    uncertainty, task-specific and self-annealing) instead of the full predictive variance (whose
    own-factor term over-explores early). The TRUE likelihood precision 1/sigma^2 is used because the
    prior lambda (inside the model) handles regularization, avoiding the scale confound that made
    naive precision-ALS fail. Faithful port of experiments/pilot_noise.EMCF (ard=False here)."""
    name = "em_cf"
    _ard = False

    def __init__(self, cfg, m, n, d_guess, seed=0):
        super().__init__(cfg, m, n, d_guess, seed)
        d = self.d
        self.lam = cfg.em_lam
        self.em_sweeps = cfg.em_sweeps
        self.refit_every = cfg.em_refit_every
        self.em_beta = cfg.em_beta
        self.coll = cfg.em_collective
        self.shrink = cfg.em_shrink
        self.eps = cfg.epsilon
        self.I = np.eye(d)
        s0 = cfg.factor_init_scale
        # per-robot posterior factors + per-column ARD prior precision (shared across this robot's columns)
        self.muP = [self.rng.normal(0, s0, (m, d)) for _ in range(m)]
        self.muU = [self.rng.normal(0, s0, (n, d)) for _ in range(m)]
        self.SP = [np.tile(self.I / self.lam, (m, 1, 1)) for _ in range(m)]
        self.SU = [np.tile(self.I / self.lam, (n, 1, 1)) for _ in range(m)]
        self.alpha = [np.full(d, float(self.lam)) for _ in range(m)]
        self.buf = [([], [], [], []) for _ in range(m)]   # per robot: (engager, task, reward, precision)

    # ---- update --------------------------------------------------------------------------------
    def observe(self, obs):
        for i in range(self.m):
            sel, rew = obs[i]["sel"], obs[i]["rew"]
            rvar = _rvar_row(self.cfg, i, self.m)
            K, J, V, B = self.buf[i]
            for k in range(self.m):
                j = sel[k]
                if j != NO_OP:
                    K.append(k); J.append(int(j)); V.append(float(rew[k]))
                    B.append(1.0 / max(rvar[k], 1e-6))
        self.t += 1
        if self.t % self.refit_every == 0:
            for i in range(self.m):
                self._refit(i)

    def _refit(self, i):
        K0, J0, V0, B0 = self.buf[i]
        if not K0:
            return
        K = np.asarray(K0); J = np.asarray(J0); V = np.asarray(V0, float); B = np.asarray(B0, float)
        d = self.d
        muP, muU, SP, SU, alpha = self.muP[i], self.muU[i], self.SP[i], self.SU[i], self.alpha[i]
        for _ in range(self.em_sweeps):
            Lam0 = np.diag(alpha)                                       # ARD per-column prior
            EPP = SP + np.einsum('id,ie->ide', muP, muP)               # E[p p^T]
            LamU = np.tile(Lam0, (self.n, 1, 1)); bU = np.zeros((self.n, d))
            np.add.at(LamU, J, B[:, None, None] * EPP[K])
            np.add.at(bU, J, (B * V)[:, None] * muP[K])
            SU = np.linalg.inv(LamU)
            muU = np.einsum('jde,je->jd', SU, bU)
            EUU = SU + np.einsum('jd,je->jde', muU, muU)               # E[u u^T]
            LamP = np.tile(Lam0, (self.m, 1, 1)); bP = np.zeros((self.m, d))
            np.add.at(LamP, K, B[:, None, None] * EUU[J])
            np.add.at(bP, K, (B * V)[:, None] * muU[J])
            SP = np.linalg.inv(LamP)
            muP = np.einsum('ide,ie->id', SP, bP)
            if self._ard:                                              # M-step for alpha (ARD)
                EP2 = (muP ** 2).sum(0) + np.einsum('idd->d', SP)
                EU2 = (muU ** 2).sum(0) + np.einsum('jdd->d', SU)
                alpha = np.clip((self.m + self.n) / (EP2 + EU2 + 2e-3), 1e-3, 1e6)
        self.muP[i], self.muU[i], self.SP[i], self.SU[i], self.alpha[i] = muP, muU, SP, SU, alpha

    # ---- prediction / exploration --------------------------------------------------------------
    def _predvar(self, i, cand):
        mi = self.muP[i][i]; Si = self.SP[i]
        t1 = np.einsum('d,jde,e->j', mi, self.SU[i][cand], mi)             # mi^T S_j mi
        t2 = np.einsum('jd,de,je->j', self.muU[i][cand], Si[i], self.muU[i][cand])  # mu_j^T S_i mu_j
        t3 = np.einsum('de,jed->j', Si[i], self.SU[i][cand])              # tr(S_i S_j)
        return np.maximum(t1 + t2 + t3, 0.0)

    def _collvar(self, i, cand):
        mi = self.muP[i][i]
        return np.maximum(np.einsum('d,jde,e->j', mi, self.SU[i][cand], mi), 0.0)

    def _scores_row(self, i):
        s = self.muU[i] @ self.muP[i][i]
        if self.shrink > 0:                                              # shrink uncertain -> popularity
            pop = self.muU[i] @ self.muP[i].mean(0)
            sd = np.sqrt(self._predvar(i, np.arange(self.n)))
            a = sd / (sd + self.shrink)
            s = (1.0 - a) * s + a * pop
        return s

    def act(self, obs):
        a = np.full(self.m, NO_OP, dtype=int)
        for i in range(self.m):
            off = self._offered(obs[i])
            if not off.size:
                continue
            s = self._scores_row(i)[off]
            if self.em_beta > 0:                                        # confidence-interval UCB
                var = (self._collvar(i, off) if self.coll else self._predvar(i, off))
                s = s + self.em_beta * np.sqrt(var) + 1e-6 * self.rng.standard_normal(off.size)
                a[i] = int(off[int(np.argmax(s))])
            elif self.rng.random() < self.eps:
                a[i] = int(self.rng.choice(off))
            else:
                a[i] = int(off[int(np.argmax(s))])
        self.eps = max(self.cfg.epsilon_min, self.eps * self.cfg.epsilon_decay)
        return a

    def predict_rows(self):
        return np.stack([self._scores_row(i) for i in range(self.m)])

    def per_robot_full(self):
        return [self.muP[i] @ self.muU[i].T for i in range(self.m)]

    # ---- ARD effective rank --------------------------------------------------------------------
    def eff_rank(self, thresh=None):
        """Mean effective number of latent dimensions ARD keeps across robots: columns whose energy
        1/alpha_r exceeds ``thresh`` of the largest. For the non-ARD base every column survives
        (alpha is the flat prior), so this returns d; ARD self-prunes (see ard_em_cf)."""
        thresh = self.cfg.ard_eff_rank_thresh if thresh is None else thresh
        out = []
        for i in range(self.m):
            energy = 1.0 / np.maximum(self.alpha[i], 1e-12)
            out.append(int((energy > thresh * energy.max()).sum()))
        return float(np.mean(out))


@algorithm("em_cf")
class EMCF(_EMCF):
    """SwarmCF-B: variational Bayesian PMF with confidence-directed (predictive-variance UCB)
    exploration. See _EMCF for the mechanism."""
    name = "em_cf"
    _ard = False


@algorithm("ard_em_cf")
class ARDEMCF(_EMCF):
    """SwarmCF-B-ARD: SwarmCF-B with automatic relevance determination (ARD). Each latent column r
    carries a prior precision alpha_r updated by the variational M-step; a column is retained when
    the observed (masked) design excites it above the prior/noise floor and pruned (alpha_r -> inf)
    otherwise, so the recovered effective rank (``eff_rank()``) equals the IDENTIFIABLE rank (<= d)
    and is invariant to the guessed rank d-hat -- removing the rank hyperparameter. Faithful port of
    experiments/pilot_noise.EMCF(ard=True) (= ARD-EMCF in the analytical harness)."""
    name = "ard_em_cf"
    _ard = True


# =================================================================================================
# SwarmCF-X and SwarmCF-Xc: count-bonus exploration (RewardCF estimator, different exploration)
# =================================================================================================
class _ALSReward(Policy):
    """Shared scaffolding: the SwarmCF online weighted-ridge ALS estimator (precision-weighted by the
    per-observer noise, like experiments/pilot_noise.RewardCF) with a per-robot buffer, plus pluggable
    selection. Subclasses override ``_select_row(i, off)``."""
    name = "_als_reward"

    def __init__(self, cfg, m, n, d_guess, seed=0):
        super().__init__(cfg, m, n, d_guess, seed)
        d = self.d
        s0 = cfg.factor_init_scale
        self.eps = cfg.epsilon
        self.P = [self.rng.normal(0, s0, (m, d)) for _ in range(m)]
        self.U = [self.rng.normal(0, s0, (n, d)) for _ in range(m)]
        self.buf = [([], [], [], []) for _ in range(m)]   # (engager, task, reward, weight=1/sigma^2)
        self.window = cfg.buffer_window
        self.I = np.eye(d)

    def _ingest(self, obs):
        """Append every visible engagement to each robot's precision-weighted buffer."""
        for i in range(self.m):
            sel, rew = obs[i]["sel"], obs[i]["rew"]
            rvar = _rvar_row(self.cfg, i, self.m)
            K, J, V, W = self.buf[i]
            for k in range(self.m):
                j = sel[k]
                if j != NO_OP:
                    K.append(k); J.append(int(j)); V.append(float(rew[k]))
                    W.append(1.0 / max(rvar[k], 1e-6))
            if len(K) > self.window:
                self.buf[i] = (K[-self.window:], J[-self.window:], V[-self.window:], W[-self.window:])

    def _als(self, i):
        K0, J0, V0, W0 = self.buf[i]
        if not K0:
            return
        K = np.asarray(K0); J = np.asarray(J0); V = np.asarray(V0, float); W = np.asarray(W0, float)
        d = self.d
        pri = self.cfg.ridge * self.I
        P, U = self.P[i], self.U[i]
        for _ in range(self.cfg.als_sweeps):
            Pk = P[K]
            AU = np.tile(pri, (self.n, 1, 1)); bU = np.zeros((self.n, d))
            np.add.at(AU, J, W[:, None, None] * np.einsum('ni,nj->nij', Pk, Pk))
            np.add.at(bU, J, (W * V)[:, None] * Pk)
            U = np.linalg.solve(AU, bU[..., None])[..., 0]
            Uj = U[J]
            AP = np.tile(pri, (self.m, 1, 1)); bP = np.zeros((self.m, d))
            np.add.at(AP, K, W[:, None, None] * np.einsum('ni,nj->nij', Uj, Uj))
            np.add.at(bP, K, (W * V)[:, None] * Uj)
            P = np.linalg.solve(AP, bP[..., None])[..., 0]
        self.P[i], self.U[i] = P, U

    def observe(self, obs):
        self._ingest(obs)
        self.t += 1
        if self.t % self.cfg.refit_every == 0:
            for i in range(self.m):
                self._als(i)

    def _select_row(self, i, off):
        raise NotImplementedError

    def act(self, obs):
        a = np.full(self.m, NO_OP, dtype=int)
        for i in range(self.m):
            off = self._offered(obs[i])
            if off.size:
                a[i] = self._select_row(i, off)
        self.eps = max(self.cfg.epsilon_min, self.eps * self.cfg.epsilon_decay)
        return a

    def predict_rows(self):
        return np.stack([self.P[i][i] @ self.U[i].T for i in range(self.m)])

    def per_robot_full(self):
        return [self.P[i] @ self.U[i].T for i in range(self.m)]


@algorithm("active_cf")
class ActiveCF(_ALSReward):
    """SwarmCF-X: active (uncertainty-reducing) exploration via an own-count latent-UCB. Replaces
    eps-greedy with a UCB on the predicted reward plus a count bonus,
        score_j = <p_i, u_j> + c_active / sqrt(gcount_j + 1),
    where gcount_j is accumulated from the BROADCAST (own + every visible teammate engagement), so
    one robot's probe of an under-observed task lowers EVERYONE's uncertainty about it (collective
    active learning, no communication); the bonus anneals as counts grow. Faithful port of
    experiments/pilot_noise.ActiveCF."""
    name = "active_cf"

    def __init__(self, cfg, m, n, d_guess, seed=0):
        super().__init__(cfg, m, n, d_guess, seed)
        self.c_active = cfg.c_active
        self.gcount = [np.zeros(n) for _ in range(m)]

    def _ingest(self, obs):
        for i in range(self.m):
            for k in range(self.m):
                j = obs[i]["sel"][k]
                if j != NO_OP:
                    self.gcount[i][int(j)] += 1.0     # collective broadcast counts (own + visible teammates)
        super()._ingest(obs)

    def _select_row(self, i, off):
        pred = self.P[i][i] @ self.U[i][off].T
        bonus = self.c_active / np.sqrt(self.gcount[i][off] + 1.0)
        jitter = 1e-6 * self.rng.standard_normal(off.size)
        return int(off[int(np.argmax(pred + bonus + jitter))])


@algorithm("coord_cf")
class CoordCF(_ALSReward):
    """SwarmCF-Xc: coordinated / negative-correlation exploration (no comms). The exploration bonus
    DOWN-weights tasks the SWARM has already probed (counted from the broadcast), bonus_j = c_explore
    / sqrt(1 + gcount_j), so the swarm divides its exploration: if a teammate already probed (and
    broadcast) task j, j's bonus is low and this robot probes elsewhere -- an explicit division of
    labor that is emergent from passive sensing. Faithful port of experiments/pilot_noise.CoordCF."""
    name = "coord_cf"

    def __init__(self, cfg, m, n, d_guess, seed=0):
        super().__init__(cfg, m, n, d_guess, seed)
        self.c_explore = cfg.c_explore
        self.gcount = [np.zeros(n) for _ in range(m)]

    def _ingest(self, obs):
        for i in range(self.m):
            for k in range(self.m):
                j = obs[i]["sel"][k]
                if j != NO_OP:
                    self.gcount[i][int(j)] += 1.0
        super()._ingest(obs)

    def _select_row(self, i, off):
        base = self.P[i][i] @ self.U[i][off].T
        bonus = self.c_explore / np.sqrt(1.0 + self.gcount[i][off])
        return int(off[int(np.argmax(base + bonus))])


# =================================================================================================
# SwarmCF-D and SwarmCF-D+: communication-free de-confliction under capacity-1 contention
# =================================================================================================
class _ContentionMixin:
    """Detects a lost contest for each robot and maintains a per-robot loss EMA. ``act`` stores each
    robot's chosen task in ``self._last_act`` for the next ``observe`` to read.

    Two loss signals are supported. The dedicated contention sweep (``sweeps.sweep_contention``)
    supplies the EXACT won/lost via ``set_lost(lost_array)`` before ``observe`` (mirroring the
    prototype's ``choices[idx] == -1``). In the generic env (where ``obs`` carries no collision flag),
    ``_update_loss`` falls back to a communication-free VISIBLE-CONTENTION proxy: robot ``i`` counts a
    loss when it engaged a task and at least one teammate it can see engaged the SAME task."""

    def _init_loss(self, loss0):
        self.loss_ema = np.full(self.m, float(loss0))
        self._last_act = np.full(self.m, NO_OP, dtype=int)
        self._lost_override = None

    def set_lost(self, lost):
        """Exact per-robot loss flags (1.0 lost / 0.0 won / NaN no-op) for the next observe; used by
        the dedicated contention sweep where won/lost is known."""
        self._lost_override = np.asarray(lost, float)

    def _update_loss(self, obs):
        lr = self.cfg.deconflict_lr
        if self._lost_override is not None:
            for i in range(self.m):
                li = self._lost_override[i]
                if not np.isnan(li):
                    self.loss_ema[i] = (1.0 - lr) * self.loss_ema[i] + lr * float(li)
            self._lost_override = None
            return
        for i in range(self.m):
            ai = int(self._last_act[i])
            if ai == NO_OP:
                continue
            sel = obs[i]["sel"]
            # lost iff a teammate the robot can see engaged the SAME task this round (contention on my pick)
            lost = 1.0 if any(k != i and int(sel[k]) == ai for k in range(self.m)) else 0.0
            self.loss_ema[i] = (1.0 - lr) * self.loss_ema[i] + lr * lost


@algorithm("contention_cf")
class ContentionCF(_ALSReward, _ContentionMixin):
    """SwarmCF-D: the SwarmCF reward estimate with a contention-aware DECISION (keep the estimate,
    change the policy). Each robot draws a FIXED private per-task offset h_i ~ N(0, eps_break^2) once
    and selects argmax_j (<p_i, u_j> + h_i[j]). Within a group of similar robots the perturbed
    argmaxes are almost surely distinct, so same-type collisions vanish, while any task with a reward
    margin above ~2||h|| is unchanged (value preserved up to O(eps_break)). The offset must be FIXED
    (a re-randomized one re-collides) and PRIVATE (a shared one re-synchronizes). Faithful port of
    experiments/pilot_noise.ContentionCF."""
    name = "contention_cf"

    def __init__(self, cfg, m, n, d_guess, seed=0):
        super().__init__(cfg, m, n, d_guess, seed)
        self.offset = [self.rng.standard_normal(n) * cfg.eps_break for _ in range(m)]

    def _select_row(self, i, off):
        score = self.P[i][i] @ self.U[i][off].T + self.offset[i][off]
        return int(off[int(np.argmax(score))])


@algorithm("contention_ada_cf")
class ContentionAdaCF(_ALSReward, _ContentionMixin):
    """SwarmCF-D+: a self-tuning, scarcity-gated private offset. It keeps the SAME fixed private
    direction h_i as SwarmCF-D but adapts the MAGNITUDE to the robot's own observed loss rate,
        scale = eps_lo + (eps_hi - eps_lo) * loss_ema ** coll_pow,
    so a robot that keeps losing contests spreads harder (trades a little predicted value for an
    uncontested task) and a robot that wins freely shrinks back toward greedy (preserves value). A
    HARD scarcity gate from observables only (swarm size m, offer size |S|) engages the offset only
    when |S| <= scarcity_k * m (otherwise there is nothing to de-conflict and the method reduces to
    plain eps-greedy CF, keeping coverage exploration). Fully communication-free: the only feedback
    is the robot's own win/loss. Faithful port of experiments/pilot_noise.ContentionAdaptiveCF."""
    name = "contention_ada_cf"

    def __init__(self, cfg, m, n, d_guess, seed=0):
        super().__init__(cfg, m, n, d_guess, seed)
        self.dir = [self.rng.standard_normal(n) for _ in range(m)]    # fixed private direction (unit-scale)
        self._init_loss(cfg.deconflict_loss0)

    def _scale(self, i):
        c = self.cfg
        return c.deconflict_eps_lo + (c.deconflict_eps_hi - c.deconflict_eps_lo) * (self.loss_ema[i] ** c.deconflict_coll_pow)

    def _scarcity(self, n_offer):
        return 1.0 if n_offer <= self.cfg.deconflict_scarcity_k * self.m else 0.0

    def _select_row(self, i, off):
        sc = self._scale(i) * self._scarcity(off.size)
        if sc <= 0.0:                                                 # abundant offers: plain eps-greedy CF
            if self.rng.random() < self.eps:
                return int(self.rng.choice(off))
            return int(off[int(np.argmax(self.P[i][i] @ self.U[i][off].T))])
        score = self.P[i][i] @ self.U[i][off].T + sc * self.dir[i][off]   # contention: stable offset spread
        return int(off[int(np.argmax(score))])

    def act(self, obs):
        a = super().act(obs)
        self._last_act = a.copy()
        return a

    def observe(self, obs):
        self._update_loss(obs)
        super().observe(obs)


# =================================================================================================
# SwarmCF-Ch and SwarmCF-RC: the action / choice channel
# =================================================================================================
class _ChoiceBase(_ALSReward):
    """Shared machinery for the choice channel. A teammate's observed choice ``c`` in its offered set
    ``S`` is turned into pseudo-observations for the weighted ALS: a positive pseudo-target (c, pos)
    and ``n_neg`` negative pseudo-targets (o != c, neg), where pos/neg are the robot's own 75th/25th
    reward percentiles, each weighted by gamma_k / s2c. ``gamma_k`` is a per-teammate competence
    weight: a temporal ramp (after warm_frac * T) times an EMA of the teammate's choice consistency
    (cosine of successive chosen task factors). Faithful port of experiments/pilot_noise.ChoiceCF's
    pseudo-observation + competence machinery."""
    name = "_choice_base"

    def __init__(self, cfg, m, n, d_guess, seed=0):
        super().__init__(cfg, m, n, d_guess, seed)
        self.s2c = cfg.choice_s2c; self.n_neg = cfg.choice_n_neg
        self.within = cfg.choice_within; self.comp = cfg.choice_competence
        self.warm_frac = cfg.choice_warm_frac; self.T_total = cfg.T
        # per robot: buffered choices (engager, task, offer, step) + consistency state
        self.cbuf = [([], [], [], []) for _ in range(m)]
        self.last_vec = [dict() for _ in range(m)]
        self.consist = [np.full(m, 0.3) for _ in range(m)]

    def _ingest_choices(self, obs):
        for i in range(self.m):
            sel = obs[i]["sel"]
            ck, cc, coff, cstep = self.cbuf[i]
            off_i = self._offered(obs[i])     # robot i's own offer (proxy for teammates' offers; see note)
            for k in range(self.m):
                c = int(sel[k])
                if c == NO_OP:
                    continue
                ck.append(k); cc.append(c); coff.append(off_i); cstep.append(self.t)
                if k in self.last_vec[i]:
                    v0 = self.last_vec[i][k]; v1 = self.U[i][c]
                    cs = float(v0 @ v1 / (np.linalg.norm(v0) * np.linalg.norm(v1) + 1e-9))
                    self.consist[i][k] = 0.9 * self.consist[i][k] + 0.1 * max(cs, 0.0)
                self.last_vec[i][k] = self.U[i][c].copy()

    def _gamma(self, i, k, tmade):
        if not self.comp:
            return 1.0
        w0 = self.warm_frac * self.T_total
        ramp = np.clip((tmade - w0) / max(self.T_total - w0, 1.0), 0.0, 1.0)
        return ramp * np.clip(self.consist[i][k], 0.0, 1.0)

    def _choice_pairs(self, i):
        """Build (K, J, V, W) pseudo-observation lists from robot i's buffered teammate choices,
        appended onto its reward buffer for the joint weighted ALS."""
        K0, J0, V0, W0 = self.buf[i]
        own_r = [v for k, v in zip(K0, V0) if k == i]
        pos = float(np.percentile(own_r, 75)) if len(own_r) >= 4 else 0.55
        neg = float(np.percentile(own_r, 25)) if len(own_r) >= 4 else 0.15
        K = list(K0); J = list(J0); V = list(V0); W = list(W0)
        ck, cc, coff, cstep = self.cbuf[i]
        for k, c, off, ts in zip(ck, cc, coff, cstep):
            g = self._gamma(i, k, ts)
            if g <= 1e-3:
                continue
            wc = g / self.s2c
            K.append(k); J.append(c); V.append(pos); W.append(wc)
            for _ in range(self.n_neg):
                o = int(self.rng.choice(off)) if (self.within and off.size) else int(self.rng.randint(self.n))
                if o != c:
                    K.append(k); J.append(o); V.append(neg); W.append(wc)
        return K, J, V, W

    def _als_pairs(self, i, K, J, V, W):
        if not K:
            return
        K = np.asarray(K); J = np.asarray(J); V = np.asarray(V, float); W = np.asarray(W, float)
        d = self.d
        pri = self.cfg.ridge * self.I
        P, U = self.P[i], self.U[i]
        for _ in range(self.cfg.als_sweeps):
            Pk = P[K]
            AU = np.tile(pri, (self.n, 1, 1)); bU = np.zeros((self.n, d))
            np.add.at(AU, J, W[:, None, None] * np.einsum('ni,nj->nij', Pk, Pk))
            np.add.at(bU, J, (W * V)[:, None] * Pk)
            U = np.linalg.solve(AU, bU[..., None])[..., 0]
            Uj = U[J]
            AP = np.tile(pri, (self.m, 1, 1)); bP = np.zeros((self.m, d))
            np.add.at(AP, K, W[:, None, None] * np.einsum('ni,nj->nij', Uj, Uj))
            np.add.at(bP, K, (W * V)[:, None] * Uj)
            P = np.linalg.solve(AP, bP[..., None])[..., 0]
        self.P[i], self.U[i] = P, U

    def _select_row(self, i, off):
        if self.rng.random() < self.eps:
            return int(self.rng.choice(off))
        return int(off[int(np.argmax(self.P[i][i] @ self.U[i][off].T))])


@algorithm("choice_cf")
class ChoiceCF(_ChoiceBase):
    """SwarmCF-Ch: learn from the action / CHOICE channel alone -- who engaged what -- which is
    immune to the cardinal reward read-off noise sigma entirely (a teammate's choice carries the same
    information whether its reward is read cleanly or noisily). The cross-agent reward channel is NOT
    used; only the robot's OWN reward (own row) plus teammates' competence-weighted choices feed the
    fit. Overtakes the reward channel once observation noise is large. Faithful port of
    experiments/pilot_noise.ChoiceCF (comp configurable via cfg.choice_competence)."""
    name = "choice_cf"

    def observe(self, obs):
        # OWN reward only into the reward buffer (no teammate rewards), then teammate choices.
        for i in range(self.m):
            j = obs[i]["sel"][i]
            if j != NO_OP:
                rvar = _rvar_row(self.cfg, i, self.m)
                K, J, V, W = self.buf[i]
                K.append(i); J.append(int(j)); V.append(float(obs[i]["rew"][i]))
                W.append(1.0 / max(rvar[i], 1e-6))
        self._ingest_choices(obs)
        self.t += 1
        if self.t % self.cfg.refit_every == 0:
            for i in range(self.m):
                self._als_pairs(i, *self._choice_pairs(i))


@algorithm("both_cf")
class BothCF(_ChoiceBase):
    """SwarmCF-RC: fuse BOTH channels in one weighted ALS -- own + teammates' (precision-weighted)
    rewards AND competence-weighted choices. The 'use every available signal' member of the family.
    Faithful port of experiments/pilot_noise.BothCF."""
    name = "both_cf"

    def observe(self, obs):
        self._ingest(obs)            # all visible rewards (own + teammates), precision-weighted
        self._ingest_choices(obs)    # plus teammate choices
        self.t += 1
        if self.t % self.cfg.refit_every == 0:
            for i in range(self.m):
                self._als_pairs(i, *self._choice_pairs(i))


# =================================================================================================
# SwarmCF-U: the unified communication-free method
# =================================================================================================
@algorithm("unified_cf")
class UnifiedCF(_EMCF, _ContentionMixin):
    """SwarmCF-U: ONE method whose refinements activate only on their triggering condition. It is
    SwarmCF-B (variational PMF + predictive-variance UCB, ARD-capable via cfg.em_lam path) UNITED
    with a loss-self-gating de-confliction offset and a loss-gated exploration anneal:
      - the de-confliction offset scale = eps_hi * loss_ema ** coll_pow uses a fixed private
        direction h_i and is exactly 0 until the robot actually loses contests (loss_ema starts at 0),
        so in non-contention settings the method reduces to plain SwarmCF-B;
      - the SAME loss signal DAMPS exploration: beta_eff = em_beta * max(0, 1 - loss_ema/beta_anneal)
        (explore when targets are plentiful, exploit + de-conflict when contested, since exploration
        wastes scarce capacity under capacity-1 matching);
      - an abundance gate damps the UCB and falls back to eps-greedy when |offer| > abundance_k * m
        (no scarcity, where exploration costs earned reward), recovering no-contention earned reward
        without touching the small-offer regimes;
      - an optional finite-horizon anneal (cfg.unified_horizon > 0) scales beta by sqrt((T-t)/T).
    Every gate is driven by quantities the robot already observes, so it stays communication-free.
    Faithful port of experiments/pilot_noise.UnifiedCF (loss_ema starts at 0, not loss0)."""
    name = "unified_cf"

    def __init__(self, cfg, m, n, d_guess, seed=0):
        super().__init__(cfg, m, n, d_guess, seed)
        self.dir = [self.rng.standard_normal(n) for _ in range(m)]    # fixed private de-confliction direction
        self.beta_anneal = cfg.unified_beta_anneal
        self.abundance_k = cfg.unified_abundance_k
        self.horizon = cfg.unified_horizon
        self._init_loss(0.0)                                          # U: loss_ema starts at 0 (off until losses)

    def act(self, obs):
        a = np.full(self.m, NO_OP, dtype=int)
        for i in range(self.m):
            off = self._offered(obs[i])
            if not off.size:
                continue
            s = self._scores_row(i)[off]
            # loss-gated exploration anneal: beta_eff -> 0 as loss_ema -> beta_anneal
            beta_eff = self.em_beta * max(0.0, 1.0 - self.loss_ema[i] / max(self.beta_anneal, 1e-9))
            if self.horizon:                                          # finite-horizon: value of info -> 0 near T
                beta_eff *= float(np.sqrt(max(self.horizon - self.t, 0.0) / self.horizon))
            abundant = bool(self.abundance_k) and off.size > self.abundance_k * self.m
            if abundant:                                              # plentiful offers: exploit, don't explore
                beta_eff = 0.0
            if beta_eff > 0:
                var = (self._collvar(i, off) if self.coll else self._predvar(i, off))
                s = s + beta_eff * np.sqrt(var)
            scale = self.cfg.deconflict_eps_hi * (self.loss_ema[i] ** self.cfg.deconflict_coll_pow)
            if scale > 1e-9:                                          # loss-gated de-confliction offset
                s = s + scale * self.dir[i][off]
            if abundant and scale <= 1e-9 and self.rng.random() < self.eps:
                a[i] = int(self.rng.choice(off))                     # eps-greedy floor at no-contention
                continue
            s = s + 1e-6 * self.rng.standard_normal(off.size)        # tie-break jitter
            a[i] = int(off[int(np.argmax(s))])
        self.eps = max(self.cfg.epsilon_min, self.eps * self.cfg.epsilon_decay)
        self._last_act = a.copy()
        return a

    def observe(self, obs):
        self._update_loss(obs)
        super().observe(obs)
