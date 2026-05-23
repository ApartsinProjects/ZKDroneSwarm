"""
DECISION-ONLY, take 2 (APPLES-TO-APPLES): clean choices vs noisy observed rewards
as the CROSS-AGENT channel, with own information held identical.

Every drone always has its OWN clean reward (sigma_own). Methods differ ONLY in
the cross-agent channel they consume:

  Tabular   : own reward only (empirical means; NO transfer).        [reward-class]
  RewardCF  : own reward + others' NOISY rewards (noise-aware MF).    [reward-class, transfer]
  ChoiceCF  : own reward + others' CLEAN choices (NO others' rewards).[choice-class, transfer]
              (naive = unweighted; comp = competence-weighted)

So RewardCF vs ChoiceCF differ ONLY in the cross-agent observation type, any gap
is attributable to choices-vs-noisy-rewards (not to differing own-info). All
methods share the SAME exploration schedule.

Sweep sigma_obs (others' reward-observation noise). Hypothesis: at low noise
RewardCF wins; as sigma_obs rises RewardCF -> Tabular while ChoiceCF stays
robust (choices are noise-free) and OVERTAKES. => decisions beat noisy outcomes.
"""
import numpy as np
import sys
sys.stdout.reconfigure(encoding="utf-8")

from pilot_structure import make_world_struct, oracle_rand


class _EpsBase:
    def __init__(self, m, n, d, idx, seed, eps0=0.5, eps_min=0.05, eps_decay=0.93, **hp):
        self.m = m; self.n = n; self.d = d; self.idx = idx
        self.rng = np.random.RandomState(seed)
        self.eps0 = eps0; self.eps_min = eps_min; self.eps_decay = eps_decay; self.eps = eps0
        self.pulled = np.zeros(n, bool)

    def _decay(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)


class Tabular(_EpsBase):
    def __init__(self, *a, **hp):
        super().__init__(*a, **hp)
        self.counts = np.zeros(self.n); self.sums = np.zeros(self.n)
        self.tot_n = 0; self.tot_s = 0.0

    def _est(self):
        default = (self.tot_s / self.tot_n) if self.tot_n > 0 else 0.5
        e = np.full(self.n, default); m = self.counts > 0
        e[m] = self.sums[m] / self.counts[m]; return e

    def select(self, t, cand):
        a = cand[self.rng.randint(len(cand))] if self.rng.rand() < self.eps \
            else cand[int(np.argmax(self._est()[cand]))]
        self.pulled[a] = True; return a

    def observe(self, t, choices, revealed, cand_sets, rvar):
        r = revealed[self.idx]
        if not np.isnan(r):
            a = int(choices[self.idx]); self.counts[a] += 1; self.sums[a] += r
            self.tot_n += 1; self.tot_s += r
        self._decay()

    def predict_scores(self):
        return self._est()
    def pulled_mask(self):
        return self.pulled


class WeightedMF(_EpsBase):
    def __init__(self, *a, prior_prec=1.0, init_scale=0.1, als_sweeps=5, refit_every=3,
                 precision=True, **hp):
        super().__init__(*a, **hp)
        self.prior_prec = prior_prec; self.als_sweeps = als_sweeps; self.refit_every = refit_every
        self.precision = precision    # weight obs by 1/sigma^2 (True) or uniformly (ablation)
        self.P = self.rng.normal(0, init_scale, (self.m, self.d))
        self.U = self.rng.normal(0, init_scale, (self.n, self.d))
        self.rk = []; self.rj = []; self.rv = []; self.rw = []   # reward obs
        self._I = np.eye(self.d)

    def _als(self, K, J, V, W):
        if len(K) == 0:
            return
        K = np.asarray(K); J = np.asarray(J); V = np.asarray(V); W = np.asarray(W)
        pri = self.prior_prec * self._I
        for _ in range(self.als_sweeps):
            Pk = self.P[K]
            AU = np.tile(pri, (self.n, 1, 1)).astype(float); bU = np.zeros((self.n, self.d))
            np.add.at(AU, J, W[:, None, None] * np.einsum('ni,nj->nij', Pk, Pk))
            np.add.at(bU, J, (W * V)[:, None] * Pk)
            self.U = np.linalg.solve(AU, bU[..., None])[..., 0]
            Uj = self.U[J]
            AP = np.tile(pri, (self.m, 1, 1)).astype(float); bP = np.zeros((self.m, self.d))
            np.add.at(AP, K, W[:, None, None] * np.einsum('ni,nj->nij', Uj, Uj))
            np.add.at(bP, K, (W * V)[:, None] * Uj)
            self.P = np.linalg.solve(AP, bP[..., None])[..., 0]

    def select(self, t, cand):
        a = cand[self.rng.randint(len(cand))] if self.rng.rand() < self.eps \
            else cand[int(np.argmax(self.U[cand] @ self.P[self.idx]))]
        self.pulled[a] = True; return a

    def predict_scores(self):
        return self.U @ self.P[self.idx]
    def pulled_mask(self):
        return self.pulled


class RewardCF(WeightedMF):
    """Cross-agent channel = others' NOISY rewards (noise-aware). No choices."""
    def _refit(self):
        self._als(self.rk, self.rj, self.rv, self.rw)

    def observe(self, t, choices, revealed, cand_sets, rvar):
        for k in range(self.m):
            r = revealed[k]
            if not np.isnan(r):
                self.rk.append(k); self.rj.append(int(choices[k]))
                self.rv.append(float(r))
                self.rw.append((1.0 / max(rvar[k], 1e-6)) if self.precision else 1.0)
        self._decay()
        if (t + 1) % self.refit_every == 0:
            self._refit()


class ChoiceCF(WeightedMF):
    """Cross-agent channel = others' CLEAN choices only. Own reward kept (own row).
    NO others' rewards. comp=True -> competence-weighted; comp=False -> naive."""
    def __init__(self, *a, comp=True, s2c=0.2, n_neg=1, within=True,
                 warm_frac=0.3, T_total=50, **hp):
        super().__init__(*a, **hp)
        self.comp = comp; self.s2c = s2c; self.n_neg = n_neg; self.within = within
        self.warm_frac = warm_frac; self.T_total = T_total
        self.ck = []; self.cc = []; self.coff = []; self.cstep = []
        self.last_vec = {}; self.consist = np.full(self.m, 0.3)

    def observe(self, t, choices, revealed, cand_sets, rvar):
        r = revealed[self.idx]                              # OWN reward only
        if not np.isnan(r):
            self.rk.append(self.idx); self.rj.append(int(choices[self.idx]))
            self.rv.append(float(r)); self.rw.append(1.0 / max(rvar[self.idx], 1e-6))
        for k in range(self.m):
            c = int(choices[k])
            if c < 0:                # masked: teammate k's choice not observed (rho)
                continue
            self.ck.append(k); self.cc.append(c); self.coff.append(cand_sets[k]); self.cstep.append(t)
            if k in self.last_vec:
                v0 = self.last_vec[k]; v1 = self.U[c]
                cs = float(v0 @ v1 / (np.linalg.norm(v0) * np.linalg.norm(v1) + 1e-9))
                self.consist[k] = 0.9 * self.consist[k] + 0.1 * max(cs, 0.0)
            self.last_vec[k] = self.U[c].copy()
        self._decay()
        if (t + 1) % self.refit_every == 0:
            self._refit()

    def _gamma(self, k, tmade):
        if not self.comp:
            return 1.0
        w0 = self.warm_frac * self.T_total
        ramp = np.clip((tmade - w0) / max(self.T_total - w0, 1.0), 0.0, 1.0)
        return ramp * np.clip(self.consist[k], 0.0, 1.0)

    def _refit(self):
        own_r = [v for k, v in zip(self.rk, self.rv) if k == self.idx]
        pos = float(np.percentile(own_r, 75)) if len(own_r) >= 4 else 0.55
        neg = float(np.percentile(own_r, 25)) if len(own_r) >= 4 else 0.15
        K = list(self.rk); J = list(self.rj); V = list(self.rv); W = list(self.rw)
        for k, c, off, ts in zip(self.ck, self.cc, self.coff, self.cstep):
            g = self._gamma(k, ts)
            if g <= 1e-3:
                continue
            wc = g / self.s2c
            K.append(k); J.append(c); V.append(pos); W.append(wc)
            for _ in range(self.n_neg):
                o = int(self.rng.choice(off)) if self.within else self.rng.randint(self.n)
                if o != c:
                    K.append(k); J.append(o); V.append(neg); W.append(wc)
        self._als(K, J, V, W)


class BothCF(ChoiceCF):
    """Fuse BOTH channels in one weighted ALS: own + others' rewards (noise-
    weighted) AND competence-weighted choices. The 'use all available signals'
    variant -- candidate to dominate the design space."""
    def observe(self, t, choices, revealed, cand_sets, rvar):
        for k in range(self.m):
            r = revealed[k]
            if not np.isnan(r):
                self.rk.append(k); self.rj.append(int(choices[k]))
                self.rv.append(float(r)); self.rw.append(1.0 / max(rvar[k], 1e-6))
        for k in range(self.m):
            c = int(choices[k])
            if c < 0:                # masked: teammate k's choice not observed (rho)
                continue
            self.ck.append(k); self.cc.append(c); self.coff.append(cand_sets[k]); self.cstep.append(t)
            if k in self.last_vec:
                v0 = self.last_vec[k]; v1 = self.U[c]
                cs = float(v0 @ v1 / (np.linalg.norm(v0) * np.linalg.norm(v1) + 1e-9))
                self.consist[k] = 0.9 * self.consist[k] + 0.1 * max(cs, 0.0)
            self.last_vec[k] = self.U[c].copy()
        self._decay()
        if (t + 1) % self.refit_every == 0:
            self._refit()


class HybridCF(RewardCF):
    """Probe-then-online-ALS (the synthesis that aims to dominate EVERYWHERE):
    a short UCB probe (informative early coverage, like PTF) -> ONE SVD warm-start
    of the empirical R_hat (global structure) -> then our ONLINE weighted-ALS keeps
    refining (handles missing entries natively; no commit, anytime). vs PTF this
    swaps PTF's weak SGD finetune for weighted-ALS; vs RewardCF it adds UCB-probe +
    SVD warm-start. Hypothesis: matches PTF at dense rho while keeping RewardCF's
    masking-robustness and anytime lead."""
    def __init__(self, *a, T_total=50, probe_frac=0.3, probe_mode="ucb", c=2.0, **hp):
        super().__init__(*a, **hp)
        self.probe_steps = int(probe_frac * T_total); self.probe_mode = probe_mode; self.c = c
        self.pc = np.zeros((self.m, self.n)); self.ps = np.zeros((self.m, self.n))
        self.ptot = np.zeros(self.m); self.step = 0; self.warmed = False

    def select(self, t, cand):
        if self.warmed:
            return super().select(t, cand)            # eps-greedy on learned P,U
        if self.probe_mode == "random":
            a = int(cand[self.rng.randint(len(cand))])
        else:                                          # UCB1 on own empirical row
            i = self.idx; cnt = self.pc[i]; tot = max(self.ptot[i], 1)
            sc = np.full(len(cand), -np.inf)
            for q, j in enumerate(cand):
                if cnt[j] == 0:
                    sc[q] = np.inf
                else:
                    sc[q] = self.ps[i, j] / cnt[j] + self.c * np.sqrt(np.log(tot) / cnt[j])
            if np.isinf(sc.max()):
                a = int(self.rng.choice([cand[q] for q in range(len(cand)) if np.isinf(sc[q])]))
            else:
                a = int(cand[int(np.argmax(sc))])
        self.pulled[a] = True; return a

    def observe(self, t, choices, revealed, cand_sets, rvar):
        for k in range(self.m):                         # accumulate empirical R_hat for SVD
            r = revealed[k]
            if not np.isnan(r):
                j = int(choices[k]); self.pc[k, j] += 1; self.ps[k, j] += r; self.ptot[k] += 1
        super().observe(t, choices, revealed, cand_sets, rvar)   # buffer + online ALS refit
        self.step += 1
        if (not self.warmed) and self.step >= self.probe_steps:
            self._warm(); self.warmed = True

    def _warm(self):
        R_hat = np.where(self.pc > 0, self.ps / np.maximum(self.pc, 1), 0.0)
        try:
            Us, S, Vt = np.linalg.svd(R_hat, full_matrices=False)
            r = min(self.d, len(S)); sq = np.sqrt(np.maximum(S[:r], 0))
            P = Us[:, :r] * sq[None, :]; U = (Vt[:r, :].T * sq[None, :])
            if r < self.d:
                P = np.hstack([P, self.rng.normal(0, 0.01, (self.m, self.d - r))])
                U = np.hstack([U, self.rng.normal(0, 0.01, (self.n, self.d - r))])
            self.P = P; self.U = U                       # warm-start; next ALS refits from here
        except np.linalg.LinAlgError:
            pass


class BothCFGated(BothCF):
    """Confidence-GATED fusion: down-weight the CHOICE channel for a target that
    already has enough REWARD data (per-target reward count). Erases the
    reward-clean penalty (choices contribute ~0 when rewards suffice) while
    keeping the choice benefit where reward data is sparse/noisy -> aims to be
    STRICTLY dominant across the design space."""
    def __init__(self, *a, gate_alpha=0.5, **hp):
        super().__init__(*a, **hp)
        self.gate_alpha = gate_alpha

    def _refit(self):
        own_r = [v for k, v in zip(self.rk, self.rv) if k == self.idx]
        pos = float(np.percentile(own_r, 75)) if len(own_r) >= 4 else 0.55
        neg = float(np.percentile(own_r, 25)) if len(own_r) >= 4 else 0.15
        rcount = (np.bincount(np.asarray(self.rj), minlength=self.n)
                  if self.rj else np.zeros(self.n))
        K = list(self.rk); J = list(self.rj); V = list(self.rv); W = list(self.rw)
        for k, c, off, ts in zip(self.ck, self.cc, self.coff, self.cstep):
            g = self._gamma(k, ts)
            if g <= 1e-3:
                continue
            gate_c = 1.0 / (1.0 + self.gate_alpha * rcount[c])   # reward-sufficiency gate
            K.append(k); J.append(c); V.append(pos); W.append(g * gate_c / self.s2c)
            for _ in range(self.n_neg):
                o = int(self.rng.choice(off)) if self.within else self.rng.randint(self.n)
                if o != c:
                    gate_o = 1.0 / (1.0 + self.gate_alpha * rcount[o])
                    K.append(k); J.append(o); V.append(neg); W.append(g * gate_o / self.s2c)
        self._als(K, J, V, W)


class BothCFPrec(BothCF):
    """Precision-gated fusion (the FIX to BothCFGated): gate the CHOICE channel by
    the per-target reward PRECISION (sum of 1/sigma^2), not the raw count. A target
    with abundant CLEAN reward data suppresses the choice channel (rewards suffice);
    a target with sparse or NOISY reward data keeps it. This is the scale-correct
    confidence gate -> aims to be >= max(RewardCF, ChoiceCF) across the noise grid
    (the count-based gate mis-fired under noise: many noisy obs != high precision)."""
    def __init__(self, *a, gate_alpha=0.05, **hp):
        super().__init__(*a, **hp)
        self.gate_alpha = gate_alpha

    def _refit(self):
        own_r = [v for k, v in zip(self.rk, self.rv) if k == self.idx]
        pos = float(np.percentile(own_r, 75)) if len(own_r) >= 4 else 0.55
        neg = float(np.percentile(own_r, 25)) if len(own_r) >= 4 else 0.15
        rprec = np.zeros(self.n)                       # per-target REWARD precision
        if self.rj:
            np.add.at(rprec, np.asarray(self.rj), np.asarray(self.rw))
        K = list(self.rk); J = list(self.rj); V = list(self.rv); W = list(self.rw)
        for k, c, off, ts in zip(self.ck, self.cc, self.coff, self.cstep):
            g = self._gamma(k, ts)
            if g <= 1e-3:
                continue
            gate_c = 1.0 / (1.0 + self.gate_alpha * rprec[c])    # suppress where reward-precise
            K.append(k); J.append(c); V.append(pos); W.append(g * gate_c / self.s2c)
            for _ in range(self.n_neg):
                o = int(self.rng.choice(off)) if self.within else self.rng.randint(self.n)
                if o != c:
                    gate_o = 1.0 / (1.0 + self.gate_alpha * rprec[o])
                    K.append(k); J.append(o); V.append(neg); W.append(g * gate_o / self.s2c)
        self._als(K, J, V, W)


class StackCF(_EpsBase):
    """Validation-stacked fusion that ADAPTS to the cross-agent reward noise without
    knowing it (ZK). Maintain a RewardCF (own + others' rewards) and a ChoiceCF
    (own reward + others' CHOICES, GLOBAL negatives = strict-ZK public-active-set).
    Hold out a fraction of the drone's OWN clean pulls; select, per drone, whichever
    channel better PREDICTS those held-out own rewards (both channels are blind to
    the held-out pulls). Picks the reward channel when others' rewards are reliable
    and the choice channel when they are noisy -> aims to dominate BOTH channels at
    every noise level. Selection is decentralized self-validation (own clean rewards
    are the drone's private ground truth)."""
    def __init__(self, *a, val_frac=0.3, **hp):
        super().__init__(*a, **hp)
        chp = dict(hp); chp["within"] = False              # strict-ZK global negatives
        self.rew = RewardCF(*a, **hp)
        self.cho = ChoiceCF(*a, **chp)
        self.val_frac = val_frac
        self.vj = []; self.vr = []                          # held-out own (target, reward)
        self.use_rew = True

    def select(self, t, cand):
        if self.rng.rand() < self.eps:
            a = int(cand[self.rng.randint(len(cand))])
        else:
            pred = self.predict_scores()
            a = int(cand[int(np.argmax(pred[cand]))])
        self.pulled[a] = True; self._decay(); return a

    def observe(self, t, choices, revealed, cand_sets, rvar):
        r_own = revealed[self.idx]
        held = (not np.isnan(r_own)) and (self.rng.rand() < self.val_frac)
        if held:                                            # hide this own pull from BOTH channels
            self.vj.append(int(choices[self.idx])); self.vr.append(float(r_own))
            ch2 = choices.copy(); ch2[self.idx] = -1
            rev2 = revealed.copy(); rev2[self.idx] = np.nan
            self.rew.observe(t, ch2, rev2, cand_sets, rvar)
            self.cho.observe(t, ch2, rev2, cand_sets, rvar)
        else:
            self.rew.observe(t, choices, revealed, cand_sets, rvar)
            self.cho.observe(t, choices, revealed, cand_sets, rvar)
        if len(self.vj) >= 3:                               # decentralized self-validation
            vj = np.asarray(self.vj); vr = np.asarray(self.vr)
            er = float(np.mean((self.rew.predict_scores()[vj] - vr) ** 2))
            ec = float(np.mean((self.cho.predict_scores()[vj] - vr) ** 2))
            self.use_rew = (er <= ec)

    def predict_scores(self):
        return self.rew.predict_scores() if self.use_rew else self.cho.predict_scores()

    def pulled_mask(self):
        return self.pulled


class ActiveCF(RewardCF):
    """Active (uncertainty-reducing) exploration in LATENT space. Replace eps-greedy
    with a UCB on the predicted reward plus a count-based uncertainty bonus:
        score_j = <p_i, u_j> + c_active / sqrt(global_count_j + 1).
    The global count per target is accumulated from the BROADCAST, so one drone's
    probe of an under-observed target lowers EVERYONE'S uncertainty about it
    (collective active learning with no communication). Anneals automatically as
    counts grow. The principled 'confidence drives exploration' idea (Motif A),
    here via a cheap count proxy for the factor-posterior uncertainty."""
    def __init__(self, *a, c_active=0.5, **hp):
        super().__init__(*a, **hp)
        self.c_active = c_active
        self.gcount = np.zeros(self.n)

    def select(self, t, cand):
        cand = np.asarray(cand)
        pred = self.U[cand] @ self.P[self.idx]
        bonus = self.c_active / np.sqrt(self.gcount[cand] + 1.0)
        jitter = 1e-6 * self.rng.randn(len(cand))
        a = int(cand[int(np.argmax(pred + bonus + jitter))])
        self.pulled[a] = True; self._decay(); return a

    def observe(self, t, choices, revealed, cand_sets, rvar):
        for k in range(self.m):
            if not np.isnan(revealed[k]):
                self.gcount[int(choices[k])] += 1     # collective broadcast counts
        super().observe(t, choices, revealed, cand_sets, rvar)


class ConfCF(RewardCF):
    """Configurable CONFIDENCE estimator: how to USE confidence WITHOUT corrupting the
    global fit (the ablation showed inverse-variance reweighting of the FIT hurts unseen
    because it under-uses the broadcast). Knobs, separable:

      conf_mode (how OBSERVED events are weighted in the ALS fit):
        'uniform'  w=1                       (best generalization at default noise)
        'full'     w=1/sigma^2               (Gauss-Markov; under-uses broadcast)
        'rel'      w=(1/sigma^2)/mean        (relative precision: keep ratio, fix the
                                              data-vs-prior scale that 'full' distorts)
        'temp'     w=sigma^-alpha            (tempered precision; alpha in [0,2])
        'cap'      w=min(1/sigma^2, cap)     (bounded precision: noise-aware but no row
                                              is allowed to dominate the broadcast)
      shrink>0 : shrink each PREDICTION toward the rank-1 popularity prior by inverse
                 COVERAGE (confident where well-observed, conservative where sparse).
                 This puts confidence in the DECISION, not the fit. alpha_j =
                 shrink/(shrink+count_j); R_hat <- (1-a) R_hat + a (U @ mean_i P_i).
      post>0  : like shrink but uses the ALS POSTERIOR precision of u_j (trace of the
                 inverted normal matrix) instead of a raw count (principled confidence).
      c_active>0 : latent-UCB exploration with the broadcast count bonus.

    All of these LEAVE the broadcast at full weight in the fit (uniform/rel/cap), so
    they keep the unseen-generalization that 'full' precision threw away, while still
    being noise-aware (rel/cap/temp) or coverage-aware (shrink/post)."""
    def __init__(self, *a, conf_mode="uniform", conf_alpha=1.0, conf_cap=5.0,
                 shrink=0.0, post=0.0, c_active=0.0, **hp):
        super().__init__(*a, **hp)
        self.conf_mode = conf_mode; self.conf_alpha = conf_alpha; self.conf_cap = conf_cap
        self.shrink = shrink; self.post = post; self.c_active = c_active
        self.rvar_buf = []; self.cnt = np.zeros(self.n); self.uprec = np.zeros(self.n)

    def observe(self, t, choices, revealed, cand_sets, rvar):
        for k in range(self.m):
            r = revealed[k]
            if not np.isnan(r):
                j = int(choices[k])
                self.rk.append(k); self.rj.append(j); self.rv.append(float(r))
                self.rvar_buf.append(float(rvar[k])); self.rw.append(1.0)
                self.cnt[j] += 1
        self._decay()
        if (t + 1) % self.refit_every == 0:
            self._refit()

    def _weights(self):
        rv = np.maximum(np.asarray(self.rvar_buf), 1e-6)
        m = self.conf_mode
        if m == "full":   w = 1.0 / rv
        elif m == "rel":  w = 1.0 / rv; w = w / max(w.mean(), 1e-9)
        elif m == "temp": w = rv ** (-self.conf_alpha / 2.0)
        elif m == "cap":  w = np.minimum(1.0 / rv, self.conf_cap)
        elif m == "relcap":   # ratio-bounded precision, mean-normalized: discount noisy
            w = 1.0 / rv      # obs but never let any row dominate/vanish; fixed prior scale
            med = np.median(w); R = self.conf_alpha
            w = np.clip(w, med / R, med * R); w = w / max(w.mean(), 1e-9)
        else:             w = np.ones(len(rv))            # uniform
        return w

    def _refit(self):
        W = self._weights()
        self._als(self.rk, self.rj, self.rv, W)
        if self.post > 0 and len(self.rk):                # ALS posterior precision of u_j
            J = np.asarray(self.rj); Wn = np.asarray(W); Pk = self.P[np.asarray(self.rk)]
            quad = Wn * np.einsum('ni,ni->n', Pk, Pk)     # w * ||p_k||^2 contribution
            self.uprec = np.zeros(self.n); np.add.at(self.uprec, J, quad)
            self.uprec += self.prior_prec

    def predict_scores(self):
        s = self.U @ self.P[self.idx]
        if self.shrink > 0 or self.post > 0:
            pop = self.U @ self.P.mean(0)                 # rank-1 popularity prediction
            if self.post > 0:                            # confidence = posterior precision
                a = self.prior_prec / np.maximum(self.uprec, 1e-9)
            else:                                        # confidence = coverage count
                a = self.shrink / (self.shrink + self.cnt)
            s = (1.0 - a) * s + a * pop
        return s

    def select(self, t, cand):
        cand = np.asarray(cand)
        if self.c_active > 0:
            sc = self.predict_scores()[cand] + self.c_active / np.sqrt(self.cnt[cand] + 1.0)
            sc = sc + 1e-6 * self.rng.randn(len(cand))
            a = int(cand[int(np.argmax(sc))])
        elif self.rng.rand() < self.eps:
            a = int(cand[self.rng.randint(len(cand))])
        else:
            a = int(cand[int(np.argmax(self.predict_scores()[cand]))])
        self.pulled[a] = True
        return a


class EMCF(_EpsBase):
    """Variational-EM / BAYESIAN factorization with CONFIDENCE INTERVALS (the principled
    way to use confidence: put uncertainty in the MODEL, not in ad-hoc observation
    weights). Mean-field VI for probabilistic matrix factorization:
        r_ij ~ N(p_i.u_j, sigma_ij^2),  p_i ~ N(0, lam^-1 I),  u_j ~ N(0, lam^-1 I).
    Each factor keeps a full posterior q(p_i)=N(mu_i, S_i), q(u_j)=N(mu_j, S_j). The
    M-step propagates uncertainty via the SECOND MOMENT E[u u^T]=S_j+mu_j mu_j^T (so a
    noisy/poorly-covered factor automatically contributes less, WITHOUT us hand-tuning a
    weight). Prediction has a real interval: Var(p_i.u_j) = mu_i^T S_j mu_i + mu_j^T S_i
    mu_j + tr(S_i S_j). em_beta>0 uses that predictive sd as a UCB exploration bonus
    (confidence-directed probing); shrink>0 shrinks high-variance predictions toward the
    popularity prior. Uses the TRUE likelihood precision 1/sigma^2 -- correct here because
    the prior lam (in the model) handles regularization, avoiding the scale confound that
    made naive precision-ALS fail."""
    def __init__(self, *a, lam=1.0, em_sweeps=6, refit_every=4, em_beta=1.0, shrink=0.0,
                 init_scale=0.1, ard=False, coll=False, **hp):
        super().__init__(*a, **hp)
        self.lam = lam; self.em_sweeps = em_sweeps; self.refit_every = refit_every
        self.em_beta = em_beta; self.shrink = shrink
        self.coll = coll      # exploration bonus uses COLLECTIVE latent uncertainty (shared u_j) only
        self.ard = ard                          # automatic relevance determination on factor columns
        self.alpha = np.full(self.d, float(lam))   # per-column prior precision (ARD self-prunes rank)
        self.I = np.eye(self.d)
        self.muP = self.rng.normal(0, init_scale, (self.m, self.d))
        self.muU = self.rng.normal(0, init_scale, (self.n, self.d))
        self.SP = np.tile(self.I / lam, (self.m, 1, 1))
        self.SU = np.tile(self.I / lam, (self.n, 1, 1))
        self.rk = []; self.rj = []; self.rv = []; self.rb = []     # reward obs + precision

    def observe(self, t, choices, revealed, cand_sets, rvar):
        for k in range(self.m):
            r = revealed[k]
            if not np.isnan(r):
                self.rk.append(k); self.rj.append(int(choices[k]))
                self.rv.append(float(r)); self.rb.append(1.0 / max(rvar[k], 1e-6))
        self._decay()
        if (t + 1) % self.refit_every == 0:
            self._refit()

    def _refit(self):
        if not self.rk:
            return
        K = np.asarray(self.rk); J = np.asarray(self.rj)
        V = np.asarray(self.rv); B = np.asarray(self.rb)
        for _ in range(self.em_sweeps):
            Lam0 = np.diag(self.alpha)                                    # ARD per-column prior
            EPP = self.SP + np.einsum('id,ie->ide', self.muP, self.muP)   # E[p p^T]
            LamU = np.tile(Lam0, (self.n, 1, 1))
            bU = np.zeros((self.n, self.d))
            np.add.at(LamU, J, B[:, None, None] * EPP[K])
            np.add.at(bU, J, (B * V)[:, None] * self.muP[K])
            self.SU = np.linalg.inv(LamU)
            self.muU = np.einsum('jde,je->jd', self.SU, bU)
            EUU = self.SU + np.einsum('jd,je->jde', self.muU, self.muU)   # E[u u^T]
            LamP = np.tile(Lam0, (self.m, 1, 1))
            bP = np.zeros((self.m, self.d))
            np.add.at(LamP, K, B[:, None, None] * EUU[J])
            np.add.at(bP, K, (B * V)[:, None] * self.muU[J])
            self.SP = np.linalg.inv(LamP)
            self.muP = np.einsum('ide,ie->id', self.SP, bP)
            if self.ard:                                                  # M-step for alpha (ARD)
                EP2 = (self.muP ** 2).sum(0) + np.einsum('idd->d', self.SP)
                EU2 = (self.muU ** 2).sum(0) + np.einsum('jdd->d', self.SU)
                self.alpha = np.clip((self.m + self.n) / (EP2 + EU2 + 2e-3), 1e-3, 1e6)

    def eff_rank(self, thresh=0.05):
        """Effective number of latent dimensions ARD keeps: columns whose 'energy'
        1/alpha_r is above a fraction `thresh` of the largest (the rest are pruned)."""
        energy = 1.0 / np.maximum(self.alpha, 1e-12)
        return int((energy > thresh * energy.max()).sum())

    def _predvar(self):
        i = self.idx; mi = self.muP[i]; Si = self.SP[i]
        t1 = np.einsum('d,jde,e->j', mi, self.SU, mi)            # mi^T S_j mi
        t2 = np.einsum('jd,de,je->j', self.muU, Si, self.muU)    # mu_j^T S_i mu_j
        t3 = np.einsum('de,jed->j', Si, self.SU)                 # tr(S_i S_j)
        return np.maximum(t1 + t2 + t3, 0.0)

    def _collvar(self):
        # COLLECTIVE latent uncertainty: variance of the SHARED factor u_j in my own
        # direction, p_i^T Sigma_{u_j} p_i. Target-specific (high for under-observed targets,
        # low for well-observed) and ANNEALS as the swarm pins down U -> directs probes to
        # reduce the swarm's structure uncertainty WITHOUT the uniform over-exploration that
        # the own-factor term (u_j^T Sigma_{p_i} u_j, large early for every target) causes.
        mi = self.muP[self.idx]
        return np.maximum(np.einsum('d,jde,e->j', mi, self.SU, mi), 0.0)

    def predict_scores(self):
        s = self.muU @ self.muP[self.idx]
        if self.shrink > 0:                                      # shrink uncertain -> popularity
            pop = self.muU @ self.muP.mean(0)
            sd = np.sqrt(self._predvar()); a = sd / (sd + self.shrink)
            s = (1.0 - a) * s + a * pop
        return s

    def select(self, t, cand):
        cand = np.asarray(cand)
        s = self.predict_scores()[cand]
        if self.em_beta > 0:                                    # confidence-interval UCB
            var = (self._collvar() if self.coll else self._predvar())[cand]
            s = s + self.em_beta * np.sqrt(var) + 1e-6 * self.rng.randn(len(cand))
            a = int(cand[int(np.argmax(s))])
        elif self.rng.rand() < self.eps:
            a = int(cand[self.rng.randint(len(cand))])
        else:
            a = int(cand[int(np.argmax(s))])
        self.pulled[a] = True
        return a

    def predict_full(self):
        return self.muP @ self.muU.T
    def pulled_mask(self):
        return self.pulled


class ChoiceEM(ChoiceCF):
    """JOINT EM over latent factors AND per-teammate choice-informativeness gamma_k
    (Section 5.6). Each sensed teammate CHOICE is modeled as a mixture: with probability
    gamma_k a Boltzmann-rational (softmax) pick on the teammate's true preference, else a
    uniform-random pick. E-step: per-choice responsibility r (posterior that the choice
    was model-based, not random). M-step: gamma_k <- mean responsibility, and the factor
    fit weights each choice's positive/negative pair by r, so a near-random choice barely
    moves the structure (the fix for low-confidence teammates poisoning the latent model).
    Replaces ChoiceCF's fixed temporal competence ramp with this learned, per-choice r."""
    def __init__(self, *a, tau=0.3, gamma_init=0.5, warm_em=0.0, predictive=False, **hp):
        super().__init__(*a, **hp)
        self.tau = tau; self.warm_em = warm_em   # warm_em: ignore choices before warm_em*T_total
        self.predictive = predictive  # HELD-OUT responsibility: score each choice ONCE, against the
        self.gamma = np.full(self.m, float(gamma_init))   # per-teammate informativeness estimate
        self.csoft = []               # model BEFORE this refit incorporates it, then freeze (so a
        self.coffn = []               # random teammate's choices can't be overfit into looking good)

    def _soft(self, k, c, off):
        """Model probability that teammate k picks its observed target c from offer off
        (softmax on OUR current estimate of k's preference). The gamma-free evidence that
        a choice is model-consistent; returns (soft_c, offer_size)."""
        off = np.asarray(off)
        logits = (self.U[off] @ self.P[k]) / max(self.tau, 1e-6)
        logits = logits - logits.max()
        w = np.exp(logits); Z = max(w.sum(), 1e-12)
        pos = np.where(off == c)[0]
        return (float(w[pos[0]] / Z) if len(pos) else 1.0 / len(off)), int(len(off))

    def _resp_from(self, k, soft_c, n_off):
        gk = float(self.gamma[k])
        return gk * soft_c / (gk * soft_c + (1.0 - gk) / n_off + 1e-12)

    def _refit(self):
        own_r = [v for kk, v in zip(self.rk, self.rv) if kk == self.idx]
        pos = float(np.percentile(own_r, 75)) if len(own_r) >= 4 else 0.55
        neg = float(np.percentile(own_r, 25)) if len(own_r) >= 4 else 0.15
        # warm-up: ignore choices made before warm_em*T_total (build the model on REWARDS
        # first so the E-step has a decent model to judge choices -> breaks the cold-start deadlock)
        t0 = self.warm_em * self.T_total
        # E-step evidence (soft_c per buffered choice). predictive=True: score each choice ONCE,
        # against the model BEFORE this refit's ALS incorporates it (the choices added since the last
        # refit are NOT yet in self.P), then FREEZE it -> a uniform-random teammate gets soft_c~1/|S|
        # held-out (cannot be overfit), so its gamma stays low. predictive=False: in-sample each refit.
        if self.predictive:
            while len(self.csoft) < len(self.ck):
                i = len(self.csoft); ts = self.cstep[i]
                if ts < t0:
                    self.csoft.append(0.0); self.coffn.append(len(self.coff[i]))
                else:
                    s, no = self._soft(self.ck[i], self.cc[i], self.coff[i])
                    self.csoft.append(s); self.coffn.append(no)
            softs = self.csoft; offns = self.coffn
        else:
            softs = []; offns = []
            for k, c, off, ts in zip(self.ck, self.cc, self.coff, self.cstep):
                if ts < t0:
                    softs.append(0.0); offns.append(len(off))
                else:
                    s, no = self._soft(k, c, off); softs.append(s); offns.append(no)
        # E-step responsibility (fresh gamma) + M-step (1): gamma_k = mean responsibility (post-warmup)
        resp = [0.0] * len(self.ck); gsum = np.zeros(self.m); gcnt = np.zeros(self.m)
        for i, (k, ts) in enumerate(zip(self.ck, self.cstep)):
            if ts < t0:
                continue
            r = self._resp_from(k, softs[i], offns[i]); resp[i] = r
            gsum[k] += r; gcnt[k] += 1.0
        self.gamma = np.where(gcnt > 0, gsum / np.maximum(gcnt, 1.0), self.gamma)
        # M-step (2): weighted ALS; choice contributions weighted by responsibility r
        K = list(self.rk); J = list(self.rj); V = list(self.rv); W = list(self.rw)
        for i, (k, c, off) in enumerate(zip(self.ck, self.cc, self.coff)):
            r = resp[i]
            if r <= 1e-3:
                continue
            wc = r / self.s2c
            K.append(k); J.append(c); V.append(pos); W.append(wc)
            for _ in range(self.n_neg):
                o = int(self.rng.choice(off)) if self.within else self.rng.randint(self.n)
                if o != c:
                    K.append(k); J.append(o); V.append(neg); W.append(wc)
        self._als(K, J, V, W)


class ContentionCF(RewardCF):
    """RewardCF ESTIMATE + a CONTENTION-AWARE DECISION (keep the estimate, change the
    policy). Two de-confliction ingredients, both communication-free: (1) a popularity
    penalty, discount a target by how often the public broadcast shows it engaged
    (contested targets get engaged repeatedly), routing drones off crowded targets onto
    their personal-but-uncontested favorites; (2) softmax SAMPLING instead of argmax, so
    drones with similar tastes diverge (symmetry breaking). Aims to WIN under contention,
    where greedy argmax makes everyone collide on the same few high-value targets."""
    def __init__(self, *a, eps_break=0.1, **hp):
        super().__init__(*a, **hp)
        # FIXED private per-target offset (deterministic symmetry breaking). Same-type drones
        # have near-identical R_hat, so this fixed perturbation flips which near-tied target each
        # consistently prefers -> they de-conflict stably (unlike random softmax, which re-collides
        # every round). Clear winners (gap > eps_break) stay greedy, so value is preserved.
        self.eps_break_used = eps_break
        self.offset = self.rng.randn(self.n) * eps_break

    def select(self, t, cand):
        cand = np.asarray(cand)
        score = self.U[cand] @ self.P[self.idx] + self.offset[cand]
        a = int(cand[int(np.argmax(score))])
        self.pulled[a] = True
        return a


class ContentionAdaptiveCF(ContentionCF):
    """Contention-aware policy with a SELF-TUNING symmetry-breaking offset. It keeps the
    SAME fixed private DIRECTION as ContentionCF (stable de-confliction, Theorem T7) but
    ADAPTS the MAGNITUDE to the drone's own observed loss rate:
      scale_t = eps_lo + (eps_hi - eps_lo) * loss_ema
    When the drone keeps LOSING contests its target is contested, so it spreads harder
    (large offset => willing to trade some predicted value for an uncontested target);
    when it WINS freely the offset shrinks back toward greedy (preserve value). This is
    fully communication-free / ZK: the only feedback used is the drone's OWN public
    win/loss outcome (choices[idx] == -1 means it lost the contest this round). Goal: match
    greedy at no-contention and beat the fixed offset at severe contention, i.e. dominate
    the whole contention range with ONE knob the swarm tunes itself."""
    def __init__(self, *a, eps_lo=0.02, eps_hi=0.8, lr=0.15, loss0=0.3, coll_pow=2.0,
                 scarcity_k=2.0, **hp):
        super().__init__(*a, **hp)
        # self.offset (fixed direction * eps_break) is inherited; reuse its DIRECTION only.
        self.dir = self.offset / max(self.eps_break_used, 1e-9)
        self.eps_lo = eps_lo; self.eps_hi = eps_hi; self.lr = lr
        self.coll_pow = coll_pow; self.scarcity_k = scarcity_k
        self.loss_ema = float(loss0)

    def _scale(self):
        # CONVEX (coll_pow>1) loss->scale law: minor, irreducible taste-collisions (small
        # loss_ema) leave the offset near-OFF (preserve value, match greedy when targets are
        # plentiful); only a HIGH sustained loss rate (true scarcity) ramps the offset up to
        # spread the swarm. coll_pow=1 recovers the plain linear schedule.
        return self.eps_lo + (self.eps_hi - self.eps_lo) * (self.loss_ema ** self.coll_pow)

    def _scarcity(self, n_offer):
        # HARD ZK scarcity gate from observables only (swarm size m, offer size |S|): engage the
        # symmetry-breaking offset ONLY when the offer is at most scarcity_k x the swarm size, i.e.
        # when targets are scarce enough that collisions actually bind. When targets vastly outnumber
        # drones there is nothing to de-conflict, and a residual (especially time-varying) offset only
        # costs value and can re-collide (Theorem T7: symmetry-breaking must be FIXED). So we switch
        # it OFF entirely there and the method reduces to plain greedy CF. A hard gate (vs a smooth
        # taper) avoids leaving a small drifting offset that re-synchronizes look-alike drones.
        return 1.0 if n_offer <= self.scarcity_k * self.m else 0.0

    def select(self, t, cand):
        cand = np.asarray(cand)
        sc = self._scale() * self._scarcity(len(cand))
        score = self.U[cand] @ self.P[self.idx] + sc * self.dir[cand]
        a = int(cand[int(np.argmax(score))])
        self.pulled[a] = True
        return a

    def observe(self, t, choices, revealed, cand_sets, rvar):
        lost = 1.0 if (self.idx < len(choices) and int(choices[self.idx]) == -1) else 0.0
        self.loss_ema = (1.0 - self.lr) * self.loss_ema + self.lr * lost
        super().observe(t, choices, revealed, cand_sets, rvar)


def run_episode(Cls, hp, world, T, p_share, seed, sigma_own, sigma_obs, cand):
    P, U, R = world; m, n = R.shape; d = P.shape[1]
    rng = np.random.RandomState(seed + 999)
    learners = [Cls(m, n, d, i, seed + 7 * i + 1, **hp) for i in range(m)]
    for t in range(T):
        cand_sets = [rng.choice(n, size=cand, replace=False) for _ in range(m)]
        choices = np.array([learners[i].select(t, cand_sets[i]) for i in range(m)])
        true_r = np.array([R[i, choices[i]] for i in range(m)])
        for i in range(m):
            revealed = np.full(m, np.nan); rvar = np.full(m, np.inf)
            revealed[i] = true_r[i] + rng.normal(0, sigma_own); rvar[i] = sigma_own ** 2
            for k in range(m):
                if k != i and rng.rand() < p_share:
                    revealed[k] = true_r[k] + rng.normal(0, sigma_obs); rvar[k] = sigma_obs ** 2
            learners[i].observe(t, choices, revealed, cand_sets, rvar)
    preds = [learners[i].predict_scores() for i in range(m)]
    g = np.random.RandomState(seed + 555); got = []
    for _ in range(100):
        for k in range(m):
            off = g.choice(n, size=cand, replace=False)
            got.append(R[k, off[int(np.argmax(preds[k][off]))]])
    return float(np.mean(got))


def skill_ms(Cls, hp, cfg, m, n, d, T, cand, so, sb, seeds):
    sk = []
    for s in seeds:
        w = make_world_struct(m, n, d, cfg["structure"], cfg["eps"], cfg["n_clusters"], cfg["sharp"], s)
        g = run_episode(Cls, hp, w, T, 1.0, s, so, sb, cand)
        oc, rd = oracle_rand(w[2], s, cand)
        sk.append((g - rd) / max(oc - rd, 1e-6))
    return float(np.mean(sk)), float(np.std(sk))


def main():
    m, n, d, T, cand = 30, 240, 5, 50, 20
    sigma_own = 0.10
    seeds = list(range(5))
    cfg = dict(structure="cluster_gauss", eps=0.10, n_clusters=15, sharp=1.0)
    base = dict(eps0=0.5, eps_min=0.05, eps_decay=0.93, als_sweeps=5, refit_every=3)
    tab = dict(eps0=0.5, eps_min=0.05, eps_decay=0.93)
    rew = dict(**base)
    ch_n = dict(comp=False, s2c=0.2, n_neg=1, within=True, T_total=T, **base)
    ch_c = dict(comp=True, s2c=0.2, n_neg=1, within=True, warm_frac=0.3, T_total=T, **base)

    print("=" * 100)
    print("APPLES-TO-APPLES: cross-agent channel = noisy rewards vs clean choices (p=1).")
    print("own reward clean (sigma_own=%.2f) for ALL. m=%d n=%d d=%d T=%d cand=%d %d seeds. clustG nc15"
          % (sigma_own, m, n, d, T, cand, len(seeds)))
    print("reward-class: Tabular, RewardCF.  choice-class: ChoiceCF_naive, ChoiceCF_comp.")
    print("=" * 100)
    print(f"{'sig_obs':>7s} | {'Tabular':>11s} {'RewardCF':>11s} | {'Ch_naive':>11s} {'Ch_comp':>11s} | "
          f"{'comp-Rew':>8s}")
    print("-" * 100)
    for sb in [0.10, 0.30, 0.60, 1.00, 2.00]:
        tm, ts = skill_ms(Tabular, tab, cfg, m, n, d, T, cand, sigma_own, sb, seeds)
        rm, rs = skill_ms(RewardCF, rew, cfg, m, n, d, T, cand, sigma_own, sb, seeds)
        nm, ns = skill_ms(ChoiceCF, ch_n, cfg, m, n, d, T, cand, sigma_own, sb, seeds)
        cm, cs = skill_ms(ChoiceCF, ch_c, cfg, m, n, d, T, cand, sigma_own, sb, seeds)
        print(f"{sb:7.2f} | {tm:6.3f}+-{ts:4.3f} {rm:6.3f}+-{rs:4.3f} | "
              f"{nm:6.3f}+-{ns:4.3f} {cm:6.3f}+-{cs:4.3f} | {cm-rm:+8.3f}")
    print("-" * 100)
    print("HEADLINE = comp-Rew (ChoiceCF_comp minus RewardCF), same own-info, differ only in")
    print("cross-agent channel. Positive at high sigma_obs => clean choices beat noisy rewards.")


if __name__ == "__main__":
    main()
