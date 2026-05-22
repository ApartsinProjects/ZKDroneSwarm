"""
Cracking the decision-only BOOTSTRAP with competence-weighted choices.

Prior cycles: at p=0 (rewards private), choices corrupt the factorization,
because early choices come from untrained (near-random) models -> implicit
feedback treats bad picks as positives. RS-style fix (Hu-Koren-Volinsky
confidence weighting + social-learning trust):

  1. per-drone DECAYING epsilon: explore early, exploit late.
  2. per-drone COMPETENCE estimate gamma_k, inferred from OBSERVABLE behaviour
     (no rewards): (a) time ramp (everyone accumulates own-reward knowledge at
     the same rate) x (b) choice CONSISTENCY (a converged/exploiting drone picks
     targets that cluster in U-space; cosine of consecutive chosen vectors).
  3. weight every drone's choice by gamma_k when folding it into the weighted
     ALS, so exploratory early choices are discounted instead of corrupting U.

Test: does competence-weighted DualConf beat Tabular and plain DualBoot at p=0,
especially in the VERY starved regime where tabular has the least room?
Reference: RewardMF @ p=1 (if rewards were shared) is the CF ceiling.
"""
import numpy as np
import sys
sys.stdout.reconfigure(encoding="utf-8")

from pilot_refit import Tabular, RefitMF, DualBoot, run_episode
from pilot_structure import make_world_struct, oracle_rand


class DualConf(RefitMF):
    def __init__(self, *a, s2c=0.2, n_neg=1, within=True,
                 eps0=0.5, eps_min=0.05, eps_decay=0.93,
                 warm_frac=0.3, T_total=50, use_consistency=True, **hp):
        hp["use_choices"] = True
        super().__init__(*a, **hp)
        self.s2c = s2c; self.n_neg = n_neg; self.within = within
        self.eps0 = eps0; self.eps_min = eps_min; self.eps_decay = eps_decay
        self.warm_frac = warm_frac; self.T_total = T_total
        self.use_consistency = use_consistency
        self.eps = eps0
        self.cstep = []
        self.last_vec = {}
        self.consist = np.full(self.m, 0.3)

    def select(self, t, cand):
        if self.rng.rand() < self.eps:
            a = cand[self.rng.randint(len(cand))]
        else:
            a = cand[int(np.argmax(self.U[cand] @ self.P[self.idx]))]
        self.pulled[a] = True
        return a

    def observe(self, t, choices, revealed, cand_sets):
        for k in range(self.m):
            r = revealed[k]
            if not np.isnan(r):
                self.rk.append(k); self.rj.append(int(choices[k])); self.rr.append(float(r))
        for k in range(self.m):
            c = int(choices[k])
            self.ck.append(k); self.cc.append(c); self.coff.append(cand_sets[k]); self.cstep.append(t)
            if k in self.last_vec:
                v0 = self.last_vec[k]; v1 = self.U[c]
                cs = float(v0 @ v1 / (np.linalg.norm(v0) * np.linalg.norm(v1) + 1e-9))
                self.consist[k] = 0.9 * self.consist[k] + 0.1 * max(cs, 0.0)
            self.last_vec[k] = self.U[c].copy()
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
        if (t + 1) % self.refit_every == 0:
            self._refit()

    def _gamma(self, k, tmade):
        w0 = self.warm_frac * self.T_total
        ramp = np.clip((tmade - w0) / max(self.T_total - w0, 1.0), 0.0, 1.0)
        if self.use_consistency:
            ramp *= np.clip(self.consist[k], 0.0, 1.0)
        return ramp

    def _refit(self):
        own_r = [r for k, r in zip(self.rk, self.rr) if k == self.idx]
        pos = float(np.percentile(own_r, 75)) if len(own_r) >= 4 else 0.55
        neg = float(np.percentile(own_r, 25)) if len(own_r) >= 4 else 0.15
        K, J, Rv, W = [], [], [], []
        wr = 1.0 / self.s2
        for k, j, r in zip(self.rk, self.rj, self.rr):
            K.append(k); J.append(j); Rv.append(r); W.append(wr)
        for k, c, off, ts in zip(self.ck, self.cc, self.coff, self.cstep):
            g = self._gamma(k, ts)
            if g <= 1e-3:
                continue
            wc = g / self.s2c
            K.append(k); J.append(c); Rv.append(pos); W.append(wc)
            for _ in range(self.n_neg):
                o = int(self.rng.choice(off)) if self.within else self.rng.randint(self.n)
                if o != c:
                    K.append(k); J.append(o); Rv.append(neg); W.append(wc)
        if not K:
            return
        K = np.array(K); J = np.array(J); Rv = np.array(Rv); W = np.array(W)
        pri = self.prior_prec * self._Ieye
        for _ in range(self.als_sweeps):
            Pk = self.P[K]
            AU = np.tile(pri, (self.n, 1, 1)).astype(float)
            bU = np.zeros((self.n, self.d))
            np.add.at(AU, J, W[:, None, None] * np.einsum('ni,nj->nij', Pk, Pk))
            np.add.at(bU, J, (W * Rv)[:, None] * Pk)
            self.U = np.linalg.solve(AU, bU[..., None])[..., 0]
            Uj = self.U[J]
            AP = np.tile(pri, (self.m, 1, 1)).astype(float)
            bP = np.zeros((self.m, self.d))
            np.add.at(AP, K, W[:, None, None] * np.einsum('ni,nj->nij', Uj, Uj))
            np.add.at(bP, K, (W * Rv)[:, None] * Uj)
            self.P = np.linalg.solve(AP, bP[..., None])[..., 0]


def skill(Cls, hp, cfg, m, n, d, T, cand, sigma, p, seeds):
    sk = []
    for s in seeds:
        w = make_world_struct(m, n, d, cfg["structure"], cfg["eps"],
                              cfg["n_clusters"], cfg["sharp"], s)
        _, g, _, _ = run_episode(Cls, hp, w, T, p, s, sigma, cand)
        oc, rd = oracle_rand(w[2], s, cand)
        sk.append((g - rd) / max(oc - rd, 1e-6))
    return float(np.mean(sk))


def main():
    m, d, T, cand, sigma = 30, 5, 50, 20, 0.10
    seeds = list(range(3))
    cfg = dict(structure="cluster_gauss", eps=0.10, n_clusters=15, sharp=1.0)
    tab = dict(eps=0.15)
    rmf = dict(use_choices=False, eps=0.15, als_sweeps=5, refit_every=3)
    boot = dict(eps=0.15, als_sweeps=5, s2c=0.2, n_neg=1, refit_every=3, within=True)
    conf = dict(als_sweeps=5, s2c=0.2, n_neg=1, refit_every=3, within=True,
                eps0=0.5, eps_min=0.05, eps_decay=0.93, warm_frac=0.3,
                T_total=T, use_consistency=True)

    print("=" * 92)
    print("BOOTSTRAP TEST: competence-weighted choices (DualConf) at p=0 (decision-only).")
    print("world=cluster_gauss nc15 (low-rank, favorable). m=%d d=%d T=%d cand=%d %d seeds"
          % (m, d, T, cand, len(seeds)))
    print("skill=(greedy-rand)/(oracle-rand). RewardMF@p=1 = CF ceiling if rewards shared.")
    print("=" * 92)
    print(f"{'n':>5s} {'cover%':>7s} | {'Tab p0':>7s} {'RewMF p0':>8s} {'Boot p0':>8s} "
          f"{'Conf p0':>8s} | {'RewMF p1':>8s}")
    print("-" * 92)
    for n in [120, 240, 480]:
        cover = 100.0 * min(1.0, T / n)
        t0 = skill(Tabular, tab, cfg, m, n, d, T, cand, sigma, 0.0, seeds)
        r0 = skill(RefitMF, rmf, cfg, m, n, d, T, cand, sigma, 0.0, seeds)
        b0 = skill(DualBoot, boot, cfg, m, n, d, T, cand, sigma, 0.0, seeds)
        c0 = skill(DualConf, conf, cfg, m, n, d, T, cand, sigma, 0.0, seeds)
        r1 = skill(RefitMF, rmf, cfg, m, n, d, T, cand, sigma, 1.0, seeds)
        print(f"{n:5d} {cover:6.0f}% | {t0:7.3f} {r0:8.3f} {b0:8.3f} {c0:8.3f} | {r1:8.3f}")
    print("-" * 92)
    print("GOAL: Conf p0 > Tab p0 (and > Boot p0). If so, competence weighting cracks")
    print("the decision-only bootstrap -> choices transfer knowledge without rewards.")


if __name__ == "__main__":
    main()
