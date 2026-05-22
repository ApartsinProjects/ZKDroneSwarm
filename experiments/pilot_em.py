"""
Principled joint model: EM over (latent factors P,U) + (per-drone competence
gamma_k) + (per-choice rational/random responsibility z).

Generative model:
  rewards:  r_{k,j} ~ N(<p_k,u_j>, sigma^2)            (always reliable)
  choices:  with prob gamma_k -> rational softmax over offered set
            with prob 1-gamma_k -> uniform random over offered set
  priors:   p_k,u_j ~ N(0, lambda^-1 I);  gamma_k ~ Beta(a,b)

EM:
  E-step:  resp r_{k,t} = gamma_k p_soft(c) / (gamma_k p_soft(c) + (1-gamma_k)/|O|)
  M-step:  gamma_k = (sum_t r + a-1)/(n_k + a+b-2)
           P,U <- maximize reward loglik + sum r_{k,t} * choice loglik
                  (folded in as responsibility-weighted implicit-feedback ALS)

Bootstrap-safe: when u_c is unknown, p_soft(c)~1/|O| so resp -> gamma_k (the
chooser's competence), NOT zero -> a competent drone's choice of an UNSEEN
target still propagates. Rewards anchor; EM responsibilities quarantine random
exploratory choices; gamma_k is the learned explore/exploit knob.

Compares at p=0 (decision-only): Tabular, RewardMF (own rewards only),
DualBoot (unweighted choices), DualConf (heuristic weights), DualEM (this).
Reference: RewardMF @ p=1 (CF ceiling if rewards were shared).
"""
import numpy as np
import sys
sys.stdout.reconfigure(encoding="utf-8")

from pilot_refit import Tabular, RefitMF, DualBoot, run_episode
from pilot_structure import make_world_struct, oracle_rand
from pilot_bootstrap import DualConf


def _softmax_rows(X):
    X = X - X.max(axis=1, keepdims=True)
    e = np.exp(X)
    return e / e.sum(axis=1, keepdims=True)


class DualEM(RefitMF):
    def __init__(self, *a, s2c=0.2, n_neg=1, tau=0.3, em_iters=4, inner_sweeps=2,
                 a_beta=1.0, b_beta=1.0, eps0=0.5, eps_min=0.05, eps_decay=0.93, **hp):
        hp["use_choices"] = True
        super().__init__(*a, **hp)
        self.s2c = s2c; self.n_neg = n_neg; self.tau = tau
        self.em_iters = em_iters; self.inner_sweeps = inner_sweeps
        self.a = a_beta; self.b = b_beta
        self.eps0 = eps0; self.eps_min = eps_min; self.eps_decay = eps_decay
        self.eps = eps0
        self.gamma = np.full(self.m, 0.5)

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
            self.ck.append(k); self.cc.append(int(choices[k])); self.coff.append(cand_sets[k])
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
        if (t + 1) % self.refit_every == 0:
            self._refit()

    def _weighted_als(self, K, J, Rv, W):
        pri = self.prior_prec * self._Ieye
        for _ in range(self.inner_sweeps):
            Pk = self.P[K]
            AU = np.tile(pri, (self.n, 1, 1)).astype(float); bU = np.zeros((self.n, self.d))
            np.add.at(AU, J, W[:, None, None] * np.einsum('ni,nj->nij', Pk, Pk))
            np.add.at(bU, J, (W * Rv)[:, None] * Pk)
            self.U = np.linalg.solve(AU, bU[..., None])[..., 0]
            Uj = self.U[J]
            AP = np.tile(pri, (self.m, 1, 1)).astype(float); bP = np.zeros((self.m, self.d))
            np.add.at(AP, K, W[:, None, None] * np.einsum('ni,nj->nij', Uj, Uj))
            np.add.at(bP, K, (W * Rv)[:, None] * Uj)
            self.P = np.linalg.solve(AP, bP[..., None])[..., 0]

    def _refit(self):
        own_r = [r for k, r in zip(self.rk, self.rr) if k == self.idx]
        pos = float(np.percentile(own_r, 75)) if len(own_r) >= 4 else 0.55
        neg = float(np.percentile(own_r, 25)) if len(own_r) >= 4 else 0.15
        rK = np.array(self.rk); rJ = np.array(self.rj); rR = np.array(self.rr)
        has_ch = len(self.ck) > 0
        if has_ch:
            cK = np.array(self.ck); cC = np.array(self.cc); OFF = np.array(self.coff)
            pos_idx = (OFF == cC[:, None]).argmax(1)
            nc, cand = OFF.shape
        # init factors from rewards once (anchor) if essentially untrained
        for em in range(self.em_iters):
            resp = None
            if has_ch:
                Pk = self.P[cK]                                  # (nc,d)
                Uoff = self.U[OFF]                               # (nc,cand,d)
                scores = np.einsum('ncd,nd->nc', Uoff, Pk) / self.tau
                sm = _softmax_rows(scores)
                p_soft = sm[np.arange(nc), pos_idx]
                g = self.gamma[cK]
                resp = g * p_soft / (g * p_soft + (1.0 - g) / cand + 1e-12)
                # M-step gamma
                num = np.zeros(self.m); cnt = np.zeros(self.m)
                np.add.at(num, cK, resp); np.add.at(cnt, cK, 1.0)
                with np.errstate(invalid='ignore', divide='ignore'):
                    gm = (num + self.a - 1.0) / (cnt + self.a + self.b - 2.0)
                self.gamma = np.where(cnt > 0, np.clip(gm, 0.02, 0.98), self.gamma)
            # M-step factors: rewards + responsibility-weighted choices
            K = [rK]; J = [rJ]; Rv = [rR]; W = [np.full(len(rK), 1.0 / self.s2)]
            if has_ch:
                wc = resp / self.s2c
                K.append(cK); J.append(cC); Rv.append(np.full(nc, pos)); W.append(wc)
                for _ in range(self.n_neg):
                    negcol = self.rng.randint(cand, size=nc)
                    negcol = np.where(negcol == pos_idx, (negcol + 1) % cand, negcol)
                    negj = OFF[np.arange(nc), negcol]
                    K.append(cK); J.append(negj); Rv.append(np.full(nc, neg)); W.append(wc)
            K = np.concatenate(K); J = np.concatenate(J); Rv = np.concatenate(Rv); W = np.concatenate(W)
            if len(K) == 0:
                return
            self._weighted_als(K, J, Rv, W)


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
    em = dict(s2c=0.2, n_neg=1, tau=0.3, em_iters=4, inner_sweeps=2, refit_every=3,
              eps0=0.5, eps_min=0.05, eps_decay=0.93)

    print("=" * 100)
    print("EM JOINT MODEL vs baselines at p=0 (decision-only). world=clustG nc15 (low-rank).")
    print("m=%d d=%d T=%d cand=%d %d seeds. skill=(greedy-rand)/(oracle-rand)." % (m, d, T, cand, len(seeds)))
    print("=" * 100)
    print(f"{'n':>5s} {'cov%':>5s} | {'Tab0':>6s} {'Rew0':>6s} {'Boot0':>6s} {'Conf0':>6s} "
          f"{'EM0':>6s} | {'Rew1(ceil)':>10s} {'EM1':>6s}")
    print("-" * 100)
    for n in [120, 240, 480]:
        cov = 100.0 * min(1.0, T / n)
        t0 = skill(Tabular, tab, cfg, m, n, d, T, cand, sigma, 0.0, seeds)
        r0 = skill(RefitMF, rmf, cfg, m, n, d, T, cand, sigma, 0.0, seeds)
        b0 = skill(DualBoot, boot, cfg, m, n, d, T, cand, sigma, 0.0, seeds)
        c0 = skill(DualConf, conf, cfg, m, n, d, T, cand, sigma, 0.0, seeds)
        e0 = skill(DualEM, em, cfg, m, n, d, T, cand, sigma, 0.0, seeds)
        r1 = skill(RefitMF, rmf, cfg, m, n, d, T, cand, sigma, 1.0, seeds)
        e1 = skill(DualEM, em, cfg, m, n, d, T, cand, sigma, 1.0, seeds)
        print(f"{n:5d} {cov:4.0f}% | {t0:6.3f} {r0:6.3f} {b0:6.3f} {c0:6.3f} "
              f"{e0:6.3f} | {r1:10.3f} {e1:6.3f}")
    print("-" * 100)
    print("GOAL: EM0 > Tab0 (and > Boot0/Conf0). EM1 ~ Rew1 (no harm when rewards shared).")


if __name__ == "__main__":
    main()
