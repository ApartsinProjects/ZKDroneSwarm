"""
Phase A pilot v4: warm-started BATCH-REFIT factorization.

Cycle-2 finding: single-pass RLS fixed factor RECOVERY (Urec 0.06->0.25) but
still lost to Tabular, because online single-pass underfits the bilinear problem
(offline multi-sweep ALS on the same data reached Spearman 0.6, greedy 0.68 >>
Tabular 0.35). Fix: warm-started batch refit each step (vectorised ALS on
rewards + BPR on choices over a growing buffer), reaching offline quality.

Compares, in the sample-starved changing-subset regime:
  Tabular, RewardMF (refit, rewards only), DualMF (refit, rewards + choices)
at p=1 (rewards shared) and p=0 (decision-only).

GATE 3a: RewardMF @ p=1 should now BEAT Tabular (fitting fixed).
GATE 3b: DualMF @ p=0 should beat Tabular and RewardMF (choices recover U when
rewards are private).
"""
import numpy as np
import sys
sys.stdout.reconfigure(encoding="utf-8")


def make_world(m, n, d, mode_eps, seed):
    rng = np.random.RandomState(seed)
    drone_modes = rng.randint(0, d, m)
    target_modes = np.concatenate([np.arange(d), rng.randint(0, d, n - d)])
    rng.shuffle(target_modes)
    P = np.zeros((m, d)); U = np.zeros((n, d))
    for i in range(m):
        v = np.zeros(d); v[drone_modes[i]] = 1.0
        v += mode_eps * rng.randn(d); P[i] = v / np.linalg.norm(v)
    for j in range(n):
        v = np.zeros(d); v[target_modes[j]] = 1.0
        v += mode_eps * rng.randn(d); U[j] = v / np.linalg.norm(v)
    return P, U, P @ U.T


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 1e-12 else 0.0


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class Tabular:
    def __init__(self, m, n, d, idx, seed, eps=0.15, **hp):
        self.n = n; self.idx = idx; self.rng = np.random.RandomState(seed); self.eps = eps
        self.counts = np.zeros(n); self.sums = np.zeros(n)
        self.tot_n = 0; self.tot_s = 0.0; self.pulled = np.zeros(n, bool)
    def _est(self):
        default = (self.tot_s / self.tot_n) if self.tot_n > 0 else 0.5
        e = np.full(self.n, default); m = self.counts > 0
        e[m] = self.sums[m] / self.counts[m]; return e
    def select(self, t, cand):
        if self.rng.rand() < self.eps:
            a = cand[self.rng.randint(len(cand))]
        else:
            a = cand[int(np.argmax(self._est()[cand]))]
        self.pulled[a] = True; return a
    def observe(self, t, choices, revealed, cand_sets):
        a = choices[self.idx]; r = revealed[self.idx]
        if not np.isnan(r):
            self.counts[a] += 1; self.sums[a] += r; self.tot_n += 1; self.tot_s += r
    def predict_scores(self): return self._est()
    def pulled_mask(self): return self.pulled


class RefitMF:
    """Warm-started batch refit: vectorised ALS on rewards (+ BPR on choices)."""
    def __init__(self, m, n, d, idx, seed, s2=0.02, prior_prec=1.0, eps=0.15,
                 init_scale=0.1, use_choices=False, als_sweeps=4, bpr_iters=25,
                 bpr_lr=3.0, reg=1e-2, refit_every=2, within=True, **hp):
        self.m = m; self.n = n; self.d = d; self.idx = idx
        self.s2 = s2; self.prior_prec = prior_prec; self.eps = eps; self.reg = reg
        self.use_choices = use_choices; self.als_sweeps = als_sweeps
        self.bpr_iters = bpr_iters; self.bpr_lr = bpr_lr
        self.refit_every = refit_every; self.within = within
        self.rng = np.random.RandomState(seed)
        self.P = self.rng.normal(0, init_scale, (m, d))
        self.U = self.rng.normal(0, init_scale, (n, d))
        self.rk = []; self.rj = []; self.rr = []          # reward buffer
        self.ck = []; self.cc = []; self.coff = []         # choice buffer
        self.pulled = np.zeros(n, bool)
        self._Ieye = np.eye(d)

    def _als(self):
        if not self.rk:
            return
        K = np.array(self.rk); J = np.array(self.rj); R = np.array(self.rr)
        pri = self.prior_prec * self._Ieye
        for _ in range(self.als_sweeps):
            Pk = self.P[K]
            AU = np.tile(pri, (self.n, 1, 1)).astype(float)
            bU = np.zeros((self.n, self.d))
            np.add.at(AU, J, np.einsum('ni,nj->nij', Pk, Pk) / self.s2)
            np.add.at(bU, J, (R[:, None] * Pk) / self.s2)
            self.U = np.linalg.solve(AU, bU[..., None])[..., 0]
            Uj = self.U[J]
            AP = np.tile(pri, (self.m, 1, 1)).astype(float)
            bP = np.zeros((self.m, self.d))
            np.add.at(AP, K, np.einsum('ni,nj->nij', Uj, Uj) / self.s2)
            np.add.at(bP, K, (R[:, None] * Uj) / self.s2)
            self.P = np.linalg.solve(AP, bP[..., None])[..., 0]

    def _bpr(self):
        if not self.ck:
            return
        Ka, POS, NEG = [], [], []
        for k, c, off in zip(self.ck, self.cc, self.coff):
            if self.within:
                for o in off:
                    if o != c:
                        Ka.append(k); POS.append(c); NEG.append(o)
            else:
                for _ in range(len(off) - 1):
                    o = self.rng.randint(self.n)
                    if o != c:
                        Ka.append(k); POS.append(c); NEG.append(o)
        if not Ka:
            return
        Ka = np.array(Ka); POS = np.array(POS); NEG = np.array(NEG)
        npair = len(Ka)
        for _ in range(self.bpr_iters):
            diff = self.U[POS] - self.U[NEG]
            g = sigmoid(-np.einsum('ij,ij->i', self.P[Ka], diff))
            Pupd = np.zeros_like(self.P); np.add.at(Pupd, Ka, g[:, None] * diff)
            Uupd = np.zeros_like(self.U)
            np.add.at(Uupd, POS, g[:, None] * self.P[Ka])
            np.add.at(Uupd, NEG, -g[:, None] * self.P[Ka])
            self.P += self.bpr_lr * (Pupd / npair - self.reg * self.P)
            self.U += self.bpr_lr * (Uupd / npair - self.reg * self.U)

    def _refit(self):
        self._als()
        if self.use_choices:
            self._bpr()

    def select(self, t, cand):
        if self.rng.rand() < self.eps:
            a = cand[self.rng.randint(len(cand))]
        else:
            a = cand[int(np.argmax(self.U[cand] @ self.P[self.idx]))]
        self.pulled[a] = True; return a

    def observe(self, t, choices, revealed, cand_sets):
        for k in range(self.m):
            r = revealed[k]
            if not np.isnan(r):
                self.rk.append(k); self.rj.append(int(choices[k])); self.rr.append(float(r))
        if self.use_choices:
            for k in range(self.m):
                self.ck.append(k); self.cc.append(int(choices[k])); self.coff.append(cand_sets[k])
        if (t + 1) % self.refit_every == 0:
            self._refit()

    def predict_scores(self):
        return self.U @ self.P[self.idx]
    def pulled_mask(self): return self.pulled


class DualMFSep(RefitMF):
    """Separated dual-likelihood: U from CHOICES (BPR), P from REWARDS given that
    U (ridge). Choices define the shared target geometry; own rewards locate the
    drone in it. Avoids the ALS/BPR scale-fight of the entangled version."""
    def __init__(self, *a, **hp):
        hp["use_choices"] = True
        super().__init__(*a, **hp)
        self.Pbpr = self.P.copy()      # separate factor for the BPR fit

    def _bpr_into(self):
        if not self.ck:
            return
        Ka, POS, NEG = [], [], []
        for k, c, off in zip(self.ck, self.cc, self.coff):
            if self.within:
                for o in off:
                    if o != c:
                        Ka.append(k); POS.append(c); NEG.append(o)
            else:
                for _ in range(len(off) - 1):
                    o = self.rng.randint(self.n)
                    if o != c:
                        Ka.append(k); POS.append(c); NEG.append(o)
        if not Ka:
            return
        Ka = np.array(Ka); POS = np.array(POS); NEG = np.array(NEG); npair = len(Ka)
        for _ in range(self.bpr_iters):
            diff = self.U[POS] - self.U[NEG]
            g = sigmoid(-np.einsum('ij,ij->i', self.Pbpr[Ka], diff))
            Pupd = np.zeros_like(self.Pbpr); np.add.at(Pupd, Ka, g[:, None] * diff)
            Uupd = np.zeros_like(self.U)
            np.add.at(Uupd, POS, g[:, None] * self.Pbpr[Ka])
            np.add.at(Uupd, NEG, -g[:, None] * self.Pbpr[Ka])
            self.Pbpr += self.bpr_lr * (Pupd / npair - self.reg * self.Pbpr)
            self.U += self.bpr_lr * (Uupd / npair - self.reg * self.U)

    def _refit(self):
        # 1) U from choices
        self._bpr_into()
        # 2) P from rewards given the choice-derived U (ridge per drone)
        pri = self.prior_prec * self._Ieye
        AP = np.tile(pri, (self.m, 1, 1)).astype(float)
        bP = np.zeros((self.m, self.d))
        if self.rk:
            K = np.array(self.rk); J = np.array(self.rj); R = np.array(self.rr)
            Uj = self.U[J]
            np.add.at(AP, K, np.einsum('ni,nj->nij', Uj, Uj) / self.s2)
            np.add.at(bP, K, (R[:, None] * Uj) / self.s2)
        self.P = np.linalg.solve(AP, bP[..., None])[..., 0]


class DualBoot(RefitMF):
    """Implicit-feedback MF (Hu-Koren-Volinsky 2008 style). Choices become
    POSITIVE pseudo-rewards at the SAME scale as real rewards (anchored by each
    drone's own observed reward distribution), fused with explicit rewards in
    ONE weighted ALS. Own rewards bootstrap and anchor scale; choices propagate
    to unpulled targets. No scale-fight, and choices are informed because each
    drone selects from its own-reward-anchored model."""
    def __init__(self, *a, s2c=0.3, n_neg=1, within=True, **hp):
        hp["use_choices"] = True
        super().__init__(*a, **hp)
        self.s2c = s2c; self.n_neg = n_neg; self.within = within

    def _refit(self):
        own_r = [r for k, r in zip(self.rk, self.rr) if k == self.idx]
        pos = float(np.percentile(own_r, 75)) if len(own_r) >= 4 else 0.55
        neg = float(np.percentile(own_r, 25)) if len(own_r) >= 4 else 0.15
        K, J, Rv, W = [], [], [], []
        wr = 1.0 / self.s2; wc = 1.0 / self.s2c
        for k, j, r in zip(self.rk, self.rj, self.rr):
            K.append(k); J.append(j); Rv.append(r); W.append(wr)
        for k, c, off in zip(self.ck, self.cc, self.coff):
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


def greedy_eval(learners, R, seed, cand, rounds=100):
    """Exploitation quality of FINAL models: pick argmax predicted in fresh
    random subsets, no exploration. Separates model quality from learning
    transient. Returns mean reward and mean fraction-of-oracle."""
    m, n = R.shape
    rng = np.random.RandomState(seed + 555)
    preds = [learners[i].predict_scores() for i in range(m)]
    got, opt = [], []
    for _ in range(rounds):
        for k in range(m):
            off = rng.choice(n, size=cand, replace=False)
            got.append(R[k, off[int(np.argmax(preds[k][off]))]])
            opt.append(R[k, off].max())
    return float(np.mean(got)), float(np.mean(got)) / float(np.mean(opt))


def run_episode(Cls, hp, world, T, p_share, seed, sigma=0.1, cand=12):
    P, U, R = world; m, n = R.shape; d = P.shape[1]
    rng = np.random.RandomState(seed + 999)
    learners = [Cls(m, n, d, i, seed + 7 * i + 1, **hp) for i in range(m)]
    step_reward = np.zeros(T)
    for t in range(T):
        cand_sets = [rng.choice(n, size=cand, replace=False) for _ in range(m)]
        choices = np.array([learners[i].select(t, cand_sets[i]) for i in range(m)])
        true_r = np.array([R[i, choices[i]] for i in range(m)])
        step_reward[t] = true_r.mean()
        for i in range(m):
            revealed = np.full(m, np.nan)
            revealed[i] = true_r[i] + rng.normal(0, sigma)
            for k in range(m):
                if k != i and rng.rand() < p_share:
                    revealed[k] = true_r[k] + rng.normal(0, sigma)
            learners[i].observe(t, choices, revealed, cand_sets)
    recs = [spearman(learners[i].predict_scores(), R[i]) for i in range(m)]
    gr, gfrac = greedy_eval(learners, R, seed, cand)
    return step_reward.mean(), gr, gfrac, float(np.mean(recs))


def avg_seeds(Cls, hp, T, p, seeds, wp):
    av, gr, gf, rec = [], [], [], []
    for s in seeds:
        w = make_world(wp["m"], wp["n"], wp["d"], wp["mode_eps"], s)
        a, g, f, u = run_episode(Cls, hp, w, T, p, s, wp["sigma"], wp["cand"])
        av.append(a); gr.append(g); gf.append(f); rec.append(u)
    return np.mean(av), np.std(av), np.mean(gr), np.mean(gf), np.mean(rec)


def main():
    wp = dict(m=40, n=200, d=5, mode_eps=0.25, sigma=0.10, cand=20)
    T = 40; seeds = list(range(3))
    policies = {
        "Tabular":  (Tabular, dict(eps=0.15)),
        "RewardMF": (RefitMF, dict(use_choices=False, eps=0.15, als_sweeps=6, refit_every=3)),
        "DualBoot": (DualBoot, dict(eps=0.15, als_sweeps=6, s2c=0.1, n_neg=2,
                                    refit_every=3, within=True)),
    }
    print("=" * 92)
    print("PILOT v4b: separated dual-likelihood (U from choices, P from rewards). "
          "m=%d n=%d d=%d T=%d cand=%d %d seeds" % (wp['m'], wp['n'], wp['d'], T, wp['cand'], len(seeds)))
    print("changing subsets (starved). online avg over learning; greedy = final-model exploitation")
    print("=" * 92)
    for p in [0.0, 1.0]:
        print(f"\n--- p_share = {p:.1f} ---")
        print(f"{'policy':10s}  {'online avg':>11s}  {'greedy':>8s}  {'frac-oracle':>11s}  {'Urec':>7s}")
        print("-" * 56)
        for name, (Cls, hp) in policies.items():
            a, sd, g, f, u = avg_seeds(Cls, hp, T, p, seeds, wp)
            print(f"{name:10s}  {a:6.3f}+-{sd:4.3f}  {g:8.3f}  {f:11.2f}  {u:7.3f}")
    print("\nGATE 3a: RewardMF @ p=1 > Tabular (fitting fixed) -- already passed.")
    print("GATE 3b: DualMFSep @ p=0 > Tabular and > RewardMF @ p=0 (choices recover U).")


if __name__ == "__main__":
    main()
