"""
Phase A, GATE #1 (make-or-break): is U recoverable from CHOICES alone?

Isolates the single question the thesis rests on, away from online bandit
confounds: given only a stream of CHOICES (drone k, offered a subset, picked
target c), with NO reward information, can we recover factors P_hat, U_hat such
that P_hat_k . U_hat ranks drone k's true rewards R[k, :] correctly?

Clean best case: choosers are COMPETENT (pick true argmax over offered subset).
If recovery fails here, thesis is dead. If it succeeds, the online learner is an
engineering problem downstream.

Recovery via BPR (collaborative ranking) full-batch gradient + momentum,
VECTORISED (np.add.at). Two negative-sampling variants:
  - within-subset negatives: c preferred over the OTHER offered targets
    (requires observing others' offered sets).
  - global-random negatives: c preferred over random targets (choices only,
    minimal observation).
Baselines: reward-MF (ALS on actual rewards) = upper bound; random = floor.

Metrics (per drone, averaged): Spearman(pred, true row); top-10 overlap;
greedy reward (argmax pred in a random subset).
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


def gen_records(R, rounds, cand, seed, noise=0.0):
    m, n = R.shape
    rng = np.random.RandomState(seed + 4321)
    K, C, OFF = [], [], []
    for _ in range(rounds):
        for k in range(m):
            off = rng.choice(n, size=cand, replace=False)
            vals = R[k, off] + (noise * rng.randn(cand) if noise > 0 else 0.0)
            C.append(off[int(np.argmax(vals))]); K.append(k); OFF.append(off)
    return np.array(K), np.array(C), OFF


def make_pairs(K, C, OFF, n, within, seed):
    rng = np.random.RandomState(seed + 13)
    Ka, POS, NEG = [], [], []
    for k, c, off in zip(K, C, OFF):
        if within:
            for o in off:
                if o != c:
                    Ka.append(k); POS.append(c); NEG.append(o)
        else:
            for _ in range(len(off) - 1):
                o = rng.randint(n)
                while o == c:
                    o = rng.randint(n)
                Ka.append(k); POS.append(c); NEG.append(o)
    return np.array(Ka), np.array(POS), np.array(NEG)


def recover_bpr(m, n, d, K, POS, NEG, iters, lr, reg, seed, beta=0.9, init=0.1):
    rng = np.random.RandomState(seed)
    P = rng.normal(0, init, (m, d)); U = rng.normal(0, init, (n, d))
    vP = np.zeros_like(P); vU = np.zeros_like(U)
    npairs = max(len(K), 1)
    for _ in range(iters):
        diff = U[POS] - U[NEG]
        x = np.einsum('ij,ij->i', P[K], diff)
        g = sigmoid(-x)
        Pupd = np.zeros_like(P); np.add.at(Pupd, K, g[:, None] * diff)
        Uupd = np.zeros_like(U)
        np.add.at(Uupd, POS, g[:, None] * P[K])
        np.add.at(Uupd, NEG, -g[:, None] * P[K])
        gP = Pupd / npairs - reg * P
        gU = Uupd / npairs - reg * U
        vP = beta * vP + gP; vU = beta * vU + gU
        P += lr * vP; U += lr * vU
    return P, U


def recover_reward_als(m, n, d, K, C, R, sweeps, reg, seed, sigma=0.1):
    rng = np.random.RandomState(seed)
    P = rng.normal(0, 0.1, (m, d)); U = rng.normal(0, 0.1, (n, d))
    by_j = [[] for _ in range(n)]; by_k = [[] for _ in range(m)]
    for k, c in zip(K, C):
        r = R[k, c] + sigma * rng.randn()
        by_j[c].append((k, r)); by_k[k].append((c, r))
    I = reg * np.eye(d)
    for _ in range(sweeps):
        for j in range(n):
            if not by_j[j]:
                continue
            A = I.copy(); b = np.zeros(d)
            for (k, r) in by_j[j]:
                A += np.outer(P[k], P[k]); b += r * P[k]
            U[j] = np.linalg.solve(A, b)
        for k in range(m):
            if not by_k[k]:
                continue
            A = I.copy(); b = np.zeros(d)
            for (j, r) in by_k[k]:
                A += np.outer(U[j], U[j]); b += r * U[j]
            P[k] = np.linalg.solve(A, b)
    return P, U


def evaluate(P_hat, U_hat, R, seed, cand=12, greedy_rounds=100):
    m, n = R.shape
    sp, top = [], []
    for k in range(m):
        pred = U_hat @ P_hat[k]
        sp.append(spearman(pred, R[k]))
        tt = set(np.argsort(R[k])[-10:]); pt = set(np.argsort(pred)[-10:])
        top.append(len(tt & pt) / 10.0)
    rng = np.random.RandomState(seed + 77); gr = []
    for _ in range(greedy_rounds):
        for k in range(m):
            off = rng.choice(n, size=cand, replace=False)
            gr.append(R[k, off[int(np.argmax(U_hat[off] @ P_hat[k]))]])
    return np.mean(sp), np.mean(top), np.mean(gr)


def oracle_greedy(R, seed, cand=12, rounds=100):
    m, n = R.shape; rng = np.random.RandomState(seed + 77); g = []
    for _ in range(rounds):
        for k in range(m):
            off = rng.choice(n, size=cand, replace=False); g.append(R[k, off].max())
    return np.mean(g)


def main():
    m, n, d, mode_eps, cand = 30, 120, 5, 0.25, 12
    seeds = list(range(3))
    print("=" * 90)
    print("GATE #1: is U recoverable from CHOICES alone? (competent choosers)")
    print(f"m={m}, n={n}, d={d}, subset={cand}, {len(seeds)} seeds")
    print("=" * 90)

    for rounds in [25, 50, 100, 200]:
        acc = {k: [[], [], []] for k in ["BPR-within", "BPR-global", "REWARD-ALS", "random"]}
        oc = []
        for s in seeds:
            P, U, R = make_world(m, n, d, mode_eps, s)
            K, C, OFF = gen_records(R, rounds, cand, s)
            # within-subset negatives
            Kw, Pw, Nw = make_pairs(K, C, OFF, n, within=True, seed=s)
            Pb, Ub = recover_bpr(m, n, d, Kw, Pw, Nw, iters=600, lr=2.0, reg=1e-2, seed=s)
            # global negatives
            Kg, Pg, Ng = make_pairs(K, C, OFF, n, within=False, seed=s)
            Pb2, Ub2 = recover_bpr(m, n, d, Kg, Pg, Ng, iters=600, lr=2.0, reg=1e-2, seed=s)
            # reward upper bound
            Pr, Ur = recover_reward_als(m, n, d, K, C, R, sweeps=20, reg=1e-2, seed=s)
            rng = np.random.RandomState(s)
            P0 = rng.normal(0, 0.1, (m, d)); U0 = rng.normal(0, 0.1, (n, d))
            for nm, (PP, UU) in [("BPR-within", (Pb, Ub)), ("BPR-global", (Pb2, Ub2)),
                                 ("REWARD-ALS", (Pr, Ur)), ("random", (P0, U0))]:
                a = evaluate(PP, UU, R, s, cand)
                for i in range(3):
                    acc[nm][i].append(a[i])
            oc.append(oracle_greedy(R, s, cand))
        print(f"\nrounds/drone={rounds}  (total choices={m*rounds})  oracle greedy={np.mean(oc):.3f}")
        print(f"  {'method':12s}  {'Spearman':>9s}  {'top10':>7s}  {'greedy':>8s}")
        for nm in ["REWARD-ALS", "BPR-within", "BPR-global", "random"]:
            sp, tp, gr = (np.mean(acc[nm][i]) for i in range(3))
            print(f"  {nm:12s}  {sp:9.3f}  {tp:7.2f}  {gr:8.3f}")

    print("\n" + "=" * 90)
    print("READ: BPR approaching REWARD-ALS => U recoverable from choices => thesis viable.")
    print("BPR near 'random' => identifiability fails.")
    print("=" * 90)


if __name__ == "__main__":
    main()
