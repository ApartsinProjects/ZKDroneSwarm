"""CORE world model: low-rank (d) robot x task compatibility reward.

REWARD MODEL: dot product of L2-UNIT-normalized factors = cosine similarity.
  - NON-NEGATIVE factors -> reward in [0,1] (compatibility >= 0; no negatives).
  - UNIT norm -> no magnitude/popularity scaling artifact.
  - BILINEAR (no nonlinear link) -> genuinely rank <= d (a nonlinear g(<p,u>)
    would raise effective rank and break the low-rank assumption CF relies on).
  - reward = SIMILARITY (high when aligned), not distance.

WORLD MODEL (make_world `model` argument):
  model="uniform" (DEFAULT): i.i.d. Gaussian latent traits, L2-normalized to the
    unit sphere (uniform on the sphere); NO discrete types. R = P U^T is rank<=d
    with no cluster structure -- every robot and task is individually distinct.
  model="block": the K1 drone-types x K2 target-types block model (type
    prototypes A, B + within-type jitter); effective rank ~ min(d, K1, K2). Kept
    for reproducibility / parity checks. The K1/K2/within knobs apply only here.

MATRICES (block model only):
  A (K1,d) drone-TYPE prototypes;  B (K2,d) target-TYPE prototypes
  C (K1,K2) = A @ B.T   type-compatibility (rank <= d)
  t (m,) drone type ids;  s (n,) target type ids
  P (m,d), U (n,d) latent traits;  R (m,n) = P @ U.T  true reward.
Knobs: m, n, d (rank), K1, K2 (block granularity), within (within-type spread),
signed (False -> [0,1] non-negative; True -> cosine [-1,1]), model.

OBSERVABILITY MAPPING (by data type):
  CHOICES (discrete action)  -> MASKING (observe a teammate's choice or not)
  REWARDS (continuous value) -> ADDITIVE NOISE (noisy measurement)
"""
import numpy as np


def _unit(X, nonneg):
    if nonneg:
        X = np.abs(X)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def make_world(m, n, d, K1, K2, within=0.15, seed=0, signed=False, model="uniform"):
    """Hidden low-rank reward R = P U^T from L2-unit-normalized latent traits.
    model="uniform" (default): i.i.d. Gaussian traits on the unit sphere, no types
    (K1/K2/within ignored). model="block": K1 x K2 type block model. See module docstring."""
    rng = np.random.RandomState(seed)
    nonneg = not signed
    if model == "uniform":
        P = _unit(rng.randn(m, d), nonneg)          # i.i.d. on the unit sphere (no types)
        U = _unit(rng.randn(n, d), nonneg)
        R = P @ U.T
        meta = dict(t=None, s=None, A=None, B=None, C=None,
                    K1=K1, K2=K2, d=d, signed=signed, model="uniform")
        return P, U, R, meta
    A = _unit(rng.randn(K1, d), nonneg)         # drone-type prototypes
    B = _unit(rng.randn(K2, d), nonneg)         # target-type prototypes
    t = rng.randint(0, K1, m); s = rng.randint(0, K2, n)
    t[:K1] = np.arange(K1); s[:K2] = np.arange(K2)   # ensure every type present
    rng.shuffle(t); rng.shuffle(s)
    P = _unit(A[t] + within * rng.randn(m, d), nonneg)
    U = _unit(B[s] + within * rng.randn(n, d), nonneg)
    R = P @ U.T
    C = A @ B.T
    meta = dict(t=t, s=s, A=A, B=B, C=C, K1=K1, K2=K2, d=d, signed=signed, model="block")
    return P, U, R, meta


def effective_rank(R, thresh=0.99):
    sv = np.linalg.svd(R, compute_uv=False)
    cum = np.cumsum(sv ** 2) / np.sum(sv ** 2)
    return int(np.searchsorted(cum, thresh) + 1)


# --- observability helpers (by data type) ---
def reward_noise(true_val, sigma, rng):
    """REWARD observability = additive noise (continuous)."""
    return true_val + rng.normal(0, sigma)


def choice_visible(rho_i, rng):
    """CHOICE observability = masking (discrete: visible or not)."""
    return rng.rand() < rho_i


class OraclePolicy:
    """CEILING baseline: CENTRALIZED decision + COMPLETE information. Knows the
    true R; for each drone picks the best target in its offered subset. In the
    non-contention setting (targets do not deplete; drones may overlap) this IS
    the centralized optimum and equals the per-round best-in-subset used to
    normalize 'skill'. Under an ASSIGNMENT CONSTRAINT (each target once /
    depletion) the centralized optimum becomes bipartite matching (Hungarian) and
    centralized DECISION/coordination starts to matter -- a separate (future) axis.
    Report it EXPLICITLY as a baseline row (skill of oracle = 1.0 by definition;
    methods report their fraction of it)."""
    def __init__(self, R, **hp):
        self.R = R

    def best(self, k, cand):
        return cand[int(np.argmax(self.R[k, cand]))]


def explore_knob(t, T, eps0=1.0, eps_min=0.05, mode="decay", confidence=None):
    """Single continuous explore<->exploit control in [0,1] (unifies the two
    'phases'): high -> learn/explore, low -> exploit.
      mode='decay'    : scheduled linear decay eps0 -> eps_min over horizon T.
      mode='adaptive' : eps = 1 - confidence (explore when unsure; pairs with the
                        dual-source confidence -- factorization precision +
                        decision-alignment).
      mode='const'    : fixed eps0.
    """
    if mode == "adaptive" and confidence is not None:
        return float(np.clip(1.0 - confidence, eps_min, 1.0))
    if mode == "const":
        return float(eps0)
    frac = t / max(T - 1, 1)
    return float(max(eps_min, eps0 * (1.0 - frac) + eps_min * frac))


def _oracle_rand(R, seed, cand=20, rounds=80):
    m, n = R.shape
    rng = np.random.RandomState(seed + 7)
    oc, rd = [], []
    for _ in range(rounds):
        for k in range(m):
            off = rng.choice(n, size=cand, replace=False)
            oc.append(R[k, off].max()); rd.append(R[k, off].mean())
    return float(np.mean(oc)), float(np.mean(rd))


def _sanity():
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 88)
    print("CORE REWARD-MODEL SANITY (m=30 n=120). verify R range, effective rank, dynamic range")
    print("=" * 88)
    print(f"{'K1':>3s} {'K2':>3s} {'d':>3s} {'signed':>6s} | {'R.min':>6s} {'R.max':>6s} "
          f"{'R.mean':>6s} {'effRank':>7s} | {'oracle':>6s} {'rand':>5s} {'range':>5s}")
    print("-" * 88)
    for (K1, K2, d) in [(3, 3, 3), (5, 5, 5), (10, 10, 5), (10, 8, 3), (20, 20, 5)]:
        for signed in [False, True]:
            P, U, R, meta = make_world(30, 120, d, K1, K2, within=0.15, seed=0, signed=signed, model="block")
            assert np.allclose(R, P @ U.T)
            er = effective_rank(R)
            oc, rd = _oracle_rand(R, 0)
            print(f"{K1:3d} {K2:3d} {d:3d} {str(signed):>6s} | {R.min():6.2f} {R.max():6.2f} "
                  f"{R.mean():6.2f} {er:7d} | {oc:6.2f} {rd:5.2f} {oc-rd:5.2f}")
    print("-" * 88)
    print("CHECK (block): non-negative (signed=False) R in [0,1]; effRank ~ min(d,K1,K2).")
    print("-" * 88)
    print("UNIFORM world (default; no types, i.i.d. on the unit sphere):")
    for d in [3, 5]:
        for signed in [False, True]:
            P, U, R, meta = make_world(30, 120, d, 10, 10, seed=0, signed=signed)   # model="uniform" default
            er = effective_rank(R); oc, rd = _oracle_rand(R, 0)
            print(f"{'-':>3s} {'-':>3s} {d:3d} {str(signed):>6s} | {R.min():6.2f} {R.max():6.2f} "
                  f"{R.mean():6.2f} {er:7d} | {oc:6.2f} {rd:5.2f} {oc-rd:5.2f}")
    print("CHECK (uniform): effRank ~ d (no clusters); dynamic range (oracle-rand) healthy.")


if __name__ == "__main__":
    _sanity()
