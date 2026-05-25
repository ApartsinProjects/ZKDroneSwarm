"""Is the low-rank reward EMERGENT (measured) rather than assumed by construction?

The body builds R = P U^T (low-rank by fiat). Here we instead ASSEMBLE the robot x task reward from a
physical heterogeneous multi-sensor detection model that never factors it, then MEASURE its singular
spectrum, and finally CLOSE THE LOOP by running the full masked-broadcast bake-off on that physical
reward to confirm SwarmCF still wins when low-rank is emergent.

Physical model (Proposal A). S sensing modalities. Robot i has non-negative sensor gains a_i in R^S
and position x_i; target j has a signature b_j in R^S and position y_j. Per-modality detection
SNR p_ijs = sigmoid(slope * (a_is b_js * falloff_ij - offset)) with a smooth range falloff
falloff_ij = 1/(1 + (||x_i-y_j||/r)^2). Detection quality (the reward) is the probability at least one
modality detects: R_ij = 1 - prod_s (1 - p_ijs). This is a product-of-misses with a nonlinear range
falloff, NOT an inner product, so nothing forces it to be rank-d.

Proposal B (robustness): a pure-geometry Gaussian coverage kernel R_ij = exp(-||x_i-y_j||^2/2 l^2).

Run from repo root:  python experiments/emergent_lowrank.py
"""
import os
import sys
import json
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from latentswarm.sweeps import base_config
from latentswarm.registry import ALGORITHMS, get
from latentswarm.env import ZKMRTAEnv
from latentswarm.metrics import UnseenPairSkillHeldout, bootstrap_ci

M, N = 30, 240
FIELD = 10.0
FIG_RANK = os.path.join(ROOT, "docs", "figures", "F_emergent_rank.png")
FIG_BAKE = os.path.join(ROOT, "docs", "figures", "F_emergent_bakeoff.png")
PDF_RANK = os.path.join(ROOT, "docs", "figures", "pdf", "F_emergent_rank.pdf")
PDF_BAKE = os.path.join(ROOT, "docs", "figures", "pdf", "F_emergent_bakeoff.pdf")


def physical_reward(rng, m, n, S, range_frac=2.0, gain=6.0, offset=0.12, conc=1.5):
    """Heterogeneous multi-sensor detection-quality reward, assembled from physics (not as P U^T).
    Each robot/target has a continuous non-negative profile over the S sensing modalities (softmax of a
    Gaussian, so the profiles are continuous, not discrete types). Per-modality detection probability is
    a rectified-linear function of the matched-filter SNR (gain * profile-product * range-falloff), so a
    modality only contributes once its SNR clears a threshold; detection quality is the probability that
    at least one modality detects, R_ij = 1 - prod_s (1 - p_ijs). A product-of-misses with a nonlinear
    range falloff, NOT an inner product.

    `conc` is the profile concentration (softmax temperature): conc -> 0 makes every profile the uniform
    1/S vector (no specialization), conc -> infinity makes them one-hot (S DISCRETE sensor/target types).
    The default 1.5 is the continuous, moderately-specialized middle (deliberately not discrete types)."""
    z = rng.randn(m, S); a = np.exp(conc * z); a /= a.sum(1, keepdims=True)
    w = rng.randn(n, S); b = np.exp(conc * w); b /= b.sum(1, keepdims=True)
    x = rng.uniform(0, FIELD, (m, 2)); y = rng.uniform(0, FIELD, (n, 2))
    dist = np.sqrt(((x[:, None, :] - y[None, :, :]) ** 2).sum(-1))     # (m, n)
    falloff = 1.0 / (1.0 + (dist / (range_frac * FIELD)) ** 2)         # smooth range attenuation
    miss = np.ones((m, n))
    for s in range(S):
        snr = np.outer(a[:, s], b[:, s]) * falloff
        p = np.clip(gain * (snr - offset), 0.0, 1.0)                   # rectified-linear detection prob
        miss *= (1.0 - p)
    return 1.0 - miss                                                  # detection quality in [0, 1]


def coverage_kernel(rng, m, n, ell_frac=0.3):
    x = rng.uniform(0, FIELD, (m, 2)); y = rng.uniform(0, FIELD, (n, 2))
    dist2 = ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)
    return np.exp(-dist2 / (2.0 * (ell_frac * FIELD) ** 2))


def eff_rank(R, energies=(0.90, 0.95, 0.99)):
    s = np.linalg.svd(R, compute_uv=False)
    cum = np.cumsum(s ** 2) / np.sum(s ** 2)
    return {q: int(np.searchsorted(cum, q) + 1) for q in energies}, s


def wide_factors(R):
    """Factor R = A diag(s) B^T into wide P, U with P U^T = R exactly (so the env reward IS the
    physical R, with no low-rank truncation imposed)."""
    A, s, Bt = np.linalg.svd(R, full_matrices=False)
    sq = np.sqrt(np.maximum(s, 0.0))
    return A * sq[None, :], Bt.T * sq[None, :]


def run_one(cfg, P, U, d_guess, seed, algo):
    env = ZKMRTAEnv(cfg, P, U, d_guess, seed=seed)
    pol = get(ALGORITHMS, algo)(cfg, cfg.m, cfg.n, d_guess, seed=1000 + seed)
    obs = env.reset()
    for _ in range(cfg.T):
        a = pol.act(obs); obs, _r, _i = env.step(a); pol.observe(obs)
    pred = pol.predict_rows()
    sk = UnseenPairSkillHeldout().compute(P=P, U=U, pred_rows=pred, engaged=env.engaged,
                                          rng=np.random.RandomState(10000 + seed), offer_size=cfg.offer_size)
    return float(sk) if sk is not None else None


def main():
    t0 = time.time()
    SEEDS = list(range(16))

    # ---- Part A: MEASURE the effective rank of the physically-assembled reward ----
    print("[emergent] Part A: effective rank of the physical sensing reward (m=%d, n=%d)" % (M, N), flush=True)
    rank_vs_S = {}
    for S in range(2, 9):
        q90 = []; q95 = []; q99 = []
        for s in SEEDS:
            R = physical_reward(np.random.RandomState(s), M, N, S, range_frac=2.0)
            er, _ = eff_rank(R)
            q90.append(er[0.90]); q95.append(er[0.95]); q99.append(er[0.99])
        rank_vs_S[S] = (float(np.mean(q90)), float(np.mean(q95)), float(np.mean(q99)))
    rank_vs_range = {}
    for rf in [0.15, 0.3, 0.5, 0.8, 1.5]:
        q95 = [eff_rank(physical_reward(np.random.RandomState(s), M, N, 4, range_frac=rf))[0][0.95] for s in SEEDS]
        rank_vs_range[rf] = float(np.mean(q95))
    rank_kernel = {}
    for lf in [0.15, 0.3, 0.5, 0.8]:
        q95 = [eff_rank(coverage_kernel(np.random.RandomState(s), M, N, ell_frac=lf))[0][0.95] for s in SEEDS]
        rank_kernel[lf] = float(np.mean(q95))
    # representative scree (physical, S=4, range 0.5)
    _, sv = eff_rank(physical_reward(np.random.RandomState(0), M, N, 4))
    Rstats = physical_reward(np.random.RandomState(0), M, N, 4)

    # ---- Part B: CLOSE THE LOOP -- masked-broadcast bake-off on the physical reward ----
    print("[emergent] Part B: bake-off on the physical reward (S=4, rho=0.25, %d seeds)" % len(SEEDS), flush=True)
    cfg = base_config(rho=0.25)
    methods = ["random", "ucb_indep", "tabular", "mf_sgd", "bias_model", "club", "knn_cf", "soft_impute", "swarm_cf"]
    raw = {a: [] for a in methods}
    for s in SEEDS:
        wr = np.random.RandomState(s)
        R = physical_reward(wr, cfg.m, cfg.n, 4)
        P, U = wide_factors(R)
        d_guess = cfg.rank_for_run(wr)
        for a in methods:
            raw[a].append(run_one(cfg, P, U, d_guess, s, a))
        print("[emergent]   seed %d done (%.0fs)" % (s, time.time() - t0), flush=True)
    bake = {a: bootstrap_ci(raw[a]) for a in methods}

    # ---- figures ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    cum = np.cumsum(sv ** 2) / np.sum(sv ** 2)
    ax[0].plot(np.arange(1, len(sv) + 1), sv / sv[0], marker="o", ms=3, color="C0", label="singular value (normalized)")
    ax2 = ax[0].twinx(); ax2.plot(np.arange(1, len(sv) + 1), cum, color="C3", lw=1.5, label="cumulative energy")
    ax2.axhline(0.95, color="gray", ls=":", lw=1); ax2.set_ylabel("cumulative energy", color="C3"); ax2.set_ylim(0, 1.02)
    ax[0].set_xlabel("singular-value index"); ax[0].set_ylabel("singular value / $\\sigma_1$")
    ax[0].set_title("(a) Spectrum of the physical reward (S=4): fast decay", fontsize=10); ax[0].grid(alpha=0.25)
    Ss = sorted(rank_vs_S); ax[1].plot(Ss, [rank_vs_S[s][1] for s in Ss], marker="o", color="C0", label="95% energy")
    ax[1].plot(Ss, [rank_vs_S[s][0] for s in Ss], marker="s", color="C2", ls="--", label="90% energy")
    ax[1].plot(Ss, Ss, color="gray", ls=":", lw=1, label="rank = S (modalities)")
    ax[1].set_xlabel("number of sensing modalities $S$"); ax[1].set_ylabel("measured effective rank")
    ax[1].set_title("(b) Effective rank tracks the modality count (n=%d)" % N, fontsize=10)
    ax[1].grid(alpha=0.25); ax[1].legend(fontsize=8, loc="upper left")
    fig.tight_layout(); os.makedirs(os.path.dirname(PDF_RANK), exist_ok=True)
    plt.savefig(FIG_RANK, dpi=150, bbox_inches="tight"); plt.savefig(PDF_RANK, bbox_inches="tight"); plt.close()

    PRETTY = {"random": "Random", "ucb_indep": "Indep-UCB", "tabular": "Tabular", "mf_sgd": "MF-SGD",
              "bias_model": "BiasModel", "club": "CLUB", "knn_cf": "kNN-CF", "soft_impute": "SoftImpute",
              "swarm_cf": "SwarmCF"}
    COL = {"swarm_cf": "C2"}
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = np.arange(len(methods)); mu = [bake[a][0] for a in methods]
    lo = [bake[a][0] - bake[a][1] for a in methods]; hi = [bake[a][2] - bake[a][0] for a in methods]
    ax.bar(xs, mu, yerr=[lo, hi], capsize=4, color=[COL.get(a, "C0") for a in methods], edgecolor="black")
    ax.set_xticks(xs); ax.set_xticklabels([PRETTY[a] for a in methods], rotation=20, ha="right", fontsize=8)
    ax.axhline(0, color="k", lw=0.6); ax.set_ylabel("unseen-pair skill (held-out)")
    ax.set_title("Bake-off on the EMERGENT-low-rank physical sensing reward (rho=0.25, 16 seeds)", fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); plt.savefig(FIG_BAKE, dpi=150, bbox_inches="tight"); plt.savefig(PDF_BAKE, bbox_inches="tight"); plt.close()

    out = {"meta": {"experiment": "emergent low-rank (physical sensing reward)", "m": M, "n": N, "seeds": SEEDS},
           "rank_vs_S": rank_vs_S, "rank_vs_range": rank_vs_range, "rank_kernel": rank_kernel,
           "scree_top20": [float(v) for v in sv[:20]], "bakeoff_unseen": {a: bake[a] for a in methods}}
    op = os.path.join(ROOT, "results", "pilots", "emergent_lowrank_%s.json" % time.strftime("%Y%m%d_%H%M%S"))
    json.dump(out, open(op, "w"), indent=1)

    print("\n=== PART A: effective rank of the PHYSICAL sensing reward (m=%d, n=%d; min(m,n)=%d) ===" % (M, N, M))
    print("reward stats (S=4): min=%.3f mean=%.3f max=%.3f" % (Rstats.min(), Rstats.mean(), Rstats.max()))
    print("%-3s | 90%% | 95%% | 99%%   effective rank (mean over 16 seeds)" % "S")
    for S in sorted(rank_vs_S):
        print("%-3d | %.1f | %.1f | %.1f" % (S, rank_vs_S[S][0], rank_vs_S[S][1], rank_vs_S[S][2]))
    print("95%% rank vs sensing range (S=4): " + ", ".join("rf=%.2f:%.1f" % (k, v) for k, v in rank_vs_range.items()))
    print("95%% rank, Gaussian coverage kernel: " + ", ".join("l=%.2f:%.1f" % (k, v) for k, v in rank_kernel.items()))
    print("\n=== PART B: bake-off unseen-pair skill on the physical reward (rho=0.25, 16 seeds) ===")
    for a in methods:
        print("  %-12s %.3f [%.3f, %.3f]" % (PRETTY[a], bake[a][0], bake[a][1], bake[a][2]))
    print("\nsaved -> %s, %s, %s  (%.0fs)" % (FIG_RANK, FIG_BAKE, op, time.time() - t0))


if __name__ == "__main__":
    main()
