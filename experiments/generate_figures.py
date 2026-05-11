"""
experiments/generate_figures.py

Generate all paper figures from experimental results.

Figures produced:
  fig1-policy-comparison.png  -- Bar chart: all 5 policies on steps + LMQ
  fig3-multiseed-learning-curves.png -- MF-CF and UCB-Indep LMQ vs episode (5 seeds)
  fig4-df-ablation.png        -- Factorization dimension: steps + LMQ vs d_f
  fig5-noise-robustness.png   -- Observation noise: steps + LMQ vs sigma

Usage:
    python experiments/generate_figures.py [--out-dir docs/academic-paper/figures]
"""

import argparse
import glob
import json
import math
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT_DIR = "docs/academic-paper/figures"

# --- Shared style ---
COLORS = {
    "random":    "#9e9e9e",
    "oracle_l":  "#2196F3",
    "oracle_hp": "#1565C0",
    "ucb_indep": "#FF9800",
    "mf_cf":     "#4CAF50",
}
LABELS = {
    "random":    "Random",
    "oracle_l":  "Oracle-L",
    "oracle_hp": "Oracle-HP",
    "ucb_indep": "UCB-Indep",
    "mf_cf":     "MF-CF",
}
POLICY_ORDER = ["random", "ucb_indep", "mf_cf", "oracle_l", "oracle_hp"]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def mean(v): return sum(v) / len(v) if v else 0.0


def std(v):
    if len(v) < 2: return 0.0
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def se(v): return std(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0


# ============================================================
# Figure 1: Policy comparison bar chart
# ============================================================
def fig1_policy_comparison(out_dir):
    stats = {}
    for p in POLICY_ORDER:
        files = sorted(glob.glob(f"results/all_seeds/{p}_s*.json"))
        eps = []
        for f in files:
            eps.extend(load(f)["episodes"])
        stats[p] = {
            "steps_mean": mean([e["steps"] for e in eps]),
            "steps_se":   se([e["steps"] for e in eps]),
            "lmq_mean":   mean([e["avg_latent_match_quality"] for e in eps]),
            "lmq_se":     se([e["avg_latent_match_quality"] for e in eps]),
        }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.8))
    x = np.arange(len(POLICY_ORDER))
    w = 0.55

    # Steps
    bars = ax1.bar(
        x,
        [stats[p]["steps_mean"] for p in POLICY_ORDER],
        w,
        yerr=[stats[p]["steps_se"] * 1.96 for p in POLICY_ORDER],
        color=[COLORS[p] for p in POLICY_ORDER],
        edgecolor="white",
        linewidth=0.8,
        capsize=4,
        error_kw={"elinewidth": 1.2, "ecolor": "#333333"},
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels([LABELS[p] for p in POLICY_ORDER], rotation=20, ha="right")
    ax1.set_ylabel("Mean steps to completion")
    ax1.set_title("(a) Episode length")
    ax1.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax1.set_axisbelow(True)
    for bar, p in zip(bars, POLICY_ORDER):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1.5,
                 f"{stats[p]['steps_mean']:.1f}",
                 ha="center", va="bottom", fontsize=8.5)

    # LMQ
    bars2 = ax2.bar(
        x,
        [stats[p]["lmq_mean"] for p in POLICY_ORDER],
        w,
        yerr=[stats[p]["lmq_se"] * 1.96 for p in POLICY_ORDER],
        color=[COLORS[p] for p in POLICY_ORDER],
        edgecolor="white",
        linewidth=0.8,
        capsize=4,
        error_kw={"elinewidth": 1.2, "ecolor": "#333333"},
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels([LABELS[p] for p in POLICY_ORDER], rotation=20, ha="right")
    ax2.set_ylabel("Mean latent match quality $\\bar{q}$")
    ax2.set_title("(b) Latent match quality")
    ax2.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax2.set_axisbelow(True)
    ax2.set_ylim(0, 0.95)
    for bar, p in zip(bars2, POLICY_ORDER):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.008,
                 f"{stats[p]['lmq_mean']:.3f}",
                 ha="center", va="bottom", fontsize=8.5)

    fig.suptitle("Policy comparison: 5 policies × 5 seeds × 35 episodes (175 episodes each)",
                 fontsize=10, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig1-policy-comparison.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ============================================================
# Figure 3: Multi-seed learning curves
# ============================================================
def fig3_learning_curves(out_dir):
    from collections import defaultdict

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    for policy_name, ax, title in [
        ("mf_cf",     ax1, "(a) MF-CF"),
        ("ucb_indep", ax2, "(b) UCB-Indep"),
    ]:
        # Per-episode LMQ, per seed
        seed_curves = {}
        for f in sorted(glob.glob(f"results/all_seeds/{policy_name}_s*.json")):
            d = load(f)
            seed = d["seed"]
            seed_curves[seed] = [e["avg_latent_match_quality"] for e in d["episodes"]]

        max_ep = max(len(v) for v in seed_curves.values())
        episodes = np.arange(1, max_ep + 1)

        # Thin seed lines
        all_vals = np.full((len(seed_curves), max_ep), np.nan)
        for i, (seed, curve) in enumerate(sorted(seed_curves.items())):
            n = len(curve)
            all_vals[i, :n] = curve
            ax.plot(episodes[:n], curve,
                    color=COLORS[policy_name], alpha=0.3, linewidth=0.8)

        # Mean +/- 1 sigma envelope
        mu = np.nanmean(all_vals, axis=0)
        sigma = np.nanstd(all_vals, axis=0, ddof=1)
        ax.plot(episodes, mu,
                color=COLORS[policy_name], linewidth=2.2,
                label=f"{LABELS[policy_name]} mean")
        ax.fill_between(episodes, mu - sigma, mu + sigma,
                        color=COLORS[policy_name], alpha=0.18, label="±1σ")

        # Oracle-L reference line
        oracle_files = sorted(glob.glob("results/all_seeds/oracle_l_s*.json"))
        oracle_vals = []
        for of in oracle_files:
            od = load(of)
            oracle_vals.extend([e["avg_latent_match_quality"] for e in od["episodes"]])
        oracle_mean = mean(oracle_vals)
        ax.axhline(oracle_mean, color=COLORS["oracle_l"], linestyle="--",
                   linewidth=1.4, label=f"Oracle-L ({oracle_mean:.3f})", alpha=0.85)

        # Random reference line
        rand_files = sorted(glob.glob("results/all_seeds/random_s*.json"))
        rand_vals = []
        for rf in rand_files:
            rd = load(rf)
            rand_vals.extend([e["avg_latent_match_quality"] for e in rd["episodes"]])
        rand_mean = mean(rand_vals)
        ax.axhline(rand_mean, color=COLORS["random"], linestyle=":",
                   linewidth=1.2, label=f"Random ({rand_mean:.3f})", alpha=0.7)

        ax.set_xlabel("Episode")
        ax.set_ylabel("Avg latent match quality $\\bar{q}$")
        ax.set_title(title)
        ax.set_xlim(1, max_ep)
        ax.set_ylim(0.1, 1.0)
        ax.legend(fontsize=9, loc="lower right")
        ax.yaxis.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
        ax.set_axisbelow(True)

    fig.suptitle("Learning curves across 5 seeds (thin = individual seeds, bold = mean ± 1σ)",
                 fontsize=10)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig3-multiseed-learning-curves.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ============================================================
# Figure 4: Factorization dimension ablation
# ============================================================
def fig4_df_ablation(out_dir):
    dims = [1, 2, 3, 5, 8]
    cond_map = {1: "mf_latent1", 2: "mf_latent2", 3: "mf_latent3",
                5: "mf_latent5", 8: "mf_latent8"}

    steps_mu, steps_se_vals, lmq_mu, lmq_se_vals = [], [], [], []
    for d in dims:
        files = sorted(glob.glob(f"results/sweep/{cond_map[d]}_s*.json"))
        eps = []
        for f in files:
            eps.extend(load(f)["episodes"])
        s = [e["steps"] for e in eps]
        q = [e["avg_latent_match_quality"] for e in eps]
        steps_mu.append(mean(s)); steps_se_vals.append(se(s) * 1.96)
        lmq_mu.append(mean(q)); lmq_se_vals.append(se(q) * 1.96)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.8))
    x = np.array(dims)

    ax1.errorbar(dims, steps_mu, yerr=steps_se_vals,
                 marker="o", color=COLORS["mf_cf"], linewidth=2,
                 markersize=6, capsize=4, elinewidth=1.3)
    ax1.set_xlabel("Factorization dimension $d_f$")
    ax1.set_ylabel("Mean steps to completion")
    ax1.set_title("(a) Episode length vs $d_f$")
    ax1.set_xticks(dims)
    ax1.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax1.set_axisbelow(True)

    ax2.errorbar(dims, lmq_mu, yerr=lmq_se_vals,
                 marker="s", color=COLORS["mf_cf"], linewidth=2,
                 markersize=6, capsize=4, elinewidth=1.3)
    ax2.set_xlabel("Factorization dimension $d_f$")
    ax2.set_ylabel("Mean latent match quality $\\bar{q}$")
    ax2.set_title("(b) Match quality vs $d_f$")
    ax2.set_xticks(dims)
    ax2.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax2.set_axisbelow(True)

    fig.suptitle("MF-CF factorization dimension ablation (5 seeds × 35 episodes)", fontsize=10)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig4-df-ablation.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ============================================================
# Figure 5: Noise robustness
# ============================================================
def fig5_noise_robustness(out_dir):
    noise_levels = [0.0, 0.1, 0.2, 0.5]

    obs_conds = {0.0: "mf_onoise0", 0.1: "mf_onoise01",
                 0.2: "mf_onoise02", 0.5: "mf_onoise05"}
    rew_conds = {0.0: "mf_rnoise0", 0.1: "mf_rnoise01",
                 0.2: "mf_rnoise02", 0.5: "mf_rnoise05"}

    def collect(cond_map):
        s_mu, s_se, q_mu, q_se = [], [], [], []
        for nl in noise_levels:
            files = sorted(glob.glob(f"results/sweep/{cond_map[nl]}_s*.json"))
            eps = []
            for f in files:
                eps.extend(load(f)["episodes"])
            s = [e["steps"] for e in eps]; q = [e["avg_latent_match_quality"] for e in eps]
            s_mu.append(mean(s)); s_se.append(se(s) * 1.96)
            q_mu.append(mean(q)); q_se.append(se(q) * 1.96)
        return s_mu, s_se, q_mu, q_se

    obs_s, obs_s_se, obs_q, obs_q_se = collect(obs_conds)
    rew_s, rew_s_se, rew_q, rew_q_se = collect(rew_conds)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    x = noise_levels

    ax1.errorbar(x, obs_s, yerr=obs_s_se, marker="o", color="#1565C0",
                 linewidth=2, markersize=6, capsize=4, label="Obs. noise $\\sigma_o$")
    ax1.errorbar(x, rew_s, yerr=rew_s_se, marker="s", color="#E91E63",
                 linewidth=2, markersize=6, capsize=4, linestyle="--",
                 label="Reward noise $\\sigma_r$")
    ax1.set_xlabel("Noise level $\\sigma$")
    ax1.set_ylabel("Mean steps to completion")
    ax1.set_title("(a) Episode length vs noise")
    ax1.legend()
    ax1.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax1.set_axisbelow(True)

    ax2.errorbar(x, obs_q, yerr=obs_q_se, marker="o", color="#1565C0",
                 linewidth=2, markersize=6, capsize=4, label="Obs. noise $\\sigma_o$")
    ax2.errorbar(x, rew_q, yerr=rew_q_se, marker="s", color="#E91E63",
                 linewidth=2, markersize=6, capsize=4, linestyle="--",
                 label="Reward noise $\\sigma_r$")
    ax2.set_xlabel("Noise level $\\sigma$")
    ax2.set_ylabel("Mean latent match quality $\\bar{q}$")
    ax2.set_title("(b) Match quality vs noise")
    ax2.legend()
    ax2.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax2.set_axisbelow(True)

    fig.suptitle("MF-CF noise robustness (5 seeds × 35 episodes per condition)", fontsize=10)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig5-noise-robustness.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Generating figures...")
    fig1_policy_comparison(args.out_dir)
    fig3_learning_curves(args.out_dir)
    fig4_df_ablation(args.out_dir)
    fig5_noise_robustness(args.out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
