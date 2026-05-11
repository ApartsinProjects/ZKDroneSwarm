"""
experiments/generate_new_figures.py

Generate figures and summary statistics from new_benchmarks results.

Figures produced:
  fig6-lmq-learning-curves.png   -- LMQ over episodes (n=54): UCBHomo vs others
  fig7-arm-scaling.png           -- avg_steps vs n (27/54/108), full + late-ep
  fig8-latent-d-scaling.png      -- avg_steps vs d (3/5/8), ucb_indep vs mf_cf

Usage:
    python experiments/generate_new_figures.py [--out-dir docs/academic-paper/figures]
"""

import argparse
import json
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SEEDS = [42, 123, 456, 789, 1337]
N_EPS = 35
NB_DIR = "results/new_benchmarks"
ALL_SEEDS_DIR = "results/all_seeds"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_all_seeds(policy, seed, n=27, base_dir=None):
    if base_dir is None:
        base_dir = ALL_SEEDS_DIR if n == 27 else NB_DIR
    if n == 27:
        path = os.path.join(base_dir, f"{policy}_s{seed}.json")
    else:
        path = os.path.join(NB_DIR, f"scale_n{n}_{policy}_s{seed}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_latentd(policy, d, seed):
    path = os.path.join(NB_DIR, f"latentd{d}_{policy}_s{seed}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_homo_baseline(seed):
    # Try both: Set A scale_n27 tag and Set C homo_baseline tag
    for prefix in ["scale_n27", "homo_baseline"]:
        path = os.path.join(NB_DIR, f"{prefix}_ucb_homo_s{seed}.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return None


def gather_across_seeds(loader_fn):
    """Call loader_fn(seed) for each seed; return list of non-None results."""
    return [d for d in (loader_fn(s) for s in SEEDS) if d is not None]


def avg_steps_from_runs(runs, late_n=None):
    """Given list of run dicts, return (mean, std) of avg_steps across seeds.
    If late_n, use last late_n episodes only.
    """
    per_seed = []
    for d in runs:
        if late_n is None:
            per_seed.append(d["summary"]["avg_steps"])
        else:
            eps = d["episodes"][-late_n:]
            per_seed.append(statistics.mean(e["steps"] for e in eps))
    if not per_seed:
        return None, None
    mean = statistics.mean(per_seed)
    std = statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0
    return mean, std


def avg_lmq_from_runs(runs, late_n=None):
    per_seed = []
    for d in runs:
        eps = d["episodes"][-late_n:] if late_n else d["episodes"]
        per_seed.append(statistics.mean(e["avg_latent_match_quality"] for e in eps))
    if not per_seed:
        return None, None
    return statistics.mean(per_seed), (statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0)


def ep_lmq_trajectories(runs):
    """Return list-of-lists: ep_lmq[ep_idx] = list of LMQ values across seeds."""
    ep_lmq = [[] for _ in range(N_EPS)]
    for d in runs:
        for i, ep in enumerate(d["episodes"][:N_EPS]):
            ep_lmq[i].append(ep["avg_latent_match_quality"])
    return ep_lmq


# ---------------------------------------------------------------------------
# Figure 6: LMQ learning curves at n=54
# ---------------------------------------------------------------------------

def fig6_lmq_curves(out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: LMQ over episodes for n=54
    ax = axes[0]
    policy_specs = [
        ("random",    "Random",            "#9E9E9E", "--",  1.5),
        ("ucb_homo",  "UCB-Homo (pooled)", "#E53935", "-.",  2.0),
        ("ucb_indep", "UCB-Indep",         "#1E88E5", "-",   2.0),
        ("mf_cf",     "MF-CF",             "#43A047", "-",   2.5),
    ]

    for policy, label, color, ls, lw in policy_specs:
        runs = gather_across_seeds(lambda s, p=policy: load_all_seeds(p, s, n=54))
        if not runs:
            continue
        traj = ep_lmq_trajectories(runs)
        means = [statistics.mean(v) if v else None for v in traj]
        stds  = [statistics.stdev(v) if len(v) > 1 else 0 for v in traj]
        xs = list(range(1, N_EPS + 1))
        valid = [(xs[i], means[i], stds[i]) for i in range(N_EPS) if means[i] is not None]
        if not valid:
            continue
        vx, vy, ve = zip(*valid)
        ax.plot(vx, vy, color=color, linestyle=ls, linewidth=lw, label=label, zorder=3)
        ax.fill_between(vx,
                        [y - s for y, s in zip(vy, ve)],
                        [y + s for y, s in zip(vy, ve)],
                        color=color, alpha=0.12, zorder=2)

    # Annotate convergence region
    ax.axvspan(26, 35, alpha=0.06, color="green")
    ax.text(30.5, 0.23, "Convergence\nwindow", ha="center", fontsize=8, color="#2E7D32")

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Avg Latent Match Quality (LMQ)", fontsize=12)
    ax.set_title("LMQ Learning Curves (n=54 targets)", fontsize=13, fontweight="bold")
    ax.set_xlim(1, N_EPS)
    ax.set_ylim(0.20, 0.95)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Right: Early vs late steps (n=54)
    ax2 = axes[1]
    pol_keys   = ["ucb_indep", "ucb_homo", "mf_cf"]
    pol_labels = ["UCB-Indep", "UCB-Homo", "MF-CF"]
    bar_colors = ["#1E88E5", "#E53935", "#43A047"]
    x = np.arange(len(pol_labels))
    width = 0.35

    for pi, (phase_name, late_n, alpha_val) in enumerate([("Early (ep 1-10)", None, 0.45),
                                                           ("Late (ep 26-35)", 10, 0.92)]):
        vals, errs = [], []
        for pk in pol_keys:
            runs = gather_across_seeds(lambda s, p=pk: load_all_seeds(p, s, n=54))
            m, e = avg_steps_from_runs(runs, late_n=late_n)
            if late_n is None:
                # early = first 10 episodes
                per_seed = []
                for d in runs:
                    per_seed.append(statistics.mean(ep["steps"] for ep in d["episodes"][:10]))
                m = statistics.mean(per_seed) if per_seed else 0
                e = statistics.stdev(per_seed) if len(per_seed) > 1 else 0
            vals.append(m if m is not None else 0)
            errs.append(e if e is not None else 0)

        offset = (pi - 0.5) * width
        ax2.bar(x + offset, vals, width, yerr=errs, capsize=4,
                label=phase_name,
                color=bar_colors,
                alpha=alpha_val,
                edgecolor="black", linewidth=0.5)

    ax2.set_xlabel("Policy", fontsize=12)
    ax2.set_ylabel("Avg Steps (lower is better)", fontsize=12)
    ax2.set_title("Early vs Late Episode Performance\n(n=54 targets)", fontsize=13, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(pol_labels, fontsize=11)
    ax2.legend(["Early (ep 1-10)", "Late (ep 26-35)"], fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(out_dir, "fig6-lmq-learning-curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Figure 7: Arm-space scaling (steps vs n)
# ---------------------------------------------------------------------------

def fig7_scaling(out_dir):
    policies_full = ["random", "oracle_hp", "oracle_l", "ucb_indep", "ucb_homo", "mf_cf"]
    policy_specs_full = {
        "random":    ("Random",            "#9E9E9E", "--",  1.5),
        "oracle_hp": ("Oracle-HP",         "#FF8F00", ":",   1.8),
        "oracle_l":  ("Oracle-L",          "#AD1457", ":",   1.8),
        "ucb_indep": ("UCB-Indep",         "#1E88E5", "-",   2.0),
        "ucb_homo":  ("UCB-Homo (pooled)", "#E53935", "-.",  2.0),
        "mf_cf":     ("MF-CF",             "#43A047", "-",   2.5),
    }

    # UCBHomo only at n=27 (from homo_baseline) and n=54, n=108
    # For n=27 UCBHomo we use the homo_baseline/scale_n27 runs
    n_vals = [27, 54, 108]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: all-episode avg
    ax = axes[0]
    # Right: late-episode avg
    ax2 = axes[1]

    for ax_idx, (ax_cur, late_n, title_suffix) in enumerate(
            [(ax, None, "Full Average"), (ax2, 10, "Late-Episode (ep 26-35)")]):

        for policy in policies_full:
            label, color, ls, lw = policy_specs_full[policy]

            means, stds, xs = [], [], []
            for n in n_vals:
                if policy == "ucb_homo" and n == 27:
                    runs = gather_across_seeds(load_homo_baseline)
                else:
                    runs = gather_across_seeds(lambda s, p=policy, nv=n: load_all_seeds(p, s, n=nv))
                if not runs:
                    continue
                m, e = avg_steps_from_runs(runs, late_n=late_n)
                if m is None:
                    continue
                means.append(m)
                stds.append(e)
                xs.append(n)

            if not xs:
                continue

            ax_cur.errorbar(xs, means, yerr=stds, color=color, linestyle=ls,
                            linewidth=lw, marker="o", markersize=5, capsize=4,
                            label=label, zorder=3)

        ax_cur.set_xlabel("Number of Targets (n)", fontsize=12)
        ax_cur.set_ylabel("Avg Steps to Completion", fontsize=12)
        ax_cur.set_title(f"Arm-Space Scaling\n({title_suffix})", fontsize=13, fontweight="bold")
        ax_cur.set_xticks(n_vals)
        ax_cur.legend(fontsize=9, loc="upper left")
        ax_cur.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "fig7-arm-scaling.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Figure 8: Latent-d scaling
# ---------------------------------------------------------------------------

def fig8_latentd(out_dir):
    d_vals = [3, 5, 8]
    policy_specs = {
        "ucb_indep": ("UCB-Indep", "#1E88E5", "-",  2.0),
        "mf_cf":     ("MF-CF",    "#43A047", "-",   2.5),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax_idx, (ax_cur, late_n, title_suffix) in enumerate(
            [(axes[0], None, "Full Average"), (axes[1], 10, "Late-Episode (ep 26-35)")]):

        for policy, (label, color, ls, lw) in policy_specs.items():
            means, stds, xs = [], [], []
            for d in d_vals:
                if d == 3:
                    runs = gather_across_seeds(lambda s, p=policy: load_all_seeds(p, s, n=27))
                else:
                    runs = gather_across_seeds(lambda s, p=policy, dv=d: load_latentd(p, dv, s))
                if not runs:
                    continue
                m, e = avg_steps_from_runs(runs, late_n=late_n)
                if m is None:
                    continue
                means.append(m)
                stds.append(e)
                xs.append(d)

            if not xs:
                continue
            ax_cur.errorbar(xs, means, yerr=stds, color=color, linestyle=ls,
                            linewidth=lw, marker="o", markersize=7, capsize=4,
                            label=label, zorder=3)

        ax_cur.set_xlabel("Environment Latent Dim (d)", fontsize=12)
        ax_cur.set_ylabel("Avg Steps to Completion", fontsize=12)
        ax_cur.set_title(f"Latent Dimension Scaling\n({title_suffix})", fontsize=13, fontweight="bold")
        ax_cur.set_xticks(d_vals)
        ax_cur.legend(fontsize=11)
        ax_cur.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "fig8-latent-d-scaling.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Summary statistics printer
# ---------------------------------------------------------------------------

def print_summary_table():
    seeds = SEEDS
    policies_ordered = ["random", "oracle_hp", "oracle_l", "ucb_indep", "ucb_homo", "mf_cf"]
    n_configs = [27, 54, 108]

    print("\n" + "="*80)
    print("SCALING SUMMARY TABLE (avg_steps, mean ± std across 5 seeds)")
    print("="*80)
    header = f"{'Policy':12s}"
    for n in n_configs:
        header += f"  {'n='+str(n):>18s}"
    print(header)
    print("-" * 70)

    for policy in policies_ordered:
        row = f"{policy:12s}"
        for n in n_configs:
            if policy == "ucb_homo" and n == 27:
                runs = gather_across_seeds(load_homo_baseline)
            else:
                runs = gather_across_seeds(lambda s, p=policy, nv=n: load_all_seeds(p, s, n=nv))
            m, e = avg_steps_from_runs(runs)
            if m is not None:
                row += f"  {m:7.1f} ± {e:5.1f}"
            else:
                row += f"  {'—':>18s}"
        print(row)

    print("\n" + "="*80)
    print("LATE-EPISODE SUMMARY (avg_steps over last 10 episodes)")
    print("="*80)
    print(header)
    print("-" * 70)

    for policy in policies_ordered:
        row = f"{policy:12s}"
        for n in n_configs:
            if policy == "ucb_homo" and n == 27:
                runs = gather_across_seeds(load_homo_baseline)
            else:
                runs = gather_across_seeds(lambda s, p=policy, nv=n: load_all_seeds(p, s, n=nv))
            m, e = avg_steps_from_runs(runs, late_n=10)
            if m is not None:
                row += f"  {m:7.1f} ± {e:5.1f}"
            else:
                row += f"  {'—':>18s}"
        print(row)

    print("\n" + "="*80)
    print("LATENT-D SUMMARY (avg_steps, d=3 from all_seeds baseline)")
    print("="*80)
    print(f"{'Policy':12s}  {'d=3':>16s}  {'d=5':>16s}  {'d=8':>16s}")
    print("-" * 65)
    for policy in ["ucb_indep", "mf_cf"]:
        row = f"{policy:12s}"
        runs3 = gather_across_seeds(lambda s, p=policy: load_all_seeds(p, s, n=27))
        m3, e3 = avg_steps_from_runs(runs3)
        row += f"  {m3:7.1f} ± {e3:5.1f}" if m3 else f"  {'—':>16s}"
        for d in [5, 8]:
            runs_d = gather_across_seeds(lambda s, p=policy, dv=d: load_latentd(p, dv, s))
            m, e = avg_steps_from_runs(runs_d)
            row += f"  {m:7.1f} ± {e:5.1f}" if m else f"  {'—':>16s}"
        print(row)

    print("\n" + "="*80)
    print("UCBHomo vs Random LMQ (confirms pooling failure)")
    print("="*80)
    for n in [27, 54, 108]:
        for policy in ["random", "ucb_homo"]:
            if policy == "ucb_homo" and n == 27:
                runs = gather_across_seeds(load_homo_baseline)
            else:
                runs = gather_across_seeds(lambda s, p=policy, nv=n: load_all_seeds(p, s, n=nv))
            if not runs:
                continue
            m, e = avg_lmq_from_runs(runs)
            print(f"  n={n} {policy:12s}: LMQ = {m:.4f} ± {e:.4f}" if m else f"  n={n} {policy}: —")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="docs/academic-paper/figures")
    parser.add_argument("--stats-only", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print_summary_table()

    if not args.stats_only:
        fig6_lmq_curves(args.out_dir)
        fig7_scaling(args.out_dir)
        fig8_latentd(args.out_dir)
        print("\nAll figures saved.")


if __name__ == "__main__":
    main()
