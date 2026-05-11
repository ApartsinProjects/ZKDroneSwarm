"""
Simulated-data figure generator for ZK-MRTA paper.

ALL DATA IS SYNTHETIC. Numbers are derived from the seed-42 run already in
the paper plus plausible statistical variation. Every output file is
watermarked "SIMULATED DATA" and every caption in the paper HTML must carry
the red-font notice.

Run from project root:
    python generate_simulated_figures.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

rng = np.random.default_rng(seed=7)   # fixed seed for reproducible synthetic data

OUTPUT_DIR = "docs/academic-paper/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Style
# ============================================================
COLORS = {
    'mf':     '#2166ac',   # blue
    'random': '#d6604d',   # red-orange
    'oracle': '#1a9641',   # green
    'band':   '#adc7e8',   # light blue fill
}
HATCH = {'mf': '', 'random': '////', 'oracle': '....'}

plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'DejaVu Serif'],
    'font.size':        8,
    'axes.titlesize':   8,
    'axes.labelsize':   8,
    'xtick.labelsize':  7,
    'ytick.labelsize':  7,
    'legend.fontsize':  7,
    'axes.linewidth':   0.7,
    'lines.linewidth':  1.4,
    'figure.dpi':       150,
    'savefig.dpi':      300,
})

WATERMARK_KW = dict(
    fontsize=9, color='red', alpha=0.55, fontweight='bold',
    ha='center', va='center', rotation=22,
    transform=None,   # set per-figure
)


def add_watermark(fig):
    fig.text(0.5, 0.5,
             "SIMULATED DATA\nREPLACE BEFORE SUBMISSION",
             fontsize=11, color='red', alpha=0.30, fontweight='bold',
             ha='center', va='center', rotation=22,
             transform=fig.transFigure,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.0))


def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {path}")
    plt.close(fig)


# ============================================================
# Figure 1 – Engagement profiles (reconstructed from paper)
# ============================================================
def _simulate_engagement_profile(n_steps, n_targets=27, target_hp=10.0,
                                  focus_fire_strength=0.0, seed=0):
    """
    Generate synthetic step-by-step (total_hp_frac, active_frac) curves.

    focus_fire_strength: 0 = random spray, 1 = perfect sequential elimination.
    """
    rng_local = np.random.default_rng(seed)
    total_hp_init    = n_targets * target_hp
    remaining_hp     = np.full(n_targets, target_hp, dtype=float)
    active           = np.ones(n_targets, dtype=bool)

    hp_curve     = [1.0]
    active_curve = [1.0]

    n_drones = 9
    for step in range(n_steps):
        active_idx = np.where(active)[0]
        if len(active_idx) == 0:
            break
        # How much damage per drone this step
        damage_per_drone = 1.0 + rng_local.normal(0, 0.15)

        # Allocate drones to targets
        if focus_fire_strength > 0.6:
            # Oracle-like: concentrate on lowest-HP active targets
            hps = remaining_hp[active_idx]
            order = np.argsort(hps)
            chosen = active_idx[order[rng_local.integers(0, max(1, int(len(order)*0.3)))]]
        elif focus_fire_strength > 0.2:
            # MF-like: prefer top-affinity targets with some spread
            weights = np.exp(-remaining_hp[active_idx] * 0.05 * focus_fire_strength)
            weights = weights / weights.sum()
            k = min(n_drones, len(active_idx))
            chosen_set = rng_local.choice(active_idx, size=k, replace=True, p=weights)
        else:
            # Random: uniform spray
            chosen_set = rng_local.choice(active_idx, size=n_drones, replace=True)

        if focus_fire_strength > 0.6:
            hits = np.bincount([chosen] * n_drones, minlength=n_targets)
        else:
            hits = np.bincount(chosen_set, minlength=n_targets)

        for t in active_idx:
            remaining_hp[t] = max(0.0, remaining_hp[t] - hits[t] * damage_per_drone)
            if remaining_hp[t] <= 0:
                active[t] = False

        hp_curve.append(remaining_hp.sum() / total_hp_init)
        active_curve.append(active.sum() / n_targets)

    # Pad if needed (shouldn't happen, but safety)
    while len(hp_curve) < n_steps + 1:
        hp_curve.append(0.0)
        active_curve.append(0.0)
    return hp_curve[:n_steps+1], active_curve[:n_steps+1]


def fig1_engagement_profiles():
    print("Generating Figure 1: Engagement profiles ...")
    # Reproduce qualitative shapes described in the paper
    mf_hp,  mf_act  = _simulate_engagement_profile(68,  focus_fire_strength=0.42, seed=1)
    rnd_hp, rnd_act = _simulate_engagement_profile(126, focus_fire_strength=0.02, seed=2)
    orc_hp, orc_act = _simulate_engagement_profile(62,  focus_fire_strength=0.85, seed=3)

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=True,
                             constrained_layout=True)
    panels = [
        (axes[0], mf_hp,  mf_act,  '(a) MF Policy\n(ep. 35, 68 steps)',     True),
        (axes[1], rnd_hp, rnd_act, '(b) Random Baseline\n(126 steps)',       False),
        (axes[2], orc_hp, orc_act, '(c) Oracle Benchmark\n(62 steps)',       False),
    ]
    for ax, hp, act, title, show_y in panels:
        x = range(len(hp))
        ax.plot(x, [v*100 for v in hp],  color=COLORS['mf'],     lw=1.4, label='Total HP')
        ax.plot(x, [v*100 for v in act], color=COLORS['random'], lw=1.4, ls='--', label='Active targets')
        ax.set_title(title, pad=3)
        ax.set_ylim(-2, 105)
        ax.set_xlim(0, len(hp)-1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', ls=':', lw=0.5, color='#cccccc', zorder=0)
        if show_y:
            ax.set_ylabel('% of initial value')
        ax.set_xlabel('Timestep')

    handles = [
        Line2D([0],[0], color=COLORS['mf'],     lw=1.4,         label='Total HP remaining'),
        Line2D([0],[0], color=COLORS['random'], lw=1.4, ls='--', label='Active targets remaining'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False,
               fontsize=7, bbox_to_anchor=(0.5, -0.10))
    add_watermark(fig)
    save(fig, 'fig1-engagement-profiles.png')


# ============================================================
# Figure 3 – Multi-seed learning curves (steps + match quality)
# ============================================================
def fig3_multiseed_learning():
    print("Generating Figure 3: Multi-seed learning curves ...")
    eps = np.arange(1, 36)
    N_SEEDS = 5

    # ---- Steps per episode: phase-structured synthetic trajectory ----
    def steps_trajectory(seed_offset):
        rng_l = np.random.default_rng(seed_offset)
        # Phase 1: rapid drop (ep 1-9)
        ph1 = np.linspace(184, 111, 9) + rng_l.normal(0, 6, 9)
        # Phase 2: plateau with crowding (ep 9-21)
        base2 = np.linspace(111, 76, 13) + rng_l.normal(0, 4, 13)
        # Phase 3: slow refinement (ep 21-35)
        base3 = np.linspace(76, 68, 15) + rng_l.normal(0, 3, 15)
        return np.clip(np.concatenate([ph1, base2[1:], base3[1:]]), 55, 200)

    steps_all = np.stack([steps_trajectory(i*100) for i in range(N_SEEDS)])
    steps_mean = steps_all.mean(0)
    steps_std  = steps_all.std(0)

    # ---- Match quality: monotonic increase with noise ----
    def match_trajectory(seed_offset):
        rng_l = np.random.default_rng(seed_offset)
        ph1 = np.linspace(0.205, 0.37, 9)  + rng_l.normal(0, 0.012, 9)
        ph2 = np.linspace(0.37,  0.53, 13) + rng_l.normal(0, 0.010, 13)
        ph3 = np.linspace(0.53,  0.55, 15) + rng_l.normal(0, 0.008, 15)
        return np.clip(np.concatenate([ph1, ph2[1:], ph3[1:]]), 0.15, 0.70)

    mq_all  = np.stack([match_trajectory(i*100+50) for i in range(N_SEEDS)])
    mq_mean = mq_all.mean(0)
    mq_std  = mq_all.std(0)

    # Oracle / Random reference lines (flat)
    orc_steps = 63.0
    rnd_steps = 129.4
    orc_mq    = 0.650
    rnd_mq    = 0.303

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), constrained_layout=True)

    # Panel (a): Steps
    ax = axes[0]
    ax.fill_between(eps, steps_mean - steps_std, steps_mean + steps_std,
                    alpha=0.25, color=COLORS['mf'], label='_nolegend_')
    ax.plot(eps, steps_mean, color=COLORS['mf'], lw=1.4, label='MF (mean ± 1σ)')
    ax.axhline(orc_steps, color=COLORS['oracle'], lw=1.2, ls='--', label=f'Oracle ({orc_steps:.0f})')
    ax.axhline(rnd_steps, color=COLORS['random'], lw=1.2, ls=':',  label=f'Random ({rnd_steps:.0f})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps to completion')
    ax.set_title('(a) Episode duration', pad=3)
    ax.set_xlim(1, 35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', ls=':', lw=0.5, color='#cccccc', zorder=0)
    ax.legend(frameon=False, fontsize=7)

    # Panel (b): Match quality
    ax = axes[1]
    ax.fill_between(eps, mq_mean - mq_std, mq_mean + mq_std,
                    alpha=0.25, color=COLORS['mf'])
    ax.plot(eps, mq_mean, color=COLORS['mf'], lw=1.4, label='MF (mean ± 1σ)')
    ax.axhline(orc_mq, color=COLORS['oracle'], lw=1.2, ls='--', label=f'Oracle ({orc_mq:.3f})')
    ax.axhline(rnd_mq, color=COLORS['random'], lw=1.2, ls=':',  label=f'Random ({rnd_mq:.3f})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Avg match quality')
    ax.set_title('(b) Match quality', pad=3)
    ax.set_xlim(1, 35)
    ax.set_ylim(0.10, 0.72)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', ls=':', lw=0.5, color='#cccccc', zorder=0)
    ax.legend(frameon=False, fontsize=7)

    add_watermark(fig)
    save(fig, 'fig3-multiseed-learning-curves.png')


# ============================================================
# Figure 4 – Factorization dimension ablation
# ============================================================
def fig4_df_ablation():
    print("Generating Figure 4: d_f ablation ...")
    df_vals = [1, 2, 3, 4, 6]
    # Mean (± std) across 5 seeds
    steps_mean = np.array([91.4, 73.2, 68.8, 70.4, 76.1])
    steps_std  = np.array([ 5.8,  4.1,  3.0,  3.7,  4.9])
    mq_mean    = np.array([0.406, 0.507, 0.543, 0.531, 0.498])
    mq_std     = np.array([0.022, 0.016, 0.013, 0.015, 0.019])

    x = np.arange(len(df_vals))
    bar_colors = ['#9ecae1', '#6baed6', COLORS['mf'], '#4292c6', '#2171b5']

    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.5), constrained_layout=True)

    # Steps
    ax = axes[0]
    bars = ax.bar(x, steps_mean, yerr=steps_std, capsize=3, width=0.55,
                  color=bar_colors, edgecolor='#333', linewidth=0.6, zorder=3)
    # Highlight the correct d_f=3
    bars[2].set_edgecolor('black')
    bars[2].set_linewidth(1.5)
    ax.axhline(63.0, color=COLORS['oracle'], lw=1.0, ls='--', label='Oracle ref.', zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f'$d_f={v}$' for v in df_vals])
    ax.set_ylabel('Steps to completion (mean ± 1σ)')
    ax.set_title('(a) Episode duration vs. $d_f$', pad=3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', ls=':', lw=0.5, color='#cccccc', zorder=0)
    ax.legend(frameon=False, fontsize=7)
    ax.text(2, steps_mean[2] + steps_std[2] + 2.5, 'true $d$',
            ha='center', fontsize=7, color='black', style='italic')

    # Match quality
    ax = axes[1]
    bars2 = ax.bar(x, mq_mean, yerr=mq_std, capsize=3, width=0.55,
                   color=bar_colors, edgecolor='#333', linewidth=0.6, zorder=3)
    bars2[2].set_edgecolor('black')
    bars2[2].set_linewidth(1.5)
    ax.axhline(0.650, color=COLORS['oracle'], lw=1.0, ls='--', label='Oracle ref.', zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f'$d_f={v}$' for v in df_vals])
    ax.set_ylabel('Avg match quality (mean ± 1σ)')
    ax.set_title('(b) Match quality vs. $d_f$', pad=3)
    ax.set_ylim(0.32, 0.72)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', ls=':', lw=0.5, color='#cccccc', zorder=0)
    ax.legend(frameon=False, fontsize=7)
    ax.text(2, mq_mean[2] + mq_std[2] + 0.012, 'true $d$',
            ha='center', fontsize=7, color='black', style='italic')

    add_watermark(fig)
    save(fig, 'fig4-df-ablation.png')


# ============================================================
# Figure 5 – Noise robustness sweep
# ============================================================
def fig5_noise_robustness():
    print("Generating Figure 5: Noise robustness ...")
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5]
    steps_mean   = np.array([62.4, 65.1, 68.8, 74.6, 89.3])
    steps_std    = np.array([ 2.1,  2.5,  3.0,  4.2,  7.8])
    mq_mean      = np.array([0.571, 0.558, 0.543, 0.521, 0.463])
    mq_std       = np.array([0.010, 0.012, 0.013, 0.016, 0.024])

    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.5), constrained_layout=True)

    for ax, means, stds, ylabel, title, oracle_ref, ylim in [
        (axes[0], steps_mean, steps_std,
         'Steps to completion (mean ± 1σ)', '(a) Episode duration',
         63.0, (50, 110)),
        (axes[1], mq_mean,   mq_std,
         'Avg match quality (mean ± 1σ)', '(b) Match quality',
         0.650, (0.38, 0.70)),
    ]:
        ax.fill_between(noise_levels,
                        means - stds, means + stds,
                        alpha=0.25, color=COLORS['mf'])
        ax.plot(noise_levels, means, 'o-',
                color=COLORS['mf'], lw=1.4, ms=4, label='MF policy')
        ax.axhline(oracle_ref, color=COLORS['oracle'], lw=1.0, ls='--',
                   label='Oracle (noiseless ref.)')
        # Highlight baseline config
        ax.axvline(0.2, color='gray', lw=0.8, ls=':', label='Baseline noise (0.2)')
        ax.set_xlabel('Noise level (reward = observation noise)')
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=3)
        ax.set_xlim(-0.03, 0.55)
        ax.set_ylim(*ylim)
        ax.set_xticks(noise_levels)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', ls=':', lw=0.5, color='#cccccc', zorder=0)
        ax.legend(frameon=False, fontsize=7)

    add_watermark(fig)
    save(fig, 'fig5-noise-robustness.png')


# ============================================================
# Run all
# ============================================================
if __name__ == '__main__':
    print("=== ZK-MRTA: Simulated Figure Generation ===\n")
    fig1_engagement_profiles()
    fig3_multiseed_learning()
    fig4_df_ablation()
    fig5_noise_robustness()
    print("\nAll figures saved to", OUTPUT_DIR)
