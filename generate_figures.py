"""
Publication-quality figure generation for ZK-MRTA paper.

Generates Figure 1 (3-panel engagement profiles) as a high-resolution
PNG/PDF suitable for Elsevier RAS submission (300 dpi, white background,
serif font, proper panel labels).

Run from project root:
    python generate_figures.py
"""

import sys, os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- project imports (use exact module paths from main_zk_mrta.py) ---------
from tabula_drone.config import load_config
from tabula_drone.envs.drone_engage_latent_mrta import DroneEngageLatentMRTA
from tabula_drone.policies.base import bind_diagnostics_provider
from tabula_drone.scenarios.latent_scenario_builder import LatentScenarioBuilder
from main_zk_mrta import get_env_diagnostics, create_policy

CONFIG_PATH = "config/scenario.json"
OUTPUT_DIR  = "docs/academic-paper/figures"

# ============================================================
# Publication style
# ============================================================
# Elsevier double-column = 190 mm wide = ~7.48 in
# Elsevier single-column = 90 mm wide = ~3.54 in
# We use double-column for a 3-panel row
FIG_W  = 7.0          # inches
FIG_H  = 2.5          # inches
DPI    = 300
FS     = 8            # base font size (pt)
TS     = 7            # tick label size
LS     = 7            # legend size
LW     = 1.4          # line width
C_HP   = '#2166ac'    # Total HP  (blue)
C_ACT  = '#d6604d'    # Active Targets (red-orange)

plt.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'DejaVu Serif'],
    'font.size':         FS,
    'axes.titlesize':    FS,
    'axes.labelsize':    FS,
    'xtick.labelsize':   TS,
    'ytick.labelsize':   TS,
    'legend.fontsize':   LS,
    'axes.linewidth':    0.7,
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'lines.linewidth':   LW,
    'figure.dpi':        DPI,
    'savefig.dpi':       DPI,
})

# ============================================================
# Simulation helpers
# ============================================================

def build_env(config, drones_cfg, targets_cfg, reward_noise, obs_noise):
    return DroneEngageLatentMRTA(
        world_size=config.world.size,
        max_steps=config.environment.max_steps,
        drones_config=drones_cfg,
        targets_config=targets_cfg,
        scenario_id="fig_gen",
        reward_noise=reward_noise,
        observation_noise=obs_noise,
        builder=None,
        latent_world=None,
        target_hp=config.targets.target_hp,
    )


def run_episode_capture(env, policy, seed):
    """Run one episode; return (hp_fracs, active_fracs, n_steps).

    hp_fracs[0] == active_fracs[0] == 1.0 (initial state before any step).
    """
    obs, infos = env.reset(seed=seed)
    bind_diagnostics_provider(policy, lambda: get_env_diagnostics(env))
    info = get_env_diagnostics(env)

    init_hp     = sum(info['target_hps'])
    init_active = sum(info['target_active'])

    hp_series     = [1.0]
    active_series = [1.0]

    done = False
    steps = 0
    while not done:
        steps += 1
        try:
            actions = policy.select_actions(obs, infos, env=env)
        except TypeError:
            actions = policy.select_actions(obs, infos)
        obs, rewards, terms, truncs, infos = env.step(actions)
        policy.update(obs)
        info = get_env_diagnostics(env)
        hp_series.append(sum(info['target_hps'])     / init_hp)
        active_series.append(sum(info['target_active']) / init_active)
        done = terms[env.agents[0]] or truncs[env.agents[0]]

    return hp_series, active_series, steps


def run_mf_train_then_capture(config, drones_cfg, targets_cfg,
                               reward_noise, obs_noise, num_episodes):
    """Train MF for (num_episodes-1) warmup episodes then capture episode N."""
    env    = build_env(config, drones_cfg, targets_cfg, reward_noise, obs_noise)
    policy = create_policy('matrix_factorization_cf', config, drones_cfg, len(targets_cfg))
    bind_diagnostics_provider(policy, lambda: get_env_diagnostics(env))

    seed = config.seed
    for ep in range(1, num_episodes):        # warmup: episodes 1 .. N-1
        ep_seed = seed + ep - 1
        obs, infos = env.reset(seed=ep_seed)
        bind_diagnostics_provider(policy, lambda: get_env_diagnostics(env))
        done = False
        while not done:
            try:
                actions = policy.select_actions(obs, infos, env=env)
            except TypeError:
                actions = policy.select_actions(obs, infos)
            obs, rewards, terms, truncs, infos = env.step(actions)
            policy.update(obs)
            done = terms[env.agents[0]] or truncs[env.agents[0]]
        policy.soft_reset()
        if ep % 5 == 0 or ep == num_episodes - 1:
            print(f"    warmup ep {ep}/{num_episodes-1} done")

    # Capture final episode
    capture_seed = seed + num_episodes - 1
    hp_s, act_s, n_steps = run_episode_capture(env, policy, seed=capture_seed)
    return hp_s, act_s, n_steps


# ============================================================
# Figure drawing
# ============================================================

def plot_panel(ax, hp_series, active_series, title, show_ylabel, show_xlabel):
    x = list(range(len(hp_series)))
    ax.plot(x, [v * 100 for v in hp_series],
            color=C_HP,  lw=LW,                label='Total HP remaining')
    ax.plot(x, [v * 100 for v in active_series],
            color=C_ACT, lw=LW, linestyle='--', label='Active targets remaining')
    ax.set_ylim(-2, 105)
    ax.set_xlim(0, max(x))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.set_title(title, pad=4, fontsize=FS)
    if show_ylabel:
        ax.set_ylabel('% of initial value', fontsize=FS)
    if show_xlabel:
        ax.set_xlabel('Timestep', fontsize=FS)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', direction='in', length=3, width=0.7)
    ax.grid(axis='y', linestyle=':', linewidth=0.5, color='#cccccc', zorder=0)


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    config = load_config(CONFIG_PATH)

    cf_cfg = config.collaborative_filtering
    mf_cfg = cf_cfg.matrix_factorization_cf if cf_cfg else None
    reward_noise = mf_cfg.reward_noise if (mf_cfg and mf_cfg.reward_noise is not None) else 0.0
    obs_noise    = cf_cfg.observation_noise if cf_cfg else 0.0

    builder = LatentScenarioBuilder(
        world_size=config.world.size,
        config=config.latent_world,
        target_hp=config.targets.target_hp,
        seed=config.seed,
    )
    builder.with_drones(
        count=config.drones.count,
        region=config.drones.region,
        min_distance_between_drones=config.drones.min_distance_between_drones,
    )
    builder.with_targets(
        count=config.targets.count,
        region=config.targets.region,
        min_distance_from_drones=config.targets.min_distance_from_drones,
        min_distance_between_targets=config.targets.min_distance_between_targets,
    )
    drones_cfg, targets_cfg = builder.build()
    num_eps = config.environment.episodic.num_episodes   # 35

    print("=== Generating Figure 1: Engagement Profiles ===")

    # ---- MF: train 34 warmup episodes, capture ep 35 --------------------
    print(f"\n[1/3] Matrix-factorization policy ({num_eps} episodes) ...")
    mf_hp, mf_act, mf_steps = run_mf_train_then_capture(
        config, drones_cfg, targets_cfg, reward_noise, obs_noise, num_eps
    )
    print(f"  MF capture: {mf_steps} steps")

    # ---- Random: one episode (deterministic seeding) ---------------------
    print("\n[2/3] Random policy (single episode) ...")
    rnd_env    = build_env(config, drones_cfg, targets_cfg, reward_noise, obs_noise)
    rnd_policy = create_policy('random', config, drones_cfg, len(targets_cfg))
    rnd_hp, rnd_act, rnd_steps = run_episode_capture(rnd_env, rnd_policy, seed=config.seed)
    print(f"  Random capture: {rnd_steps} steps")

    # ---- Oracle: one episode (deterministic) -----------------------------
    print("\n[3/3] Oracle policy (single episode) ...")
    orc_env    = build_env(config, drones_cfg, targets_cfg, reward_noise, obs_noise)
    orc_policy = create_policy('max_damage_oracle', config, drones_cfg, len(targets_cfg))
    orc_hp, orc_act, orc_steps = run_episode_capture(orc_env, orc_policy, seed=config.seed)
    print(f"  Oracle capture: {orc_steps} steps")

    # ---- Plot ------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_H),
                             sharey=True, constrained_layout=True)

    panels = [
        (axes[0], mf_hp,  mf_act,  f'(a) MF Policy\n(ep. {num_eps}, {mf_steps} steps)',    True,  False),
        (axes[1], rnd_hp, rnd_act, f'(b) Random Baseline\n({rnd_steps} steps)',             False, True),
        (axes[2], orc_hp, orc_act, f'(c) Oracle Benchmark\n({orc_steps} steps)',            False, False),
    ]
    for ax, hp_s, act_s, title, show_y, show_x in panels:
        plot_panel(ax, hp_s, act_s, title, show_ylabel=show_y, show_xlabel=show_x)

    # Shared legend below all panels
    legend_handles = [
        Line2D([0], [0], color=C_HP,  lw=LW,                 label='Total HP remaining'),
        Line2D([0], [0], color=C_ACT, lw=LW, linestyle='--', label='Active targets remaining'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=2,
               frameon=False, fontsize=LS,
               bbox_to_anchor=(0.5, -0.10))

    # Save both PDF (for submission) and PNG (for HTML)
    pdf_path = os.path.join(OUTPUT_DIR, 'fig1-engagement-profiles.pdf')
    png_path = os.path.join(OUTPUT_DIR, 'fig1-engagement-profiles.png')
    fig.savefig(pdf_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    fig.savefig(png_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"\nSaved:\n  {pdf_path}\n  {png_path}")
    print(f"\nFigure summary: MF={mf_steps} steps, Random={rnd_steps} steps, Oracle={orc_steps} steps")
    print("Done.")


if __name__ == '__main__':
    main()
