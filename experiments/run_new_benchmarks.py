"""
experiments/run_new_benchmarks.py

Three new experiment sets that expose when MF-CF genuinely dominates:

  Set A -- Arm-space scaling: n = 27 / 54 / 108 / 216 targets (m=9 fixed)
           Compares random, oracle_hp, oracle_l, ucb_indep, ucb_homo, mf_cf
           n=27 results reuse the existing all_seeds/ data (skipped here).

  Set B -- Environment latent dimension: d = 3 / 5 / 8
           Only ucb_indep and mf_cf (with d_f = env d).
           Tests whether MF-CF's structural advantage grows with d.

  Set C -- UCB-Homogeneous at the baseline (n=27) to anchor
           the pooled-arm failure mode.
           (ucb_homo is also included in Set A at every n level.)

Outputs:
  results/new_benchmarks/<condition>_<policy>_s<seed>.json
  results/logs/nb_<condition>_<policy>_s<seed>.log

Usage:
    python experiments/run_new_benchmarks.py [--dry-run]
    python experiments/run_new_benchmarks.py --sets A B C
    python experiments/run_new_benchmarks.py --sets A      # scaling only
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

PYTHON = sys.executable
RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_experiment_sweep.py")
SEEDS = [42, 123, 456, 789, 1337]
EPISODES = 35

# ---------------------------------------------------------------------------
# Set A: arm-space scaling
# n=27 is the baseline (results/all_seeds/); run n=54 and n=108 here.
# max_steps scales proportionally with n so all policies can complete.
#   n=27 (baseline): max_steps=250
#   n=54  (2x):      max_steps=500
#   n=108 (4x):      max_steps=1000
# ucb_homo is also run at n=27 to anchor the pooled-arm failure mode.
# ---------------------------------------------------------------------------
SCALING_POLICIES = ["random", "oracle_hp", "oracle_l", "ucb_indep", "ucb_homo", "mf_cf"]

# n=27 max_steps scaling: 250 * (n / 27) rounded to nearest 50
def scaled_max_steps(n, base_n=27, base_steps=250):
    import math
    raw = base_steps * n / base_n
    return int(math.ceil(raw / 50) * 50)

SET_A_CONDITIONS = []
for n, ms in [(54, scaled_max_steps(54)), (108, scaled_max_steps(108))]:
    for policy in SCALING_POLICIES:
        SET_A_CONDITIONS.append({
            "name": f"scale_n{n}",
            "policy": policy,
            "num_targets": n,
            "max_steps": ms,
        })
# ucb_homo at baseline n=27 (anchor for failure-mode table)
SET_A_CONDITIONS.append({
    "name": "scale_n27",
    "policy": "ucb_homo",
    "num_targets": 27,
})

# ---------------------------------------------------------------------------
# Set B: environment latent dimension
# MF-CF d_f is auto-matched to env_latent_dim by apply_overrides.
# d=3 baseline already in all_seeds/; run d=5 and d=8.
# ---------------------------------------------------------------------------
SET_B_CONDITIONS = []
for d in [5, 8]:
    for policy in ["ucb_indep", "mf_cf"]:
        SET_B_CONDITIONS.append({
            "name": f"latentd{d}",
            "policy": policy,
            "env_latent_dim": d,
        })

# ---------------------------------------------------------------------------
# Set C: UCBHomogeneous at baseline -- already covered in Set A (scale_n27)
# but also run ucb_homo with explicit "homo_baseline" tag for clarity in
# figures / tables.
# ---------------------------------------------------------------------------
SET_C_CONDITIONS = [
    {"name": "homo_baseline", "policy": "ucb_homo", "num_targets": 27},
]
# homo_baseline is identical to scale_n27/ucb_homo but lives under a
# separate tag so figures can reference it independently.

# ---------------------------------------------------------------------------
# Set D: Probe-Then-Fit warm-start ablation
#
# Compare MF-CF (cold start) vs PTF with probe_episodes K in {5, 8, 12}
# at the baseline (n=27, d=3) AND at the harder regime (n=108, d=3).
#
# Conditions:
#   ptf_k5  / ptf_k8 / ptf_k12 at n=27   (5 seeds each = 15 runs)
#   ptf_k8                      at n=108  (5 seeds      =  5 runs)
#   mf_cf   (reuse all_seeds/)  at n=27   -- reference, no new runs
#   mf_cf                       at n=108  -- reuse scale_n108 results
#   ucb_indep (reuse all_seeds) at n=27   -- reference
#
# Outputs: results/new_benchmarks/ptf_{name}_s{seed}.json
# ---------------------------------------------------------------------------
SET_D_CONDITIONS = []

# Baseline n=27: K sweep
for k in [5, 8, 12]:
    SET_D_CONDITIONS.append({
        "name": f"ptf_n27_k{k}",
        "policy": f"ptf_k{k}",
        "num_targets": 27,
    })

# Large arm-space n=108: best K only (K=8 is a good default)
SET_D_CONDITIONS.append({
    "name": "ptf_n108_k8",
    "policy": "ptf_k8",
    "num_targets": 108,
    "max_steps": scaled_max_steps(108),
})


# ---------------------------------------------------------------------------
# Set E: PTF at d=8, n=108 (the joint hard regime predicted by Theorems 5, 8)
# Theory predicts K* approx d^0.75 * K*(d=3) approx 10 probe episodes at d=8.
# ---------------------------------------------------------------------------
SET_E_CONDITIONS = []
for k in [8, 10, 12]:
    SET_E_CONDITIONS.append({
        "name": f"ptf_n108_d8_k{k}",
        "policy": f"ptf_k{k}",
        "num_targets": 108,
        "env_latent_dim": 8,
        "max_steps": 2500,
    })
for pol in ["ucb_indep", "mf_cf"]:
    SET_E_CONDITIONS.append({
        "name": "scale_d8_n108",
        "policy": pol,
        "num_targets": 108,
        "env_latent_dim": 8,
        "max_steps": 2500,
    })

# ---------------------------------------------------------------------------
# Set G: ZK-compliant Independent Q-Learning (IQL-ZK) baseline
#
# A natural ZK-MARL baseline adapted from Tampuu et al. 2017's independent
# DQN: each drone maintains a private tabular Q-table updated only from its
# own observed rewards via TD updates with epsilon-greedy exploration. Tests
# whether per-agent Q-learning provides a meaningfully different point on the
# regret-flexibility trade-off compared to UCB-Indep.
# ---------------------------------------------------------------------------
SET_G_CONDITIONS = [
    {"name": "iql_baseline", "policy": "iql_zk", "num_targets": 27},
]
# Also test at the larger scales for scaling-ratio comparison
for n, ms in [(54, scaled_max_steps(54)), (108, scaled_max_steps(108))]:
    SET_G_CONDITIONS.append({
        "name": f"scale_n{n}",
        "policy": "iql_zk",
        "num_targets": n,
        "max_steps": ms,
    })


# ---------------------------------------------------------------------------
# Set H: 10+ seed robustness check (revision item 2). Extra seeds for the
# baseline configuration to enable Wilcoxon/Hodges-Lehmann at n=10.
# ---------------------------------------------------------------------------
EXTRA_SEEDS = [2024, 2025, 31337, 99999, 1729]
SET_H_CONDITIONS = []
for policy in ["random", "ucb_indep", "mf_cf", "iql_zk", "ptf_k5", "oracle_l", "oracle_hp"]:
    SET_H_CONDITIONS.append({
        "name": "robustness10",
        "policy": policy,
        "num_targets": 27,
    })

# ---------------------------------------------------------------------------
# Set I: IQL-ZK hyperparameter coordinate search (revision item 3) on
# held-out seed 314, mirroring the MF-CF protocol of section 6.3.
# ---------------------------------------------------------------------------
SET_I_CONDITIONS = []
# We sweep alpha (TD learning rate) and epsilon decay
# Note: the policy uses fixed hyperparameters inside the IQLZKPolicy class.
# To make the sweep externally configurable we expose alpha and
# epsilon_decay via overrides keys recognised by apply_overrides
# (see run_experiment_sweep.py extensions for iql_alpha, iql_decay).
for alpha in [0.05, 0.10, 0.20, 0.50]:
    for decay in [0.999, 0.9995, 0.9998]:
        SET_I_CONDITIONS.append({
            "name": f"iql_sweep_a{int(alpha*100):02d}_d{int(decay*10000)}",
            "policy": "iql_zk",
            "num_targets": 27,
            "iql_alpha": alpha,
            "iql_epsilon_decay": decay,
        })

# ---------------------------------------------------------------------------
# Set J: Non-orthogonal latent geometry (revision item 6) at n=54, d=3.
# Default benchmark uses center_mode="one_hot" (axis-aligned). This set tests
# "random" and "orthogonal" mode-center placements to verify the methods
# generalise beyond the axis-aligned regime.
# ---------------------------------------------------------------------------
SET_J_CONDITIONS = []
for center_mode in ["random", "orthogonal"]:
    for policy in ["ucb_indep", "iql_zk", "mf_cf", "ptf_k5"]:
        SET_J_CONDITIONS.append({
            "name": f"geom_{center_mode}",
            "policy": policy,
            "num_targets": 54,
            "max_steps": scaled_max_steps(54),
            "center_mode": center_mode,
        })


# ---------------------------------------------------------------------------
# Set K: Optional improvement comparisons (centralised ESTR + Thompson-MF)
# ---------------------------------------------------------------------------
SET_K_CONDITIONS = []
# ESTR baseline at n=27 and n=108
SET_K_CONDITIONS.append({
    "name": "compare_n27", "policy": "estr", "num_targets": 27,
})
SET_K_CONDITIONS.append({
    "name": "compare_n108", "policy": "estr",
    "num_targets": 108, "max_steps": scaled_max_steps(108),
})
# Thompson-sampling MF variant at n=27 and n=108
SET_K_CONDITIONS.append({
    "name": "compare_n27", "policy": "ts_mf", "num_targets": 27,
})
SET_K_CONDITIONS.append({
    "name": "compare_n108", "policy": "ts_mf",
    "num_targets": 108, "max_steps": scaled_max_steps(108),
})

# ---------------------------------------------------------------------------
# Set L: No-broadcast MF-CF long-horizon (70 episodes) for s_eff and T6 viz
# Note: this dataset is NOT used as paper claim; it produces a visualisation
# of the no-broadcast cold-start curve at the predicted T6 horizon.
# ---------------------------------------------------------------------------
SET_L_CONDITIONS = [
    {"name": "long70_bcast",   "policy": "mf_cf",    "num_targets": 27},
    {"name": "long70_nbcast",  "policy": "mf_cf_nb", "num_targets": 27},
]


ALL_SETS = {
    "A": SET_A_CONDITIONS,
    "B": SET_B_CONDITIONS,
    "C": SET_C_CONDITIONS,
    "D": SET_D_CONDITIONS,
    "E": SET_E_CONDITIONS,
    "G": SET_G_CONDITIONS,
    "H": SET_H_CONDITIONS,
    "I": SET_I_CONDITIONS,
    "J": SET_J_CONDITIONS,
    "K": SET_K_CONDITIONS,
    "L": SET_L_CONDITIONS,
}


def build_jobs(conditions, seeds, out_dir, episodes):
    jobs = []
    for cond in conditions:
        # Allow per-condition seed override (used by Sets H and I)
        seed_list = cond.pop("__seeds", None) if isinstance(cond, dict) else None
        these_seeds = seed_list if seed_list is not None else seeds
        for seed in these_seeds:
            out = os.path.join(out_dir, f"{cond['name']}_{cond['policy']}_s{seed}.json")
            if os.path.exists(out):
                print(f"SKIP (exists): {out}")
                continue
            jobs.append((cond, seed, out))
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sets", nargs="+", default=["A", "B", "C"],
                        choices=["A", "B", "C", "D", "E", "G", "H", "I", "J", "K", "L"], metavar="SET",
                        help="Which experiment sets to run (default: A B C; K = ESTR + TS-MF; L = no-bcast long horizon)")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--out-dir", default="results/new_benchmarks")
    parser.add_argument("--log-dir", default="results/logs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Collect all conditions from selected sets.
    # Set H uses EXTRA_SEEDS (separate 10-seed robustness run); we tag those
    # conditions so build_jobs uses the extended seed list for them.
    # Set I uses a single held-out seed 314 (mirroring the MF-CF protocol of §6.3).
    conditions = []
    for s in args.sets:
        for cond in ALL_SETS[s]:
            tagged = dict(cond)
            if s == "H":
                tagged["__seeds"] = EXTRA_SEEDS
            elif s == "I":
                tagged["__seeds"] = [314]
            conditions.append(tagged)

    jobs = build_jobs(conditions, args.seeds, args.out_dir, args.episodes)

    if args.dry_run:
        print(f"Plan: {len(jobs)} job(s), {args.episodes} ep each")
        for cond, seed, out in jobs:
            overrides = {k: v for k, v in cond.items() if k not in ("name", "policy")}
            print(f"  {cond['name']:20s}  {cond['policy']:12s}  seed={seed}"
                  f"  overrides={overrides}")
        return 0

    print(f"[{datetime.now():%H:%M:%S}] Starting {len(jobs)} job(s) "
          f"(sets {', '.join(args.sets)}), {args.episodes} episodes each...")

    failed = []
    for i, (cond, seed, out) in enumerate(jobs, 1):
        log_path = os.path.join(args.log_dir,
                                f"nb_{cond['name']}_{cond['policy']}_s{seed}.log")
        overrides = {k: v for k, v in cond.items() if k not in ("name", "policy")}
        cmd = [PYTHON, RUNNER,
               "--policy", cond["policy"],
               "--seed", str(seed),
               "--episodes", str(args.episodes),
               "--out", out,
               "--override-json", json.dumps(overrides)]

        print(f"[{datetime.now():%H:%M:%S}] [{i:3d}/{len(jobs)}] "
              f"{cond['name']:20s} {cond['policy']:12s} seed={seed}",
              end="", flush=True)

        with open(log_path, "w", encoding="utf-8") as log_fh:
            log_fh.write(f"CMD: {' '.join(cmd)}\n")
            log_fh.write(f"CONDITION: {json.dumps(cond)}\n")
            log_fh.write(f"START: {datetime.now()}\n\n")
            ret = subprocess.call(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
            log_fh.write(f"\nEND: {datetime.now()}  exit={ret}\n")

        if ret == 0:
            try:
                with open(out, encoding="utf-8") as f:
                    d = json.load(f)
                s = d["summary"]
                print(f"  steps={s['avg_steps']:6.1f}  success={s['success_rate']:5.1f}%")
            except Exception:
                print("  done")
        else:
            print(f"  ERROR (exit {ret})  log={log_path}")
            failed.append((cond["name"], cond["policy"], seed))

    print(f"\n[{datetime.now():%H:%M:%S}] DONE: "
          f"{len(jobs) - len(failed)} ok, {len(failed)} failed")
    for name, policy, seed in failed:
        print(f"  FAILED: {name} {policy} seed={seed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
