"""
experiments/run_sweep.py

Ablation sweep for Tables 6-10.

Conditions cover: latent_dim, noise levels, epsilon schedule, lambda_reg,
use_integration_matrix, reward_noise, observation_noise, UCB-c.

Outputs:
  results/sweep/<condition_name>_s<seed>.json
  results/logs/sweep_<condition_name>_s<seed>.log

Usage:
    python experiments/run_sweep.py [--seeds 42] [--out-dir results/sweep]
    python experiments/run_sweep.py --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

PYTHON = sys.executable
RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_experiment_sweep.py")

DEFAULT_SEEDS = [42, 123, 456, 789, 1337]

ABLATION_CONDITIONS = [
    # Table 6: latent_dim ablation (mf_cf)
    {"name": "mf_latent1",  "policy": "mf_cf", "latent_dim": 1},
    {"name": "mf_latent2",  "policy": "mf_cf", "latent_dim": 2},
    {"name": "mf_latent3",  "policy": "mf_cf", "latent_dim": 3},   # baseline
    {"name": "mf_latent5",  "policy": "mf_cf", "latent_dim": 5},
    {"name": "mf_latent8",  "policy": "mf_cf", "latent_dim": 8},
    # Table 7: reward_noise ablation (mf_cf)
    {"name": "mf_rnoise0",   "policy": "mf_cf", "reward_noise": 0.0},
    {"name": "mf_rnoise01",  "policy": "mf_cf", "reward_noise": 0.1},
    {"name": "mf_rnoise02",  "policy": "mf_cf", "reward_noise": 0.2},  # baseline
    {"name": "mf_rnoise05",  "policy": "mf_cf", "reward_noise": 0.5},
    # Table 8: observation_noise ablation (mf_cf)
    {"name": "mf_onoise0",   "policy": "mf_cf", "observation_noise": 0.0},
    {"name": "mf_onoise01",  "policy": "mf_cf", "observation_noise": 0.1},
    {"name": "mf_onoise02",  "policy": "mf_cf", "observation_noise": 0.2},  # baseline
    {"name": "mf_onoise05",  "policy": "mf_cf", "observation_noise": 0.5},
    # Table 9: use_integration_matrix (mf_cf)
    {"name": "mf_no_intmat",  "policy": "mf_cf", "use_integration_matrix": False},
    {"name": "mf_yes_intmat", "policy": "mf_cf", "use_integration_matrix": True},  # baseline
    # Table 10: UCBIndep exploration constant
    {"name": "ucb_c05",  "policy": "ucb_indep", "ucb_c": 0.5},
    {"name": "ucb_c10",  "policy": "ucb_indep", "ucb_c": 1.0},
    {"name": "ucb_c20",  "policy": "ucb_indep", "ucb_c": 2.0},  # baseline
    {"name": "ucb_c50",  "policy": "ucb_indep", "ucb_c": 5.0},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--episodes", type=int, default=35)
    parser.add_argument("--out-dir", default="results/sweep")
    parser.add_argument("--log-dir", default="results/logs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    jobs = []
    for cond in ABLATION_CONDITIONS:
        for seed in args.seeds:
            out = os.path.join(args.out_dir, f"{cond['name']}_s{seed}.json")
            if os.path.exists(out):
                print(f"SKIP (exists): {out}")
                continue
            jobs.append((cond, seed, out))

    if args.dry_run:
        print(f"Sweep plan: {len(jobs)} job(s), {args.episodes} ep each")
        for cond, seed, out in jobs:
            print(f"  {cond['name']:20s}  seed={seed}  -> {out}")
        return 0

    print(f"[{datetime.now():%H:%M:%S}] Starting {len(jobs)} sweep job(s), "
          f"{args.episodes} episodes each...")

    failed = []
    for i, (cond, seed, out) in enumerate(jobs, 1):
        log_path = os.path.join(args.log_dir, f"sweep_{cond['name']}_s{seed}.log")
        overrides = {k: v for k, v in cond.items() if k not in ("name", "policy")}
        cmd = [PYTHON, RUNNER,
               "--policy", cond["policy"],
               "--seed", str(seed),
               "--episodes", str(args.episodes),
               "--out", out,
               "--override-json", json.dumps(overrides)]

        print(f"[{datetime.now():%H:%M:%S}] [{i:3d}/{len(jobs)}] "
              f"{cond['name']:20s} seed={seed}", end="", flush=True)

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
                print(f"  steps={s['avg_steps']:6.1f}  "
                      f"neutralized={s['avg_targets']:5.2f}  "
                      f"success={s['success_rate']:5.1f}%")
            except Exception:
                print(f"  done")
        else:
            print(f"  ERROR (exit {ret})  log={log_path}")
            failed.append((cond["name"], seed))

    print(f"\n[{datetime.now():%H:%M:%S}] run_sweep DONE: "
          f"{len(jobs) - len(failed)} ok, {len(failed)} failed")
    for name, seed in failed:
        print(f"  FAILED: {name} seed={seed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
