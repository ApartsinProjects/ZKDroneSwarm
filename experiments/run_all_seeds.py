"""
experiments/run_all_seeds.py

Run all 5 policies across 5 seeds to produce Tables 3 and 4.

Seeds:  42, 123, 456, 789, 1337
Policies: random, oracle_hp, oracle_l, ucb_indep, mf_cf

Outputs:
  results/all_seeds/<policy>_s<seed>.json   -- per-run result JSON
  results/logs/all_seeds_<policy>_s<seed>.log -- per-run stdout log

Usage:
    python experiments/run_all_seeds.py [--episodes N] [--out-dir results/all_seeds]
    python experiments/run_all_seeds.py --dry-run   # print plan only
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime

SEEDS = [42, 123, 456, 789, 1337]
POLICIES = ["random", "oracle_hp", "oracle_l", "ucb_indep", "mf_cf"]

PYTHON = sys.executable
RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_experiment.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=35,
                        help="Episodes per seed for all policies (default: 35)")
    parser.add_argument("--out-dir", default="results/all_seeds")
    parser.add_argument("--log-dir", default="results/logs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true",
                        help="Pass --verbose to each run_experiment call")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    jobs = []
    for policy in POLICIES:
        for seed in SEEDS:
            out = os.path.join(args.out_dir, f"{policy}_s{seed}.json")
            if os.path.exists(out):
                print(f"SKIP (exists): {out}")
                continue
            jobs.append((policy, seed, out))

    if args.dry_run:
        print(f"Plan: {len(jobs)} job(s), {args.episodes} ep each")
        for policy, seed, out in jobs:
            print(f"  {policy} seed={seed} -> {out}")
        return 0

    print(f"[{datetime.now():%H:%M:%S}] Starting {len(jobs)} job(s), "
          f"{args.episodes} episodes each...")

    failed = []
    for i, (policy, seed, out) in enumerate(jobs, 1):
        log_path = os.path.join(args.log_dir, f"all_seeds_{policy}_s{seed}.log")
        cmd = [PYTHON, RUNNER,
               "--policy", policy,
               "--seed", str(seed),
               "--episodes", str(args.episodes),
               "--out", out]
        if args.verbose:
            cmd.append("--verbose")

        print(f"[{datetime.now():%H:%M:%S}] [{i:2d}/{len(jobs)}] "
              f"{policy:12s} seed={seed}", end="", flush=True)

        with open(log_path, "w", encoding="utf-8") as log_fh:
            log_fh.write(f"CMD: {' '.join(cmd)}\n")
            log_fh.write(f"START: {datetime.now()}\n\n")
            ret = subprocess.call(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
            log_fh.write(f"\nEND: {datetime.now()}  exit={ret}\n")

        if ret == 0:
            # Print summary line from JSON
            try:
                import json
                with open(out, encoding="utf-8") as f:
                    d = json.load(f)
                s = d["summary"]
                print(f"  steps={s['avg_steps']:6.1f}  "
                      f"neutralized={s['avg_targets']:5.2f}  "
                      f"success={s['success_rate']:5.1f}%  "
                      f"log={log_path}")
            except Exception:
                print(f"  done  log={log_path}")
        else:
            print(f"  ERROR (exit {ret})  log={log_path}")
            failed.append((policy, seed))

    print(f"\n[{datetime.now():%H:%M:%S}] run_all_seeds DONE: "
          f"{len(jobs) - len(failed)} ok, {len(failed)} failed")
    for policy, seed in failed:
        print(f"  FAILED: {policy} seed={seed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
