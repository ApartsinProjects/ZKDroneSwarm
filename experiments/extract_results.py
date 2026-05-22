"""
experiments/extract_results.py

Read all result JSONs and produce table-ready values for each paper table.

Tables produced:
  Table 3: Per-policy per-seed mean metrics (avg_steps, avg_targets, success_rate, lmq)
  Table 4: Cross-seed mean +/- std for each policy
  Table 5: Convergence analysis for mf_cf (first-half vs second-half episode performance)
  Table 6-10: Ablation sweeps from results/sweep/

Usage:
    python experiments/extract_results.py --all-seeds-dir results/all_seeds
    python experiments/extract_results.py --all-seeds-dir results/all_seeds --sweep-dir results/sweep
    python experiments/extract_results.py --all-seeds-dir results/all_seeds --csv tables/
"""

import argparse
import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

POLICIES_ORDER = ["random", "oracle_l", "oracle_hp", "ucb_indep", "mf_cf"]
POLICY_LABELS = {
    "random": "Random",
    "oracle_hp": "Oracle-HP",
    "oracle_l": "Oracle-L",
    "ucb_indep": "UCB-Indep",
    "mf_cf": "MF-CF",
}

METRICS = ["avg_steps", "avg_targets", "success_rate", "avg_ammo", "avg_overkill", "avg_reward"]
METRIC_LABELS = {
    "avg_steps": "Avg Steps",
    "avg_targets": "Avg Neutralized",
    "success_rate": "Success Rate (%)",
    "avg_ammo": "Avg Ammo",
    "avg_overkill": "Avg Overkill",
    "avg_reward": "Avg Reward",
}


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def std(vals):
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def load_all_seeds(directory):
    """Load all per-seed result files. Returns dict: policy -> list of summary dicts."""
    data = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        try:
            result = load_json(path)
        except Exception as e:
            print(f"WARN: could not load {path}: {e}")
            continue
        policy = result.get("policy", "unknown")
        data[policy].append(result)
    return data


def print_table3(data):
    """Per-seed values."""
    print("\n=== TABLE 3: Per-seed metrics ===")
    header = f"{'Policy':<12} {'Seed':>6} {'Steps':>8} {'Neutralized':>12} {'Success%':>10} {'Ammo':>8} {'LMQ':>8}"
    print(header)
    print("-" * len(header))
    for policy in POLICIES_ORDER:
        for result in sorted(data.get(policy, []), key=lambda r: r.get("seed", 0)):
            s = result["summary"]
            # Compute avg LMQ from episode list
            eps = result.get("episodes", [])
            lmq_vals = [e["avg_latent_match_quality"] for e in eps if "avg_latent_match_quality" in e]
            avg_lmq = mean(lmq_vals) if lmq_vals else float("nan")
            print(f"{POLICY_LABELS.get(policy, policy):<12} "
                  f"{result.get('seed', '?'):>6} "
                  f"{s['avg_steps']:>8.1f} "
                  f"{s['avg_targets']:>12.2f} "
                  f"{s['success_rate']:>10.1f} "
                  f"{s['avg_ammo']:>8.1f} "
                  f"{avg_lmq:>8.4f}")


def print_table4(data):
    """Cross-seed mean +/- std."""
    print("\n=== TABLE 4: Cross-seed summary (mean +/- std) ===")
    row_fmt = "{:<12}  {:>12}  {:>14}  {:>12}  {:>10}  {:>10}"
    print(row_fmt.format("Policy", "Avg Steps", "Avg Neutralized", "Success%", "Avg Ammo", "Avg LMQ"))
    print("-" * 75)
    for policy in POLICIES_ORDER:
        results = data.get(policy, [])
        if not results:
            print(f"{POLICY_LABELS.get(policy, policy):<12}  (no data)")
            continue

        steps_list = [r["summary"]["avg_steps"] for r in results]
        targets_list = [r["summary"]["avg_targets"] for r in results]
        success_list = [r["summary"]["success_rate"] for r in results]
        ammo_list = [r["summary"]["avg_ammo"] for r in results]

        lmq_list = []
        for r in results:
            eps = r.get("episodes", [])
            vals = [e["avg_latent_match_quality"] for e in eps if "avg_latent_match_quality" in e]
            if vals:
                lmq_list.append(mean(vals))

        def fmt(vals):
            m, s = mean(vals), std(vals)
            return f"{m:.1f} +/- {s:.1f}"

        def fmt4(vals):
            m, s = mean(vals), std(vals)
            return f"{m:.4f}+/-{s:.4f}"

        print(row_fmt.format(
            POLICY_LABELS.get(policy, policy),
            fmt(steps_list),
            fmt(targets_list),
            fmt(success_list),
            fmt(ammo_list),
            fmt4(lmq_list) if lmq_list else "N/A",
        ))


def print_table5_convergence(data):
    """Convergence: compare first-half vs second-half episode performance for mf_cf."""
    print("\n=== TABLE 5: Convergence (mf_cf first-half vs second-half) ===")
    results = data.get("mf_cf", [])
    if not results:
        print("  (no mf_cf data)")
        return

    row_fmt = "{:>6}  {:>12}  {:>12}  {:>12}  {:>12}"
    print(row_fmt.format("Seed", "H1 Targets", "H2 Targets", "H1 LMQ", "H2 LMQ"))
    print("-" * 60)
    for r in sorted(results, key=lambda x: x.get("seed", 0)):
        eps = r.get("episodes", [])
        n = len(eps)
        if n < 2:
            continue
        h1, h2 = eps[: n // 2], eps[n // 2 :]
        h1_tgt = mean([e["targets_neutralized"] for e in h1])
        h2_tgt = mean([e["targets_neutralized"] for e in h2])
        h1_lmq = mean([e["avg_latent_match_quality"] for e in h1])
        h2_lmq = mean([e["avg_latent_match_quality"] for e in h2])
        print(row_fmt.format(
            r.get("seed", "?"),
            f"{h1_tgt:.2f}", f"{h2_tgt:.2f}",
            f"{h1_lmq:.4f}", f"{h2_lmq:.4f}",
        ))


def print_sweep_table(sweep_dir, ablation_group, title):
    """Print one ablation group from the sweep directory."""
    print(f"\n=== {title} ===")
    files = sorted(glob.glob(os.path.join(sweep_dir, "*.json")))
    group_data = defaultdict(list)
    for path in files:
        name = os.path.basename(path).rsplit("_s", 1)[0]
        if name in ablation_group:
            try:
                result = load_json(path)
                group_data[name].append(result)
            except Exception:
                pass

    row_fmt = "{:<20}  {:>12}  {:>14}  {:>12}"
    print(row_fmt.format("Condition", "Avg Steps", "Avg Neutralized", "Success%"))
    print("-" * 62)
    for name in ablation_group:
        results = group_data.get(name, [])
        if not results:
            print(f"{name:<20}  (no data)")
            continue
        steps = mean([r["summary"]["avg_steps"] for r in results])
        targets = mean([r["summary"]["avg_targets"] for r in results])
        success = mean([r["summary"]["success_rate"] for r in results])
        print(row_fmt.format(name, f"{steps:.1f}", f"{targets:.2f}", f"{success:.1f}"))


def save_csv(data, out_dir):
    """Save Table 4 as CSV."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "table4_summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Policy", "N_seeds", "avg_steps_mean", "avg_steps_std",
                         "avg_targets_mean", "avg_targets_std",
                         "success_rate_mean", "success_rate_std",
                         "avg_lmq_mean", "avg_lmq_std"])
        for policy in POLICIES_ORDER:
            results = data.get(policy, [])
            if not results:
                continue
            steps_list = [r["summary"]["avg_steps"] for r in results]
            targets_list = [r["summary"]["avg_targets"] for r in results]
            success_list = [r["summary"]["success_rate"] for r in results]
            lmq_list = []
            for r in results:
                vals = [e["avg_latent_match_quality"] for e in r.get("episodes", [])
                        if "avg_latent_match_quality" in e]
                if vals:
                    lmq_list.append(mean(vals))
            writer.writerow([
                policy, len(results),
                f"{mean(steps_list):.2f}", f"{std(steps_list):.2f}",
                f"{mean(targets_list):.4f}", f"{std(targets_list):.4f}",
                f"{mean(success_list):.2f}", f"{std(success_list):.2f}",
                f"{mean(lmq_list):.6f}" if lmq_list else "N/A",
                f"{std(lmq_list):.6f}" if len(lmq_list) > 1 else "N/A",
            ])
    print(f"\nCSV saved to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-seeds-dir", default="results/all_seeds")
    parser.add_argument("--sweep-dir", default=None)
    parser.add_argument("--csv", default=None, help="Directory to save CSV files")
    args = parser.parse_args()

    if not os.path.isdir(args.all_seeds_dir):
        print(f"ERROR: {args.all_seeds_dir} not found")
        return 1

    data = load_all_seeds(args.all_seeds_dir)
    print(f"Loaded data for policies: {sorted(data.keys())}")

    print_table3(data)
    print_table4(data)
    print_table5_convergence(data)

    if args.sweep_dir and os.path.isdir(args.sweep_dir):
        print_sweep_table(args.sweep_dir,
                          ["mf_latent1", "mf_latent2", "mf_latent3", "mf_latent5", "mf_latent8"],
                          "TABLE 6: latent_dim ablation (MF-CF)")
        print_sweep_table(args.sweep_dir,
                          ["mf_rnoise0", "mf_rnoise01", "mf_rnoise02", "mf_rnoise05"],
                          "TABLE 7: reward_noise ablation (MF-CF)")
        print_sweep_table(args.sweep_dir,
                          ["mf_onoise0", "mf_onoise01", "mf_onoise02", "mf_onoise05"],
                          "TABLE 8: observation_noise ablation (MF-CF)")
        print_sweep_table(args.sweep_dir,
                          ["mf_no_intmat", "mf_yes_intmat"],
                          "TABLE 9: integration_matrix ablation (MF-CF)")
        print_sweep_table(args.sweep_dir,
                          ["ucb_c05", "ucb_c10", "ucb_c20", "ucb_c50"],
                          "TABLE 10: UCB exploration constant ablation")

    if args.csv:
        save_csv(data, args.csv)

    return 0


if __name__ == "__main__":
    sys.exit(main())
