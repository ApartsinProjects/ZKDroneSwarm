"""Aggregate Set N: Structural Assumption Stress Test.

Tests the central thesis: CF methods dominate tabular methods when the
compatibility matrix is low-rank, and lose when it approaches full rank.
"""
import json, os, statistics, glob, sys
sys.stdout.reconfigure(encoding="utf-8")

def load(pattern):
    return [json.load(open(f)) for f in sorted(glob.glob(pattern))]

def avg_steps(runs):
    vals = [statistics.mean(ep["steps"] for ep in r["episodes"]) for r in runs]
    if not vals: return None, None
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0)

def late_lmq(runs, last_n=5):
    vals = [statistics.mean(ep["avg_latent_match_quality"] for ep in r["episodes"][-last_n:]) for r in runs]
    if not vals: return None
    return statistics.mean(vals)

nb = "results/new_benchmarks"
TABULAR = ["ucb_indep", "iql_zk"]
CF = ["mf_cf", "ptf_k5", "bpmf"]
ALL_POL = ["ucb_indep", "iql_zk", "mf_cf", "ptf_k5", "ts_mf", "bpmf"]

conditions = [
    ("struct_d2", "d=2"),
    ("struct_d3", "d=3 (baseline)"),
    ("struct_d5", "d=5"),
    ("struct_d8", "d=8"),
    ("struct_full", "full rank (d=9, random)"),
]

print("=" * 100)
print("SET N: STRUCTURAL ASSUMPTION STRESS TEST")
print("avg_steps (5 seeds) by policy and true rank")
print("=" * 100)
header = f"{'Condition':25s}  " + "  ".join(f"{p:>9s}" for p in ALL_POL)
print(header)
print("-" * len(header))

data = {}
for cond, label in conditions:
    data[cond] = {}
    row = f"{label:25s}  "
    for p in ALL_POL:
        runs = load(os.path.join(nb, f"{cond}_{p}_s*.json"))
        if runs:
            a, _ = avg_steps(runs)
            data[cond][p] = a
            row += f"{a:9.1f}  "
        else:
            data[cond][p] = None
            row += f"{'--':>9s}  "
    print(row)

print()
print("=" * 100)
print("CF vs TABULAR ratio (best-CF / best-tabular). <1 means CF wins.")
print("=" * 100)
print(f"{'Condition':25s}  {'best CF':>9s}  {'best tab':>9s}  {'ratio':>7s}  {'prediction'}")
print("-" * 90)
for cond, label in conditions:
    cf_vals = [data[cond][p] for p in CF if data[cond].get(p) is not None]
    tab_vals = [data[cond][p] for p in TABULAR if data[cond].get(p) is not None]
    if cf_vals and tab_vals:
        best_cf = min(cf_vals)
        best_tab = min(tab_vals)
        ratio = best_cf / best_tab
        if "d=2" in label: pred = "CF should start winning"
        elif "baseline" in label: pred = "~tie (existing finding)"
        elif "d=5" in label: pred = "CF advantage shrinking"
        elif "d=8" in label: pred = "~tie (existing §7.15)"
        elif "full" in label: pred = "CF should LOSE (ratio>1)"
        else: pred = ""
        flag = "CF wins" if ratio < 0.97 else ("CF loses" if ratio > 1.03 else "~tie")
        print(f"{label:25s}  {best_cf:9.1f}  {best_tab:9.1f}  {ratio:7.3f}  [{flag}] {pred}")

print()
print("=" * 100)
print("LATE LMQ (latent recovery) by policy and rank")
print("=" * 100)
header = f"{'Condition':25s}  " + "  ".join(f"{p:>9s}" for p in ALL_POL)
print(header)
print("-" * len(header))
for cond, label in conditions:
    row = f"{label:25s}  "
    for p in ALL_POL:
        runs = load(os.path.join(nb, f"{cond}_{p}_s*.json"))
        if runs:
            ll = late_lmq(runs)
            row += f"{ll:9.4f}  "
        else:
            row += f"{'--':>9s}  "
    print(row)

print()
print("=" * 100)
print("BPMF vs IQL-ZK head-to-head (the two strongest policies)")
print("=" * 100)
print(f"{'Condition':25s}  {'BPMF':>9s}  {'IQL-ZK':>9s}  {'BPMF/IQL':>9s}")
print("-" * 60)
for cond, label in conditions:
    bpmf = data[cond].get("bpmf")
    iql = data[cond].get("iql_zk")
    if bpmf and iql:
        print(f"{label:25s}  {bpmf:9.1f}  {iql:9.1f}  {bpmf/iql:9.3f}")
