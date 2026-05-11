"""Aggregate results from Sets H (robustness), I (IQL sweep), J (non-orthogonal)."""
import json, os, statistics, glob, sys, math
sys.stdout.reconfigure(encoding='utf-8')

def load(pattern):
    return [json.load(open(f)) for f in sorted(glob.glob(pattern))]

def avg_steps(runs):
    vals = [statistics.mean(ep["steps"] for ep in r["episodes"]) for r in runs]
    if not vals:
        return None, None
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0)

def best_ep_steps(runs):
    vals = [min(ep["steps"] for ep in r["episodes"]) for r in runs]
    if not vals:
        return None, None
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0)

def late_lmq(runs, last_n=5):
    vals = [statistics.mean(ep["avg_latent_match_quality"] for ep in r["episodes"][-last_n:]) for r in runs]
    if not vals:
        return None, None
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0)

def avg_lmq(runs):
    vals = [statistics.mean(ep["avg_latent_match_quality"] for ep in r["episodes"]) for r in runs]
    if not vals:
        return None, None
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0)

def get_avg_steps_per_seed(runs):
    return [statistics.mean(ep["steps"] for ep in r["episodes"]) for r in runs]

# Wilcoxon signed-rank using SciPy
def wilcoxon_pair(runs_a, runs_b):
    """Paired Wilcoxon by matching seed order."""
    from scipy.stats import wilcoxon
    a = get_avg_steps_per_seed(runs_a)
    b = get_avg_steps_per_seed(runs_b)
    if len(a) != len(b) or len(a) < 2:
        return None, None
    diff = [x - y for x, y in zip(a, b)]
    if all(d == 0 for d in diff):
        return None, None
    w, p = wilcoxon(a, b, alternative="two-sided", zero_method="pratt")
    # Hodges-Lehmann pseudo-median of pairwise diffs
    pairs = []
    for x in a:
        for y in b:
            pairs.append(x - y)
    hl = statistics.median(pairs)
    return p, hl

base = "results/all_seeds"
nb = "results/new_benchmarks"

print("=" * 80)
print("SET H: 10-seed robustness check (revision item 2)")
print("=" * 80)

# Combine the 5 original seeds with the 5 new seeds
ORIG_SEEDS = [42, 123, 456, 789, 1337]
EXTRA_SEEDS = [2024, 2025, 31337, 99999, 1729]

print(f"\n{'Policy':18s}  {'n=5 avg':>10s}  {'n=10 avg':>10s}  {'best ep':>8s}  {'std':>6s}")
print("-" * 60)
data = {}
for policy in ["random", "ucb_indep", "mf_cf", "iql_zk", "ptf_k5", "oracle_l", "oracle_hp"]:
    # Original seeds (from all_seeds or new_benchmarks)
    if policy == "iql_zk":
        orig = load(os.path.join(nb, "iql_baseline_iql_zk_s*.json"))
    elif policy == "ptf_k5":
        orig = load(os.path.join(nb, "ptf_n27_k5_ptf_k5_s*.json"))
    else:
        orig = load(os.path.join(base, f"{policy}_s*.json"))
    extra = load(os.path.join(nb, f"robustness10_{policy}_s*.json"))
    all10 = orig + extra
    a5, s5 = avg_steps(orig)
    a10, s10 = avg_steps(all10) if all10 else (None, None)
    bep, _ = best_ep_steps(all10) if all10 else (None, None)
    data[policy] = (orig, extra, all10)
    if a5 is not None and a10 is not None:
        print(f"{policy:18s}  {a5:10.1f}  {a10:10.1f}  {bep:8.1f}  {s10:6.2f}")
    else:
        print(f"{policy:18s}  (incomplete)")

print()
print("Pairwise Wilcoxon (n=10 seeds), avg-steps per seed:")
pairs = [
    ("mf_cf", "ucb_indep"),
    ("mf_cf", "iql_zk"),
    ("mf_cf", "random"),
    ("ptf_k5", "ucb_indep"),
    ("ptf_k5", "iql_zk"),
    ("ptf_k5", "mf_cf"),
    ("iql_zk", "ucb_indep"),
    ("iql_zk", "random"),
]
for a, b in pairs:
    if a in data and b in data:
        runs_a = data[a][2]
        runs_b = data[b][2]
        if runs_a and runs_b and len(runs_a) == len(runs_b):
            p, hl = wilcoxon_pair(runs_a, runs_b)
            if p is not None:
                print(f"  {a:10s} vs {b:10s}: p={p:.4f}  HL_pseudo-median={hl:+.2f} steps")

print()
print("=" * 80)
print("SET I: IQL-ZK hyperparameter sweep on held-out seed 314 (revision item 3)")
print("=" * 80)
print(f"\n{'alpha':>6s}  {'decay':>7s}  {'avg steps':>10s}  {'late LMQ':>9s}")
print("-" * 42)

iql_sweep_data = {}
for alpha in [0.05, 0.10, 0.20, 0.50]:
    for decay in [0.999, 0.9995, 0.9998]:
        name = f"iql_sweep_a{int(alpha*100):02d}_d{int(decay*10000)}"
        runs = load(os.path.join(nb, f"{name}_iql_zk_s314.json"))
        if not runs:
            print(f"  {alpha:>6.2f}  {decay:>7.4f}  (missing)")
            continue
        a, _ = avg_steps(runs)
        ll, _ = late_lmq(runs)
        iql_sweep_data[(alpha, decay)] = (a, ll)
        print(f"  {alpha:>6.2f}  {decay:>7.4f}  {a:>10.1f}  {ll:>9.4f}")

# Find best config
if iql_sweep_data:
    best = min(iql_sweep_data.items(), key=lambda kv: kv[1][0])
    print(f"\nBest config: alpha={best[0][0]}, decay={best[0][1]}, avg_steps={best[1][0]:.1f}")

print()
print("=" * 80)
print("SET J: Non-orthogonal latent geometry at n=54, d=3 (revision item 6)")
print("=" * 80)

for cm in ["random", "orthogonal"]:
    print(f"\n--- center_mode = {cm} ---")
    print(f"{'Policy':18s}  {'avg steps':>10s}  {'best ep':>8s}  {'late LMQ':>9s}")
    print("-" * 50)
    for policy in ["ucb_indep", "iql_zk", "mf_cf", "ptf_k5"]:
        runs = load(os.path.join(nb, f"geom_{cm}_{policy}_s*.json"))
        if not runs:
            print(f"  {policy:18s}  (missing)")
            continue
        a, _ = avg_steps(runs)
        b, _ = best_ep_steps(runs)
        ll, _ = late_lmq(runs)
        print(f"  {policy:18s}  {a:>10.1f}  {b:>8.1f}  {ll:>9.4f}")

print()
print("Reference baseline (center_mode=one_hot at n=54):")
print(f"{'Policy':18s}  {'avg steps':>10s}")
for policy in ["ucb_indep", "mf_cf"]:
    runs = load(os.path.join(nb, f"scale_n54_{policy}_s*.json"))
    if runs:
        a, _ = avg_steps(runs)
        print(f"  {policy:18s}  {a:>10.1f}")
