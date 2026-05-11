"""Aggregate Set K (ESTR + TS-MF) results."""
import json, os, statistics, glob, sys
sys.stdout.reconfigure(encoding="utf-8")

def load(pattern):
    return [json.load(open(f)) for f in sorted(glob.glob(pattern))]

def avg_steps(runs):
    vals = [statistics.mean(ep["steps"] for ep in r["episodes"]) for r in runs]
    if not vals:
        return None
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0)

def best_ep_steps(runs):
    vals = [min(ep["steps"] for ep in r["episodes"]) for r in runs]
    return statistics.mean(vals) if vals else None

def late_lmq(runs, last_n=5):
    vals = [statistics.mean(ep["avg_latent_match_quality"] for ep in r["episodes"][-last_n:]) for r in runs]
    return statistics.mean(vals) if vals else None

nb = "results/new_benchmarks"
base = "results/all_seeds"

print("=" * 90)
print("Set K (ESTR + TS-MF) results, plus all-method comparison")
print("=" * 90)

# At n=27
print(f"\n--- n=27, d=3, 5 seeds ---")
print(f"{'Policy':22s}  {'avg':>10s}  {'best ep':>9s}  {'late LMQ':>10s}")
print("-" * 60)

# Original baseline runs
random_27 = load(os.path.join(base, "random_s*.json"))
ucb_27    = load(os.path.join(base, "ucb_indep_s*.json"))
mf_27     = load(os.path.join(base, "mf_cf_s*.json"))
iql_27    = load(os.path.join(nb, "iql_baseline_iql_zk_s*.json"))
ptf5_27   = load(os.path.join(nb, "ptf_n27_k5_ptf_k5_s*.json"))
estr_27   = load(os.path.join(nb, "compare_n27_estr_s*.json"))
ts_27     = load(os.path.join(nb, "compare_n27_ts_mf_s*.json"))
ol_27     = load(os.path.join(base, "oracle_l_s*.json"))
oh_27     = load(os.path.join(base, "oracle_hp_s*.json"))

for label, runs in [("Random", random_27), ("UCB-Indep", ucb_27),
                    ("IQL-ZK", iql_27), ("ESTR (centralised)", estr_27),
                    ("MF-CF", mf_27), ("TS-MF (Thompson)", ts_27),
                    ("PTF K=5", ptf5_27),
                    ("Oracle-L", ol_27), ("Oracle-HP", oh_27)]:
    if not runs:
        print(f"  {label:22s}  (missing)")
        continue
    a = avg_steps(runs)[0]
    b = best_ep_steps(runs)
    ll = late_lmq(runs)
    print(f"  {label:22s}  {a:>10.1f}  {b:>9.1f}  {ll:>10.4f}")

print(f"\n--- n=108, d=3, 5 seeds ---")
print(f"{'Policy':22s}  {'avg':>10s}  {'best ep':>9s}  {'late LMQ':>10s}")
print("-" * 60)

ucb_108   = load(os.path.join(nb, "scale_n108_ucb_indep_s*.json"))
mf_108    = load(os.path.join(nb, "scale_n108_mf_cf_s*.json"))
iql_108   = load(os.path.join(nb, "scale_n108_iql_zk_s*.json"))
ptf8_108  = load(os.path.join(nb, "ptf_n108_k8_ptf_k8_s*.json"))
estr_108  = load(os.path.join(nb, "compare_n108_estr_s*.json"))
ts_108    = load(os.path.join(nb, "compare_n108_ts_mf_s*.json"))

for label, runs in [("UCB-Indep", ucb_108),
                    ("IQL-ZK", iql_108),
                    ("ESTR (centralised)", estr_108),
                    ("MF-CF", mf_108),
                    ("TS-MF (Thompson)", ts_108),
                    ("PTF K=8", ptf8_108)]:
    if not runs:
        print(f"  {label:22s}  (missing)")
        continue
    a = avg_steps(runs)[0]
    b = best_ep_steps(runs)
    ll = late_lmq(runs)
    print(f"  {label:22s}  {a:>10.1f}  {b:>9.1f}  {ll:>10.4f}")
