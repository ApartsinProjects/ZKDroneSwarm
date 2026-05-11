"""Aggregate Set M results: two-stage noise comparison (Theorem 6' validation)."""
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
    if not vals: return None, None
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0)

nb = "results/new_benchmarks"

# Set M regimes: (label, sigma_e, sigma_r)
REGIMES = [
    ("baseline (sigma_e=0, sigma_r=0.2)", 0.0, 0.2),  # baseline from existing data
    ("obs03 (sigma_e=0, sigma_r=0.3)",    0.0, 0.3),
    ("obs05 (sigma_e=0, sigma_r=0.5)",    0.0, 0.5),
    ("eff03 (sigma_e=0.3, sigma_r=0)",    0.3, 0.0),
    ("mix03 (sigma_e=0.3, sigma_r=0.3)",  0.3, 0.3),
]

POLICIES = ["ucb_indep", "iql_zk", "mf_cf", "ptf_k5", "ts_mf", "estr"]

print("=" * 95)
print("TWO-STAGE NOISE COMPARISON (Theorem 6' validation, Set M)")
print("=" * 95)
print(f"{'Regime':45s}  " + "  ".join(f"{p:>10s}" for p in POLICIES))
print("-" * 105)

# Baseline (existing data)
baseline = {}
mapping = {
    "ucb_indep": "results/all_seeds/ucb_indep_s*.json",
    "iql_zk": "results/new_benchmarks/iql_baseline_iql_zk_s*.json",
    "mf_cf": "results/all_seeds/mf_cf_s*.json",
    "ptf_k5": "results/new_benchmarks/ptf_n27_k5_ptf_k5_s*.json",
    "ts_mf": "results/new_benchmarks/compare_n27_ts_mf_s*.json",
    "estr": "results/new_benchmarks/compare_n27_estr_s*.json",
}
for p, pat in mapping.items():
    runs = load(pat)
    if runs:
        a, _ = avg_steps(runs)
        baseline[p] = a

row = f"{'baseline (sigma_e=0, sigma_r=0.2)':45s}  "
for p in POLICIES:
    row += f"{baseline.get(p, 0):10.1f}  "
print(row)

# Set M regimes
regime_data = {}
for label, sigma_e, sigma_r in REGIMES[1:]:
    # Match filename: "noise_obs03", "noise_obs05", "noise_eff03", "noise_mix03"
    sigma_e_int = int(sigma_e * 10)
    sigma_r_int = int(sigma_r * 10)
    if sigma_e == 0 and sigma_r == 0.3: rname = "obs03"
    elif sigma_e == 0 and sigma_r == 0.5: rname = "obs05"
    elif sigma_e == 0.3 and sigma_r == 0: rname = "eff03"
    elif sigma_e == 0.3 and sigma_r == 0.3: rname = "mix03"
    regime_data[rname] = {}
    row = f"{label:45s}  "
    for p in POLICIES:
        runs = load(os.path.join(nb, f"noise_{rname}_{p}_s*.json"))
        if runs:
            a, _ = avg_steps(runs)
            regime_data[rname][p] = a
            row += f"{a:10.1f}  "
        else:
            row += f"{'?':>10s}  "
    print(row)

print()
print("LMQ comparison (late episode, episodes 31-35):")
print(f"{'Regime':45s}  " + "  ".join(f"{p:>10s}" for p in POLICIES))
print("-" * 105)

for label, sigma_e, sigma_r in REGIMES[1:]:
    if sigma_e == 0 and sigma_r == 0.3: rname = "obs03"
    elif sigma_e == 0 and sigma_r == 0.5: rname = "obs05"
    elif sigma_e == 0.3 and sigma_r == 0: rname = "eff03"
    elif sigma_e == 0.3 and sigma_r == 0.3: rname = "mix03"
    row = f"{label:45s}  "
    for p in POLICIES:
        runs = load(os.path.join(nb, f"noise_{rname}_{p}_s*.json"))
        if runs:
            ll, _ = late_lmq(runs)
            row += f"{ll:10.4f}  "
        else:
            row += f"{'?':>10s}  "
    print(row)

print()
print("=" * 95)
print("MF-CF / tabular RATIO (lower is better for MF-CF)")
print("=" * 95)
print(f"{'Regime':35s}  {'MF-CF/UCB-Indep':>17s}  {'PTF/IQL-ZK':>13s}  {'TS-MF/IQL-ZK':>15s}  T6' prediction")
print("-" * 100)
for label, sigma_e, sigma_r in REGIMES[1:]:
    if sigma_e == 0 and sigma_r == 0.3: rname = "obs03"
    elif sigma_e == 0 and sigma_r == 0.5: rname = "obs05"
    elif sigma_e == 0.3 and sigma_r == 0: rname = "eff03"
    elif sigma_e == 0.3 and sigma_r == 0.3: rname = "mix03"
    d = regime_data[rname]
    mf_ucb = d.get("mf_cf", 0) / d.get("ucb_indep", 1) if d.get("ucb_indep") else None
    ptf_iql = d.get("ptf_k5", 0) / d.get("iql_zk", 1) if d.get("iql_zk") else None
    ts_iql = d.get("ts_mf", 0) / d.get("iql_zk", 1) if d.get("iql_zk") else None

    # Predict T6' variance ratio
    if sigma_e > 0 and sigma_r == 0:
        pred = "ratio approx 1.0 (no advantage, pure effect noise)"
    elif sigma_e == 0 and sigma_r > 0:
        pred = "ratio approx sqrt(1/m) = 0.33 (MF should win)"
    elif sigma_e > 0 and sigma_r > 0:
        ratio = (sigma_e**2 + sigma_r**2/9) / (sigma_e**2 + sigma_r**2)
        pred = f"ratio approx {ratio:.2f} (partial advantage)"
    else:
        pred = "n/a"
    print(f"{label:35s}  {mf_ucb:>17.3f}  {ptf_iql:>13.3f}  {ts_iql:>15.3f}  {pred}")
