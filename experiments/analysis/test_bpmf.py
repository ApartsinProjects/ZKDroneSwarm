"""Quick 5-seed comparison: BPMF variants vs MF-CF, PTF, IQL-ZK."""
import subprocess, json, os, statistics, sys
sys.stdout.reconfigure(encoding="utf-8")

SEEDS = [42, 123, 456, 789, 1337]
PY = "C:/Python314/python.exe"
RUNNER = "experiments/run_experiment_sweep.py"

policies_to_test = [
    "bpmf",       # MAP-Thompson
    "bpmf_ucb",   # MAP-UCB
    "bpmf_vb",    # VB-Thompson
]

results = {}
for policy in policies_to_test:
    results[policy] = []
    for seed in SEEDS:
        out_path = f"/tmp/bpmf_test_{policy}_s{seed}.json"
        cmd = [PY, RUNNER, "--policy", policy, "--seed", str(seed),
               "--episodes", "35", "--out", out_path,
               "--override-json", '{"num_targets": 27}']
        subprocess.run(cmd, capture_output=True, text=True)
        if os.path.exists(out_path):
            with open(out_path) as f:
                d = json.load(f)
            avg = statistics.mean(e["steps"] for e in d["episodes"])
            late_lmq = statistics.mean(
                e["avg_latent_match_quality"] for e in d["episodes"][-5:]
            )
            results[policy].append((avg, late_lmq))

print(f"{'Policy':18s}  {'avg_steps_5seeds':>17s}  {'late_LMQ_5seeds':>16s}")
print("-" * 55)
for policy, runs in results.items():
    avgs = [r[0] for r in runs]
    lmqs = [r[1] for r in runs]
    avg_mean = statistics.mean(avgs)
    avg_std = statistics.stdev(avgs) if len(avgs) > 1 else 0
    lmq_mean = statistics.mean(lmqs)
    print(f"{policy:18s}  {avg_mean:9.1f} +/- {avg_std:4.1f}  {lmq_mean:>16.4f}")

print()
print("Reference (from earlier runs, n=27 baseline 5 seeds):")
print(f"  Random      107.6                 0.4251")
print(f"  UCB-Indep    69.7                 0.7260")
print(f"  IQL-ZK       63.3                 0.7517")
print(f"  ESTR         66.5                 0.7745")
print(f"  MF-CF        86.9                 0.7532")
print(f"  PTF K=5      66.1                 0.7515")
print(f"  TS-MF        78.8                 0.7687")
print(f"  Oracle-L     60.2                 0.7799")
print(f"  Oracle-HP    55.8                 0.7940")
