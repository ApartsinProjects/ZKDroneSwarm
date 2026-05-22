"""Bootstrap 95% CIs + paired-difference CIs for the headline results, from SAVED
data (no re-run). Uses crossover c15 (unseen, 8 seeds) and anytime c16 (8 seeds).
Paired difference (ours - baseline) per seed -> bootstrap the mean difference;
'significant' if the 95% CI excludes 0. Writes docs/CI_REPORT.md.
"""
import json, glob
import numpy as np

RNG = np.random.RandomState(0)


def latest(p):
    fs = sorted(glob.glob(p)); return fs[-1] if fs else None


def boot_ci(vals, B=10000):
    a = np.asarray(vals, float); n = len(a)
    if n == 0:
        return (float("nan"),) * 3
    idx = RNG.randint(0, n, size=(B, n))
    means = a[idx].mean(1)
    return float(a.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_ci(ours, base, B=10000):
    a = np.asarray(ours, float); b = np.asarray(base, float)
    n = min(len(a), len(b)); a, b = a[:n], b[:n]; diff = a - b
    idx = RNG.randint(0, n, size=(B, n))
    md = diff[idx].mean(1)
    lo, hi = np.percentile(md, 2.5), np.percentile(md, 97.5)
    sig = "yes" if (lo > 0 or hi < 0) else "no"
    return float(diff.mean()), float(lo), float(hi), sig


L = ["# CI report (bootstrap 95%, from saved data; no re-run)\n",
     "Per-seed bootstrap (10k resamples). Paired difference = ours - baseline per seed;",
     "significant if the 95% CI excludes 0. Data: crossover c15 (unseen, 8 seeds),",
     "anytime c16 (8 seeds). For tighter CIs, E1 re-runs at 20 seeds.\n"]

c15 = json.load(open(latest("results/pilots/c15_crossover_*.json")))
c16 = json.load(open(latest("results/pilots/c16_anytime_*.json")))

for rho in ["1.0", "0.25"]:
    L.append("## UNSEEN-pair skill, rho=%s (c15)" % rho)
    L.append("| method | mean | 95% CI |")
    L.append("|---|---|---|")
    present = [m for m in ["HybridCF", "RewardCF", "BothCF", "PTF", "ESTR", "BPMF", "UCBIndep"]
               if m in c15["raw"][rho]]
    for m in present:
        mu, lo, hi = boot_ci(c15["raw"][rho][m]["unseen"])
        L.append("| %s | %.3f | [%.3f, %.3f] |" % (m, mu, lo, hi))
    L.append("\n_Paired (ours - baseline), unseen, rho=%s:_" % rho)
    L.append("| comparison | mean diff | 95% CI | sig |")
    L.append("|---|---|---|---|")
    for ours in ["HybridCF", "RewardCF"]:
        for base in ["PTF", "UCBIndep"]:
            if ours in c15["raw"][rho] and base in c15["raw"][rho]:
                d, lo, hi, sig = paired_ci(c15["raw"][rho][ours]["unseen"], c15["raw"][rho][base]["unseen"])
                L.append("| %s - %s | %+.3f | [%+.3f, %+.3f] | %s |" % (ours, base, d, lo, hi, sig))
    L.append("")

for rho in ["1.0", "0.25"]:
    L.append("## ANYTIME (final-round cumulative) skill, rho=%s (c16)" % rho)
    L.append("| method | mean | 95% CI |")
    L.append("|---|---|---|")
    def final(m):
        return [t[-1] for t in c16["raw"][rho][m]]
    present = [m for m in ["RewardCF", "BothCF", "HybridCF", "PTF", "ESTR", "Tabular", "UCBIndep"]
               if m in c16["raw"][rho]]
    for m in present:
        mu, lo, hi = boot_ci(final(m))
        L.append("| %s | %.3f | [%.3f, %.3f] |" % (m, mu, lo, hi))
    L.append("\n_Paired (ours - baseline), anytime, rho=%s:_" % rho)
    L.append("| comparison | mean diff | 95% CI | sig |")
    L.append("|---|---|---|---|")
    for ours in ["RewardCF"]:
        for base in ["PTF", "ESTR", "Tabular", "UCBIndep"]:
            if ours in c16["raw"][rho] and base in c16["raw"][rho]:
                d, lo, hi, sig = paired_ci(final(ours), final(base))
                L.append("| %s - %s | %+.3f | [%+.3f, %+.3f] | %s |" % (ours, base, d, lo, hi, sig))
    L.append("")

open("docs/CI_REPORT.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("wrote docs/CI_REPORT.md")
