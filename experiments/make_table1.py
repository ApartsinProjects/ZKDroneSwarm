"""Render the paper's headline comparison table (T1) from saved data:
  - final-policy UNSEEN-pair skill (from C14b bake-off) at rho=1 and rho=0.25
  - ANYTIME cumulative-reward skill (from C16) at rho=1 and rho=0.25
One row per method, grouped by structural class. Markdown -> docs/TABLE1_comparison.md
(regenerable; no re-runs).
"""
import json, glob
import numpy as np


def latest(p):
    fs = sorted(glob.glob(p)); return fs[-1] if fs else None


def m(xs):
    return float(np.mean(xs))


c14 = json.load(open(latest("results/pilots/c14_compare_*.json")))
c16 = json.load(open(latest("results/pilots/c16_anytime_*.json")))
order = c14["meta"]["methods"]; group = c14["meta"]["group"]
u = c14["raw"]; a = c16["raw"]

rows = []
for nm in order:
    un1 = m(u["1.0"][nm]["unseen"]); un25 = m(u["0.25"][nm]["unseen"])
    # anytime: final-round cumulative-normalized skill = traj[-1] per seed
    an1 = m([t[-1] for t in a["1.0"][nm]]); an25 = m([t[-1] for t in a["0.25"][nm]])
    rows.append((nm, group[nm], un1, un25, an1, an25))

lines = []
lines.append("# Table 1: method comparison (regenerated from saved data)\n")
lines.append("Fair guessed rank d_hat=8; block-model world; decentralized masked broadcast.")
lines.append("UNSEEN = final-policy unseen-pair skill [C14b, 5 seeds]. "
             "ANYTIME = final-round cumulative-reward skill [C16, 8 seeds].")
lines.append("Skill = (method - random) / (oracle - random); ~0 = no better than random.\n")
lines.append("| Method | Class | UNSEEN @rho=1 | UNSEEN @rho=0.25 | ANYTIME @rho=1 | ANYTIME @rho=0.25 |")
lines.append("|---|---|---|---|---|---|")
for nm, g, un1, un25, an1, an25 in rows:
    star = " **" if g.endswith("(ours)") else " "
    nmd = ("**%s**" % nm) if g.endswith("(ours)") else nm
    lines.append("| %s | %s | %.3f | %.3f | %.3f | %.3f |" % (nmd, g, un1, un25, an1, an25))
lines.append("")
lines.append("Reading: every low-rank method clears the no-structure UNSEEN floor "
             "(estimator-independent categorical result). On ANYTIME (the operational "
             "metric) our online weighted-ALS (RewardCF/BothCF) leads at both densities; "
             "UCBIndep's strong final-policy skill collapses to ~0 anytime (n>>T); "
             "PTF leads UNSEEN only at rho=1 (full broadcast).")

open("docs/TABLE1_comparison.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("wrote docs/TABLE1_comparison.md")
