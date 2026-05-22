"""Generate the headline paper figures from SAVED experiment data
(results/pilots/*.json) -- no re-runs needed (data preservation pays off).
Figures -> docs/figures/.
"""
import json
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "docs/figures"
os.makedirs(OUT, exist_ok=True)


def latest(pattern):
    fs = sorted(glob.glob(pattern))
    return fs[-1] if fs else None


def ms(xs):
    a = np.asarray(xs, float)
    return a.mean(), a.std()


# ---- F2: C11 unseen-pair under masking ----
f = latest("results/pilots/c11_masking_*.json")
if f:
    d = json.load(open(f)); rhos = d["meta"]["rhos"]; raw = d["raw"]
    cf = [ms(raw[str(r)]["RewardCF"]["unseen"]) for r in rhos]
    tb = [ms(raw[str(r)]["Tabular"]["unseen"]) for r in rhos]
    x = rhos
    plt.figure(figsize=(5, 3.5))
    plt.errorbar(x, [c[0] for c in cf], yerr=[c[1] for c in cf], marker="o", label="CF (RewardCF)", capsize=3)
    plt.errorbar(x, [t[0] for t in tb], yerr=[t[1] for t in tb], marker="s", label="Tabular (independent)", capsize=3)
    plt.axhline(0, color="gray", lw=0.8, ls=":")
    plt.gca().invert_xaxis()
    plt.xlabel("observation density  rho  (fraction of broadcast seen)")
    plt.ylabel("UNSEEN-pair skill")
    plt.title("Acting on never-observed pairs under heterogeneous masking")
    plt.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/F2_unseen_masking.png", dpi=150); plt.close()
    print("F2_unseen_masking.png  <-", os.path.basename(f))

# ---- F3: C12 onboarding ----
f = latest("results/pilots/c12_onboard_*.json")
if f:
    d = json.load(open(f)); pl = d["meta"]["probes_list"]; raw = d["raw"]; m = d["meta"]["m"]; dh = d["meta"]["d_hat"]
    cf = [ms(raw[str(p)]["cf"]) for p in pl]; tb = [ms(raw[str(p)]["tab"]) for p in pl]
    plt.figure(figsize=(5, 3.5))
    plt.errorbar(pl, [c[0] for c in cf], yerr=[c[1] for c in cf], marker="o", label="CF (fold-in given P)", capsize=3)
    plt.errorbar(pl, [t[0] for t in tb], yerr=[t[1] for t in tb], marker="s", label="Tabular (self-probed only)", capsize=3)
    plt.axvline(dh, color="green", lw=0.8, ls="--", label=f"d_hat={dh}")
    plt.axvline(m, color="red", lw=0.8, ls="--", label=f"m={m}")
    plt.xscale("log")
    plt.xlabel("# shared probes per NEW target")
    plt.ylabel("skill on new targets")
    plt.title("Dynamic target onboarding:  Theta(d) vs Theta(m)")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{OUT}/F3_onboard.png", dpi=150); plt.close()
    print("F3_onboard.png  <-", os.path.basename(f))

# ---- F4: C13 unseen vs true rank ----
f = latest("results/pilots/c13_rank_unseen_*.json")
if f:
    d = json.load(open(f)); ranks = d["meta"]["ranks"]; raw = d["raw"]
    cf = [ms(raw[str(r)]["cf_unseen"]) for r in ranks]; tb = [ms(raw[str(r)]["tab_unseen"]) for r in ranks]
    plt.figure(figsize=(5, 3.5))
    plt.errorbar(ranks, [c[0] for c in cf], yerr=[c[1] for c in cf], marker="o", label="CF unseen", capsize=3)
    plt.errorbar(ranks, [t[0] for t in tb], yerr=[t[1] for t in tb], marker="s", label="Tabular unseen", capsize=3)
    plt.axhline(0, color="gray", lw=0.8, ls=":")
    plt.xlabel("true rank  d")
    plt.ylabel("UNSEEN-pair skill")
    plt.title("CF unseen-pair skill scales with low-rankness (Tabular at floor)")
    plt.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/F4_rank.png", dpi=150); plt.close()
    print("F4_rank.png  <-", os.path.basename(f))

print("figures written to", OUT)
