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

# ---- F5: C15 masking-robustness crossover (unseen skill vs rho) ----
f = latest("results/pilots/c15_crossover_*.json")
if f:
    d = json.load(open(f)); rhos = d["meta"]["rhos"]; raw = d["raw"]
    # group: ours (online weighted-ALS) vs batch-SVD hybrids vs no-structure floor
    styles = {
        "HybridCF": dict(marker="D", color="C6", lw=2.4, label="HybridCF (ours: probe->SVD->online ALS)"),
        "RewardCF": dict(marker="o", color="C0", lw=2.2, label="RewardCF (ours, online ALS)"),
        "BothCF":   dict(marker="o", color="C1", lw=2.2, label="BothCF (ours, online ALS)"),
        "PTF":      dict(marker="s", color="C3", ls="--", label="PTF (probe->SVD->finetune)"),
        "ESTR":     dict(marker="^", color="C4", ls="--", label="ESTR (explore->SVD->commit)"),
        "BPMF":     dict(marker="v", color="C5", ls="--", label="BPMF (Bayesian PMF)"),
        "UCBIndep": dict(marker="x", color="gray", ls=":", label="UCBIndep (no-structure floor)"),
    }
    plt.figure(figsize=(6, 4))
    for nm, st in styles.items():
        if nm not in raw[str(rhos[0])]:
            continue
        mu = [np.mean(raw[str(r)][nm]["unseen"]) for r in rhos]
        sd = [np.std(raw[str(r)][nm]["unseen"]) for r in rhos]
        plt.errorbar(rhos, mu, yerr=sd, capsize=2, markersize=5, **st)
    plt.axhline(0, color="black", lw=0.8, ls=":")
    plt.gca().invert_xaxis()  # left = full broadcast, right = heavy masking
    plt.xlabel("observation density  rho  (fraction of broadcast seen)")
    plt.ylabel("UNSEEN-pair skill")
    plt.title("Masking-robustness (unseen-pair skill): our online-ALS methods stay\nflat as the broadcast is masked; batch-SVD hybrids (PTF/ESTR/BPMF) decay", fontsize=10)
    plt.legend(fontsize=7, loc="lower left"); plt.tight_layout()
    plt.savefig(f"{OUT}/F5_crossover.png", dpi=150); plt.close()
    print("F5_crossover.png  <-", os.path.basename(f))

# ---- F6: C16 anytime cumulative-reward trajectory (rho=0.25) ----
f = latest("results/pilots/c16_anytime_*.json")
if f:
    d = json.load(open(f)); raw = d["raw"]; T = d["meta"]["T"]
    rho_key = "0.25" if "0.25" in raw else list(raw.keys())[-1]
    rounds = np.arange(1, T + 1)
    styles = {
        "RewardCF": dict(color="C0", lw=2.3, label="RewardCF (ours, online ALS)"),
        "BothCF":   dict(color="C1", lw=2.3, label="BothCF (ours, online ALS)"),
        "PTF":      dict(color="C3", ls="--", label="PTF (probe-then-fit)"),
        "ESTR":     dict(color="C4", ls="--", label="ESTR (explore-then-commit)"),
        "Tabular":  dict(color="C2", ls="-.", label="Tabular (eps-greedy own-row)"),
        "UCBIndep": dict(color="gray", ls=":", label="UCBIndep (n>>T: stuck exploring)"),
    }
    plt.figure(figsize=(6, 4))
    for nm, st in styles.items():
        if nm not in raw[rho_key]:
            continue
        A = np.array(raw[rho_key][nm])           # (seeds, T)
        mu = A.mean(0); sd = A.std(0)
        plt.plot(rounds, mu, **st)
        plt.fill_between(rounds, mu - sd, mu + sd, color=st.get("color"), alpha=0.12)
    plt.axhline(0, color="black", lw=0.8, ls=":")
    plt.axvline(int(0.4 * T), color="red", lw=0.8, ls=":", alpha=0.6)
    plt.text(int(0.4 * T) + 0.5, plt.ylim()[0] + 0.02, "probe-phase end\n(ESTR/PTF)",
             fontsize=6, color="red")
    plt.xlabel("round  t")
    plt.ylabel("cumulative-normalized skill (reward earned)")
    plt.title("Anytime reward (rho=0.25): online CF earns from round 1;\n"
              "explore-then-commit pays a probe phase; UCBIndep stuck (n>>T)", fontsize=10)
    plt.legend(fontsize=7, loc="lower right"); plt.tight_layout()
    plt.savefig(f"{OUT}/F6_anytime.png", dpi=150); plt.close()
    print("F6_anytime.png  <-", os.path.basename(f))

print("figures written to", OUT)
