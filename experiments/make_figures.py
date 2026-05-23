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
PDF_OUT = "docs/figures/pdf"
os.makedirs(OUT, exist_ok=True)
os.makedirs(PDF_OUT, exist_ok=True)


def latest(pattern):
    fs = sorted(glob.glob(pattern))
    return fs[-1] if fs else None


def save_fig(stem):
    """Save the current figure as BOTH a raster PNG (for the HTML tutorial/paper)
    and a vector PDF (for the LaTeX camera-ready), then close it."""
    plt.savefig(f"{OUT}/{stem}.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{PDF_OUT}/{stem}.pdf", bbox_inches="tight")
    plt.close("all")


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
    plt.legend(); plt.tight_layout(); save_fig("F2_unseen_masking")
    print("F2_unseen_masking  <-", os.path.basename(f))

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
    plt.legend(fontsize=8); plt.tight_layout(); save_fig("F3_onboard")
    print("F3_onboard  <-", os.path.basename(f))

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
    plt.legend(); plt.tight_layout(); save_fig("F4_rank")
    print("F4_rank  <-", os.path.basename(f))

# ---- F5: C15 masking-robustness crossover (unseen skill vs rho) ----
f = latest("results/pilots/c15_crossover_*.json")
if f:
    d = json.load(open(f)); rhos = d["meta"]["rhos"]; raw = d["raw"]
    # group: ours (online weighted-ALS) vs batch-SVD hybrids vs no-structure floor
    styles = {
        "HybridCF": dict(marker="D", color="C6", lw=2.4, label="SwarmCF-H (probe-then-online, ours)"),
        "RewardCF": dict(marker="o", color="C0", lw=2.2, label="SwarmCF (online ALS, ours)"),
        "BothCF":   dict(marker="o", color="C1", lw=2.2, label="SwarmCF-RC (reward+choice, ours)"),
        "PTF":      dict(marker="s", color="C3", ls="--", label="PTF (probe->SVD->finetune)"),
        "ESTR":     dict(marker="^", color="C4", ls="--", label="ESTR (explore-then-commit)"),
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
    save_fig("F5_crossover")
    print("F5_crossover  <-", os.path.basename(f))

# ---- F6: C16 anytime cumulative-reward trajectory (rho=0.25) ----
f = latest("results/pilots/c16_anytime_*.json")
if f:
    d = json.load(open(f)); raw = d["raw"]; T = d["meta"]["T"]
    rho_key = "0.25" if "0.25" in raw else list(raw.keys())[-1]
    rounds = np.arange(1, T + 1)
    styles = {
        "RewardCF": dict(color="C0", lw=2.3, label="SwarmCF (online ALS, ours)"),
        "BothCF":   dict(color="C1", lw=2.3, label="SwarmCF-RC (reward+choice, ours)"),
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
    save_fig("F6_anytime")
    print("F6_anytime  <-", os.path.basename(f))

# ---- F7: E3 channel grid (overall skill vs sigma_obs, panel per rho) ----
f = latest("results/pilots/e3_channels_*.json")
if f:
    d = json.load(open(f)); raw = d["raw"]; rhos = d["meta"]["rhos"]
    sigs = d["meta"]["sigmas"]; methods = d["meta"]["methods"]
    st = {"Tabular": dict(color="gray", ls=":", marker="."),
          "RewardCF": dict(color="C0", marker="o", lw=2), "ChoiceCF": dict(color="C2", marker="s", lw=2),
          "BothCF": dict(color="C1", marker="D", lw=2), "PTF": dict(color="C3", ls="--", marker="^")}
    fig, axes = plt.subplots(1, len(rhos), figsize=(12, 3.6), sharey=True)
    for ax, rho in zip(axes, rhos):
        for nm in methods:
            mu = [np.mean(raw[str(rho)][str(s)][nm]["overall"]) for s in sigs]
            sd = [np.std(raw[str(rho)][str(s)][nm]["overall"]) for s in sigs]
            ax.errorbar(sigs, mu, yerr=sd, capsize=2, markersize=4,
                        label=nm + (" (ours)" if nm in ("RewardCF", "ChoiceCF", "BothCF") else ""), **st.get(nm, {}))
        ax.set_title("rho = %.2f" % rho, fontsize=10)
        ax.set_xlabel("reward-obs noise  sigma_obs"); ax.grid(alpha=0.25)
    axes[0].set_ylabel("overall skill")
    axes[-1].legend(fontsize=7, loc="lower left")
    fig.suptitle("Two observability channels: clean CHOICES (ChoiceCF, flat) overtake noisy "
                 "REWARDS (RewardCF, decays) as sigma_obs rises; BothCF tracks the best", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); save_fig("F7_channels")
    print("F7_channels  <-", os.path.basename(f))

# ---- F8: E12 persistent vs iid masking (Theorem 4) ----
f = latest("results/pilots/e12_iid_masking_*.json")
if f:
    d = json.load(open(f)); A = d["rawA"]; B = d["rawB"]
    rhos = d["meta"]["rhos"]; Tgrid = d["meta"]["Tgrid"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.7))
    # (a) unseen vs rho, (b) anytime vs rho: persistent (solid) vs iid (dashed)
    for col, metric in enumerate(["unseen", "anytime"]):
        for nm, color in [("RewardCF", "C0"), ("HybridCF", "C6"), ("PTF", "C3")]:
            for mode, ls, mk in [("persistent", "-", "o"), ("iid", "--", "x")]:
                if nm in A[mode][str(rhos[0])]:
                    mu = [np.mean(A[mode][str(r)][nm][metric]) for r in rhos]
                    ax[col].plot(rhos, mu, ls=ls, marker=mk, color=color, markersize=4,
                                 label="%s %s" % (nm, mode[:4]))
        ax[col].invert_xaxis(); ax[col].set_xlabel("rho"); ax[col].grid(alpha=0.25)
        ax[col].set_title("%s skill: persistent vs iid" % metric, fontsize=10)
    ax[0].set_ylabel("skill"); ax[0].legend(fontsize=6, ncol=1, loc="lower left")
    # (c) state-uniqueness vs T (RewardCF, rho=0.25)
    for mode, ls, mk, col in [("persistent", "-", "o", "C0"), ("iid", "--", "x", "C3")]:
        mu = [np.nanmean(B[mode][str(T)]["uniq"]) for T in Tgrid]
        ax[2].plot(Tgrid, mu, ls=ls, marker=mk, color=col, label="%s" % mode)
    ax[2].set_xlabel("horizon T"); ax[2].set_ylabel("state-uniqueness"); ax[2].grid(alpha=0.25)
    ax[2].set_title("decentralization durability (rho=0.25)\npersistent durable, iid transient", fontsize=9)
    ax[2].legend(fontsize=8)
    fig.suptitle("Unseen/anytime skill is invariant to the masking model; "
                 "state-uniqueness durable (persistent) vs transient (iid)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); save_fig("F8_iid_vs_persistent")
    print("F8_iid_vs_persistent  <-", os.path.basename(f))

# ---- F9: E2/E4/E6 scaling sweeps (unseen & anytime vs d, T, n, d_hat) ----
f = latest("results/pilots/e246_scaling_*.json")
if f:
    d = json.load(open(f)); raw = d["raw"]; sweeps = d["meta"]["sweeps"]; methods = d["meta"]["methods"]
    order = [s for s in ["d", "T", "n", "dhat"] if s in sweeps]
    xlabels = {"d": "true rank d", "T": "horizon T", "n": "targets n", "dhat": "guessed rank d_hat"}
    st = {"Tabular": dict(color="gray", ls=":", marker="."),
          "UCBIndep": dict(color="C7", ls=":", marker="x"),
          "PTF": dict(color="C3", ls="--", marker="s"),
          "RewardCF": dict(color="C0", marker="o", lw=2),
          "HybridCF": dict(color="C6", marker="D", lw=2)}
    fig, axes = plt.subplots(2, len(order), figsize=(3.1 * len(order), 6.4), sharex="col")
    for col, sw in enumerate(order):
        vals = sweeps[sw]
        for row, metric in enumerate(["unseen", "anytime"]):
            ax = axes[row][col]
            for nm in methods:
                mu = [np.mean(raw[sw][str(v)][nm][metric]) for v in vals]
                ax.plot(vals, mu, markersize=4, label=nm, **st.get(nm, {}))
            ax.axhline(0, color="black", lw=0.6, ls=":"); ax.grid(alpha=0.2)
            if row == 1:
                ax.set_xlabel(xlabels.get(sw, sw))
            if col == 0:
                ax.set_ylabel("%s skill" % metric)
            if sw in ("n",):
                ax.set_xscale("log")
    axes[0][0].legend(fontsize=6, loc="best")
    fig.suptitle("Scaling: unseen (top) and anytime (bottom) skill vs rank d, horizon T, "
                 "targets n, guessed rank d_hat (rho=0.5)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); save_fig("F9_scaling")
    print("F9_scaling  <-", os.path.basename(f))

# ---- F10: E7 newcomer cold-start (skill vs own probes), STRICT-ZK rho sweep ----
f = latest("results/pilots/e7_newcomer_*.json")
if f:
    d = json.load(open(f)); raw = d["raw"]; probes = d["meta"]["probes"]; dh = d["meta"]["d_hat"]
    plt.figure(figsize=(5.2, 3.6))
    if "cf" in raw:                                   # legacy flat format (pre-cycle-62)
        raw = {"1.0": raw}
    rhos = sorted(raw.keys(), key=float, reverse=True)
    cfcol = {"1.0": "C0", "0.5": "C4", "0.25": "C1"}
    for rho in rhos:                                  # one CF fold-in line per rho
        mu = [np.mean(raw[rho]["cf"][str(k)]) for k in probes]
        sd = [np.std(raw[rho]["cf"][str(k)]) for k in probes]
        plt.errorbar([max(p, 0.5) for p in probes], mu, yerr=sd, capsize=2, markersize=4,
                     marker="o", lw=2, color=cfcol.get(rho, None), label="CF foldin (rho=%s)" % rho)
    tabref = raw[rhos[0]]["tab"]                      # tabular floor (rho-independent)
    plt.errorbar([max(p, 0.5) for p in probes], [np.mean(tabref[str(k)]) for k in probes],
                 yerr=[np.std(tabref[str(k)]) for k in probes], capsize=2, markersize=4,
                 marker="s", ls="--", color="C3", label="Tabular newcomer (own probes only)")
    plt.axhline(0, color="black", lw=0.8, ls=":")
    plt.axvline(dh, color="green", lw=0.8, ls="--", label="d_hat=%d" % dh)
    plt.xscale("log")
    plt.xlabel("# own probes by the newcomer")
    plt.ylabel("newcomer skill on UNSEEN targets")
    plt.title("Newcomer cold-start (strict ZK): recovers U from its OWN masked\n"
              "broadcast; tabular at the floor (Theta(d) vs Theta(n)); slope flattens as rho falls", fontsize=8)
    plt.legend(fontsize=6); plt.tight_layout(); save_fig("F10_newcomer")
    print("F10_newcomer  <-", os.path.basename(f))

# ---- F11: Pareto frontier (anytime vs unseen) showing competitors dominated ----
f = latest("results/pilots/e8_active_*.json")
if f:
    d = json.load(open(f)); raw = d["raw"]; rhos = d["meta"]["rhos"]; methods = d["meta"]["methods"]
    panels = [r for r in (1.0, 0.25) if str(r) in raw["unseen"]]
    mk = {"RewardCF": ("o", "C0"), "HybridCFconv": ("D", "C6"), "ActiveCFconv": ("*", "C2"),
          "PTF": ("s", "C3"), "HybridCF": ("v", "C1")}
    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 4.0))
    if len(panels) == 1:
        axes = [axes]
    for ax, rho in zip(axes, panels):
        for nm in methods:
            x = np.mean(raw["anytime"][str(rho)][nm]); y = np.mean(raw["unseen"][str(rho)][nm])
            m_, c_ = mk.get(nm, ("o", "gray"))
            ours = nm in ("RewardCF", "HybridCFconv", "ActiveCFconv", "HybridCF")
            ax.scatter([x], [y], marker=m_, color=c_, s=170 if nm == "ActiveCFconv" else 90,
                       edgecolor="k" if ours else "none", zorder=3)
            ax.annotate(nm, (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("anytime cumulative-reward skill"); ax.set_ylabel("final-policy UNSEEN skill")
        ax.set_title("rho = %.2f" % rho, fontsize=10); ax.grid(alpha=0.25)
    fig.suptitle("Pareto frontier (up-right = better): our methods dominate or match PTF on both "
                 "axes;\nPTF sacrifices all anytime for unseen (dominated under masking, rho=0.25)",
                 fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); save_fig("F11_pareto")
    print("F11_pareto  <-", os.path.basename(f))

# ---- F12: more low-rank/structured baselines (E15) ----
f = latest("results/pilots/e15_morebase_*.json")
if f:
    d = json.load(open(f)); raw = d["raw"]; rhos = d["meta"]["rhos"]
    st = {"BiasModel": dict(color="gray", ls=":", marker="."),
          "KNNCF": dict(color="C5", ls="--", marker="v"),
          "SoftImpute": dict(color="C3", ls="--", marker="s"),
          "PTF": dict(color="C4", ls="--", marker="^"),
          "HybridCFconv": dict(color="C6", marker="D", lw=2),
          "ActiveCFconv": dict(color="C2", marker="*", lw=2)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ["unseen", "anytime"]):
        for nm, s in st.items():
            if nm in raw[metric][str(rhos[0])]:
                mu = [np.mean(raw[metric][str(r)][nm]) for r in rhos]
                sd = [np.std(raw[metric][str(r)][nm]) for r in rhos]
                ax.errorbar(rhos, mu, yerr=sd, capsize=2, markersize=5,
                            label=nm + (" (ours)" if "conv" in nm else ""), **s)
        ax.invert_xaxis(); ax.axhline(0, color="black", lw=0.7, ls=":")
        ax.set_xlabel("observation density rho"); ax.set_ylabel("%s skill" % metric)
        ax.set_title("%s skill vs rho" % metric, fontsize=10); ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("More fair baselines: SoftImpute (convex), kNN-CF (memory), BiasModel (additive) "
                 "vs ours.\nOurs dominate under masking and on anytime; SoftImpute leads only "
                 "dense-rho unseen.", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.92]); save_fig("F12_morebaselines")
    print("F12_morebaselines  <-", os.path.basename(f))

# ---- F13: real tabula_drone simulator validation ----
f = "results/pilots/tabula_bench_real.json"
if os.path.exists(f):
    d = json.load(open(f)); sk = d["skill"]; traj = d["traj"]
    order = ["random", "mf", "ucb_indep", "weighted_als", "oracle"]
    lab = {"random": "Random", "mf": "MF (env SGD)", "ucb_indep": "UCBIndep",
           "weighted_als": "WeightedALS (ours)", "oracle": "Oracle"}
    col = {"random": "gray", "mf": "C3", "ucb_indep": "C4", "weighted_als": "C2", "oracle": "k"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    present = [p for p in order if p in sk]
    mu = [np.mean(sk[p]) for p in present]; sd = [np.std(sk[p]) for p in present]
    ax[0].bar(range(len(present)), mu, yerr=sd, capsize=4,
              color=[col[p] for p in present])
    ax[0].set_xticks(range(len(present))); ax[0].set_xticklabels([lab[p] for p in present], rotation=20, ha="right", fontsize=8)
    ax[0].set_ylabel("skill = (policy - random)/(oracle - random)")
    ax[0].set_title("Real simulator: converged skill", fontsize=10); ax[0].grid(alpha=0.25, axis="y")
    for p in ["random", "mf", "ucb_indep", "weighted_als", "oracle"]:
        if p in traj:
            arr = np.array(traj[p]); muc = arr.mean(0)
            ax[1].plot(range(1, len(muc) + 1), muc, label=lab[p], color=col[p],
                       lw=2 if p == "weighted_als" else 1.3,
                       ls="-" if p in ("weighted_als", "random", "oracle") else "--")
    ax[1].set_xlabel("episode"); ax[1].set_ylabel("reward per step")
    ax[1].set_title("Learning curves", fontsize=10); ax[1].grid(alpha=0.25); ax[1].legend(fontsize=7)
    fig.suptitle("Validation in the real tabula_drone simulator (spatial, HP-depletion, episodic): "
                 "ours beats the env's SGD-MF and UCBIndep, approaching the oracle", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); save_fig("F13_realsim")
    print("F13_realsim  <- tabula_bench_real.json")

# ---- F14: assumption-stress (approx low-rank + nonlinear link) ----
f = latest("results/pilots/stress_assump_*.json")
if f:
    d = json.load(open(f)); raw = d["raw"]; sweeps = d["meta"]["sweeps"]; methods = d["meta"]["methods"]
    st = {"UCBIndep": dict(color="gray", ls=":", marker="x"), "PTF": dict(color="C3", ls="--", marker="^"),
          "RewardCF": dict(color="C0", marker="o"), "HybridCFconv": dict(color="C6", marker="D", lw=2),
          "ActiveCFconv": dict(color="C2", marker="*", lw=2)}
    knobs = [k for k in ("approx", "nonlin") if k in sweeps]
    fig, axes = plt.subplots(2, len(knobs), figsize=(5.4 * len(knobs), 7))
    title = {"approx": "approximate low-rank (entrywise noise)", "nonlin": "nonlinear reward link"}
    for col, kn in enumerate(knobs):
        vals = sweeps[kn]
        ers = [int(round(np.mean(raw[kn][str(v)][methods[0]]["er"]))) for v in vals]
        xt = ["%s\n(er=%d)" % (v, e) for v, e in zip(vals, ers)]
        for row, metric in enumerate(["unseen", "anytime"]):
            ax = axes[row][col]
            for nm in methods:
                mu = [np.mean(raw[kn][str(v)][nm][metric]) for v in vals]
                ax.plot(range(len(vals)), mu, markersize=5, label=nm + (" (ours)" if "conv" in nm else ""), **st.get(nm, {}))
            ax.axhline(0, color="black", lw=0.6, ls=":"); ax.grid(alpha=0.25)
            ax.set_xticks(range(len(vals))); ax.set_xticklabels(xt, fontsize=8)
            ax.set_ylabel("%s skill" % metric)
            if row == 0:
                ax.set_title(title[kn], fontsize=10)
    axes[0][0].legend(fontsize=7, loc="best")
    fig.suptitle("Assumption stress: graceful degradation as the world leaves exact low-rank "
                 "(er = realized effective rank); ours stay best", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); save_fig("F14_stress")
    print("F14_stress  <-", os.path.basename(f))

# ---- F15: de-confliction primitives (earned reward vs pool), capacity-1 contention ----
f = latest("results/pilots/contention8_*.json")
if f:
    d = json.load(open(f)); raw = d["raw"]; pools = d["meta"]["pools"]
    cand = ["ContentionAdaCF", "ContentionCF", "CBBAlite", "MusicalChairs", "RewardCFconv", "PTF"]
    methods = [m for m in cand if m in raw[str(pools[0])]]
    sty = {"ContentionAdaCF": dict(color="C0", marker="o", lw=2),
           "ContentionCF": dict(color="C1", marker="o", lw=2, ls="--"),
           "CBBAlite": dict(color="C2", marker="s"),
           "MusicalChairs": dict(color="C4", marker="^"),
           "RewardCFconv": dict(color="C3", marker="x", ls=":"),
           "PTF": dict(color="C5", marker="d", ls=":")}
    lab = {"ContentionAdaCF": "ContentionAdaCF (ours)", "ContentionCF": "ContentionCF (ours)",
           "CBBAlite": "CBBAlite (CBBA auction+backoff)", "MusicalChairs": "MusicalChairs (SIC-MMAB re-seat)",
           "RewardCFconv": "greedy RewardCF", "PTF": "PTF (batch low-rank)"}
    x = list(range(len(pools)))
    plt.figure(figsize=(5.4, 3.8))
    for nm in methods:
        mu = [np.mean(raw[str(p)][nm]["anytime"]) for p in pools]
        sd = [np.std(raw[str(p)][nm]["anytime"]) for p in pools]
        plt.errorbar(x, mu, yerr=sd, capsize=2, markersize=5, label=lab.get(nm, nm), **sty.get(nm, {}))
    plt.xticks(x, ["%d" % p for p in pools])
    plt.xlabel("offer pool size |S| (left = no contention, right = severe; m=30 drones)")
    plt.ylabel("earned-reward skill (matching-normalized)")
    plt.title("De-confliction under capacity-1 contention: proactive PRIVATE offset (ours)\n"
              "beats the auction-backoff and MAB re-seating field primitives at severe contention", fontsize=8)
    plt.axhline(0, color="black", lw=0.6, ls=":"); plt.grid(alpha=0.25)
    plt.legend(fontsize=6); plt.tight_layout(); save_fig("F15_deconfliction")
    print("F15_deconfliction  <-", os.path.basename(f))

# ---- F16: sensing-grounded observability (unseen skill vs sensing coverage) ----
f = latest("results/pilots/sensing_*.json")
if f:
    d = json.load(open(f)); raw = d["raw"]; Rs = d["meta"]["R_senses"]; covd = d["meta"].get("coverage", {})
    xs = [covd.get(str(r), r) for r in Rs]                 # x-axis = effective coverage
    sty = {"RewardCF": dict(color="C0", marker="o", lw=2, label="RewardCF (low-rank CF, ours)"),
           "KNNCF": dict(color="C1", marker="s", label="KNNCF (memory CF)"),
           "Tabular": dict(color="C3", marker="x", ls="--", label="Tabular (no structure)"),
           "UCBIndep": dict(color="C7", marker="d", ls=":", label="UCBIndep (no structure)")}
    plt.figure(figsize=(5.2, 3.8))
    for nm in ["RewardCF", "KNNCF", "Tabular", "UCBIndep"]:
        if nm not in raw[str(Rs[0])]:
            continue
        mu = [np.mean(raw[str(r)][nm]["unseen"]) for r in Rs]
        sd = [np.std(raw[str(r)][nm]["unseen"]) for r in Rs]
        plt.errorbar(xs, mu, yerr=sd, capsize=2, markersize=5, **sty[nm])
    plt.axhline(0, color="black", lw=0.7, ls=":")
    plt.xlabel("effective sensing coverage (fraction of engagements sensed)")
    plt.ylabel("unseen-pair skill")
    plt.title("Sensing-grounded observability: the categorical win survives\n"
              "geometry-limited, distance-noisy sensing (masking emergent from physics)", fontsize=8)
    plt.grid(alpha=0.25); plt.legend(fontsize=7); plt.tight_layout(); save_fig("F16_sensing")
    print("F16_sensing  <-", os.path.basename(f))

# ---- F17: operational target-servicing mission (ours vs FULL field, masked) ----
f = latest("results/pilots/strike_*.json")
if f:
    d = json.load(open(f))
    sk = d.get("skill", {}); rhos = d["meta"]["rhos"]; methods = d["meta"]["methods"]
    if sk and all(str(r) in sk for r in rhos):
        x = np.arange(len(methods)); wbar = 0.8 / max(len(rhos), 1)
        cls = {"RewardCF": "C0", "EMCF": "C0"}                       # ours = blue; structured = orange; sf = gray
        struct = {"PTF", "ESTR", "BPMF", "SoftImpute", "MFSGD"}
        plt.figure(figsize=(7.2, 3.8))
        for bi, r in enumerate(rhos):
            mu = [np.mean(sk[str(r)][nm]) for nm in methods]
            sd = [np.std(sk[str(r)][nm]) for nm in methods]
            plt.bar(x + (bi - (len(rhos) - 1) / 2.0) * wbar, mu, wbar, yerr=sd, capsize=2,
                    label="$\\rho$=%.2f" % r, alpha=0.6 + 0.4 * bi)
        cols = ["C0" if nm in ("RewardCF", "EMCF") else ("C1" if nm in struct else "C7") for nm in methods]
        DISPLAY = {"RewardCF": "SwarmCF", "EMCF": "SwarmCF-B"}
        plt.xticks(x, [DISPLAY.get(nm, nm) for nm in methods], rotation=30, ha="right", fontsize=7.5)
        for tl, c in zip(plt.gca().get_xticklabels(), cols):
            tl.set_color(c)
        plt.ylabel("servicing skill  (0=random dispatch, 1=oracle)")
        plt.axhline(0, color="black", lw=0.6)
        plt.title("Operational target-servicing mission: our online CF (blue) vs the low-rank field (orange,\n"
                  "incl. our batch hybrid) vs structure-free (gray); separation opens at $\\rho$=0.25", fontsize=8)
        plt.legend(fontsize=8); plt.tight_layout(); save_fig("F17_mission")
        print("F17_mission  <-", os.path.basename(f))

# ---- F18: (a) collaboration value (vs rho, incl isolated) + (b) positive scaling with swarm size m ----
fc = latest("results/pilots/collab_*.json")
fm = latest("results/pilots/scale_m_*.json")
if fc and fm:
    dc = json.load(open(fc)); rawc = dc["raw"]; rhos = dc["meta"]["rhos"]
    dm = json.load(open(fm)); rawm = dm["raw"]; mssz = dm["meta"]["ms"]
    sty = {"RewardCF": dict(color="C0", marker="o", lw=2, label="RewardCF (low-rank CF, ours)"),
           "PTF":      dict(color="C1", marker="s", lw=2, label="PTF (batch-refit low-rank)"),
           "UCBIndep": dict(color="C7", marker="d", ls="--", label="UCBIndep (structure-free)"),
           "Tabular":  dict(color="C3", marker="x", ls=":", label="Tabular (no structure)")}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]                                                 # (a) unseen skill vs broadcast rate rho
    for nm in ["RewardCF", "PTF", "UCBIndep", "Tabular"]:
        if nm not in rawc[str(rhos[0])]:
            continue
        mu = [np.mean(rawc[str(r)][nm]["unseen"]) for r in rhos]
        sd = [np.std(rawc[str(r)][nm]["unseen"]) for r in rhos]
        ax.errorbar(rhos, mu, yerr=sd, capsize=2, markersize=5, **sty[nm])
    ax.axhline(0, color="black", lw=0.7, ls=":")
    ax.set_xlabel("broadcast rate $\\rho$  (0 = isolated: each drone sees only its own outcomes)")
    ax.set_ylabel("unseen-pair skill")
    ax.set_title("(a) Collaboration value: the comms-free broadcast unlocks\n"
                 "generalization for low-rank CF, is worthless to structure-free", fontsize=8)
    ax.grid(alpha=0.25); ax.legend(fontsize=7)
    ax = axes[1]                                                 # (b) unseen skill vs swarm size m
    for nm in ["RewardCF", "UCBIndep", "Tabular"]:
        if nm not in rawm[str(mssz[0])]:
            continue
        mu = [np.mean(rawm[str(mm)][nm]["unseen"]) for mm in mssz]
        sd = [np.std(rawm[str(mm)][nm]["unseen"]) for mm in mssz]
        ax.errorbar(mssz, mu, yerr=sd, capsize=2, markersize=5, **sty[nm])
    ax.axhline(0, color="black", lw=0.7, ls=":")
    ax.set_xscale("log", base=2); ax.set_xticks(mssz); ax.set_xticklabels([str(mm) for mm in mssz])
    ax.set_xlabel("swarm size $m$  (fixed $n$, horizon $T$, broadcast rate $\\rho$)")
    ax.set_ylabel("unseen-pair skill")
    ax.set_title("(b) Positive scaling: the swarm gets SMARTER as it grows;\n"
                 "more drones feed the SHARED structure; structure-free stays flat", fontsize=8)
    ax.grid(alpha=0.25); ax.legend(fontsize=7)
    fig.tight_layout(); save_fig("F18_collab_scaling")
    print("F18_collab_scaling  <-", os.path.basename(fc), "+", os.path.basename(fm))

# ---- F19: operational metrics (re-analysis): time-to-competence, cumulative regret,
#          new-asset readiness latency, resilience to attrition ----
f = latest("results/pilots/opmetrics_*.json")
if f:
    from matplotlib.patches import Patch
    M = json.load(open(f))["raw"]
    a = M["anytime"]; rd = M["readiness"]; rs = M["resilience"]; T = a["T"]
    OURS = {"RewardCF", "BothCF", "HybridCF", "EMCF", "ActiveCFconv"}
    LOWRANK = {"PTF", "BPMF", "SoftImpute", "MFSGD", "KNNCF", "ESTR"}   # low-rank field (incl our batch hybrid PTF)

    def mcol(nm):
        return "C0" if nm in OURS else ("C1" if nm in LOWRANK else "C7")

    def xlab(nm):
        return nm
    order_ab = [m for m in ["RewardCF", "BothCF", "PTF", "BPMF", "MFSGD", "ESTR", "Tabular", "UCBIndep", "Random"]
                if m in a["methods"]]
    xs = np.arange(len(order_ab))
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.4))
    # (a) time-to-competence: rounds to 25% of oracle (partial broadcast rho=0.25)
    A = ax[0, 0]
    for xi, nm in zip(xs, order_ab):
        c = a["ttc"]["0.25"][nm]["0.25"]; reached = c["rounds_mean"] and c["frac_reached"] >= 0.5
        A.bar(xi, c["rounds_mean"] if reached else T, color=mcol(nm), alpha=0.85,
              hatch=None if reached else "//", edgecolor=(None if reached else "black"))
        if not reached:
            A.text(xi, T * 0.5, "never", ha="center", va="center", fontsize=6.5)
    A.set_xticks(xs); A.set_xticklabels([xlab(nm) for nm in order_ab], rotation=30, ha="right", fontsize=7)
    A.set_ylabel("rounds to reach 25% of oracle dispatch")
    A.set_title("(a) Time-to-competence (partial broadcast): ours reach mission-\ncompetence fastest; most competitors never reach it", fontsize=9)
    A.grid(alpha=0.2, axis="y")
    # (b) cumulative regret (lost mission value), lower is better
    B = ax[0, 1]
    mu = [a["regret"]["0.25"][nm]["mean"] for nm in order_ab]; sd = [a["regret"]["0.25"][nm]["sd"] for nm in order_ab]
    B.bar(xs, mu, yerr=sd, capsize=2, color=[mcol(nm) for nm in order_ab], alpha=0.85)
    B.set_xticks(xs); B.set_xticklabels([xlab(nm) for nm in order_ab], rotation=30, ha="right", fontsize=7)
    B.set_ylabel("cumulative regret over the run (lower = better)")
    B.set_title("(b) Cumulative lost mission value = sum of (oracle - earned):\nours leave the least value on the table", fontsize=9)
    B.grid(alpha=0.2, axis="y")
    # (c) new-asset readiness latency (full broadcast): skill on unseen vs own engagements
    C = ax[1, 0]; probes = rd["probes"]
    for who, col, mk, labl, lw, ls in [("cf", "C0", "o", "new drone via CF fold-in", 2.2, "-"),
                                       ("pop", "C5", "d", "popularity prior", 1.5, "--"),
                                       ("tab", "C7", "x", "structure-free newcomer", 1.5, "--")]:
        C.plot(probes, rd["lat"]["1.0"][who]["curve"], marker=mk, color=col, label=labl, lw=lw, ls=ls)
    C.axhline(0, color="black", lw=0.6, ls=":")
    C.set_xlabel("own engagements since joining the swarm"); C.set_ylabel("skill on never-tried targets")
    C.set_title("(c) New-asset readiness: a new drone becomes effective on unseen\ntargets after a few engagements; a structure-free newcomer never does", fontsize=9)
    C.grid(alpha=0.25); C.legend(fontsize=7.5)
    # (d) resilience to attrition: skill retained under continuous turnover
    D = ax[1, 1]; rmeth = rs["methods"]; xr = np.arange(len(rmeth)); w = 0.38
    act = [rs["res"][nm]["active_mean"] for nm in rmeth]; rec = [rs["res"][nm]["recent_mean"] for nm in rmeth]
    D.bar(xr - w / 2, act, w, color=[mcol(nm) for nm in rmeth], alpha=0.95)
    D.bar(xr + w / 2, rec, w, color=[mcol(nm) for nm in rmeth], alpha=0.95, hatch="//", edgecolor="white")
    D.set_xticks(xr); D.set_xticklabels(rmeth, rotation=30, ha="right", fontsize=7)
    D.set_ylabel("skill retained under turnover")
    D.set_title("(d) Resilience to attrition: confidence-directed CF keeps skill on the\nactive set AND recent arrivals; structure-free collapses on newcomers", fontsize=9)
    D.grid(alpha=0.2, axis="y")
    D.legend(handles=[Patch(facecolor="gray", label="active set"),
                      Patch(facecolor="gray", hatch="//", edgecolor="white", label="recent arrivals")], fontsize=7.5)
    fig.tight_layout(); save_fig("F19_operational")
    print("F19_operational  <-", os.path.basename(f))

print("figures written to", OUT)
