"""Generate Figure 12: 70-episode broadcast vs no-broadcast trajectories."""
import json, os, statistics, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.stdout.reconfigure(encoding="utf-8")


def load(pattern):
    return [json.load(open(f)) for f in sorted(glob.glob(pattern))]


def ep_mean_se(runs, field):
    n_ep = len(runs[0]["episodes"])
    means, ses = [], []
    for e in range(n_ep):
        vals = [r["episodes"][e][field] for r in runs]
        means.append(statistics.mean(vals))
        ses.append(statistics.stdev(vals) / len(vals) ** 0.5 if len(vals) > 1 else 0)
    return np.array(means), np.array(ses)


nb_dir = "results/new_benchmarks"

bcast = load(os.path.join(nb_dir, "long70_bcast_mf_cf_s*.json"))
nbcast = load(os.path.join(nb_dir, "long70_nbcast_mf_cf_nb_s*.json"))

T = len(bcast[0]["episodes"])
print(f"Loaded {len(bcast)} broadcast runs and {len(nbcast)} no-broadcast runs over {T} episodes")

eps = np.arange(1, T + 1)
bm, bse = ep_mean_se(bcast, "steps")
nm, nse = ep_mean_se(nbcast, "steps")
blm, blse = ep_mean_se(bcast, "avg_latent_match_quality")
nlm, nlse = ep_mean_se(nbcast, "avg_latent_match_quality")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(eps, bm, "-", color="#2ca02c", lw=2.0, label="MF-CF (broadcast)")
ax1.fill_between(eps, bm - bse, bm + bse, color="#2ca02c", alpha=0.13)
ax1.plot(eps, nm, "-", color="#d65f5f", lw=1.6, label="MF-CF (no broadcast)")
ax1.fill_between(eps, nm - nse, nm + nse, color="#d65f5f", alpha=0.13)
ax1.axvline(8, color="black", lw=0.7, ls=":", alpha=0.5)
ax1.axvline(72, color="grey", lw=0.7, ls="--", alpha=0.5)
ax1.text(72 + 0.5, ax1.get_ylim()[1] * 0.95, "T6: $K^{\\mathrm{nb}}_{\\mathrm{cold}}\\approx 72$",
         fontsize=9, alpha=0.7, va="top")
ax1.text(8 + 0.5, ax1.get_ylim()[1] * 0.95, "T6: $K^{\\mathrm{bcast}}_{\\mathrm{cold}}\\approx 8$",
         fontsize=9, alpha=0.7, va="top")
ax1.set_xlabel("Episode", fontsize=11)
ax1.set_ylabel("Steps to completion", fontsize=11)
ax1.set_title("Figure 12a — Long-horizon (70 ep) broadcast vs no-broadcast", fontsize=11)
ax1.legend(loc="upper right", fontsize=10)
ax1.set_xlim(1, T)
ax1.grid(True, alpha=0.3)

ax2.plot(eps, blm, "-", color="#2ca02c", lw=2.0, label="MF-CF (broadcast)")
ax2.fill_between(eps, blm - blse, blm + blse, color="#2ca02c", alpha=0.13)
ax2.plot(eps, nlm, "-", color="#d65f5f", lw=1.6, label="MF-CF (no broadcast)")
ax2.fill_between(eps, nlm - nlse, nlm + nlse, color="#d65f5f", alpha=0.13)
ax2.axvline(8, color="black", lw=0.7, ls=":", alpha=0.5)
ax2.axvline(72, color="grey", lw=0.7, ls="--", alpha=0.5)
ax2.set_xlabel("Episode", fontsize=11)
ax2.set_ylabel("Latent Match Quality", fontsize=11)
ax2.set_title("Figure 12b — LMQ trajectories", fontsize=11)
ax2.legend(loc="lower right", fontsize=10)
ax2.set_xlim(1, T)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs("docs/academic-paper/figures", exist_ok=True)
plt.savefig("docs/academic-paper/figures/fig12-long-horizon.svg", format="svg", bbox_inches="tight")
plt.savefig("docs/academic-paper/figures/fig12-long-horizon.png", dpi=150, bbox_inches="tight")
print("Saved Figure 12")
plt.close()

# Also print summary statistics
print()
print("Summary:")
print(f"  Broadcast MF-CF: avg = {statistics.mean(bm[34:]):.1f} steps eps 35-end; LMQ end = {blm[-1]:.4f}")
print(f"  No-bcast MF-CF:  avg = {statistics.mean(nm[34:]):.1f} steps eps 35-end; LMQ end = {nlm[-1]:.4f}")
print(f"  Slowdown ratio at episode 35: {nm[34]/bm[34]:.2f}x; at episode 70: {nm[-1]/bm[-1]:.2f}x")
