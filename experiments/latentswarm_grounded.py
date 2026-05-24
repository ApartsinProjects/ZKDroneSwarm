"""X2: a ROBOTICS-GROUNDED ZK-MRTA instance (addresses the RAS reviewer's "synthetic-only" concern).

Instead of abstract Gaussian-mixture traits and an abstract Bernoulli(rho) mask, this instantiates the
Section 3 model physically:
  - Traits are d NON-NEGATIVE sensing MODALITIES (EO / IR / acoustic / LiDAR / range-endurance), in the
    STRATA / Prorok trait-aggregation style: a robot's capability p_i and a site's requirement u_j are
    profiles over those modalities, so reward R_ij=<p_i,u_j> is the modality MATCH (rank<=d, personalized).
  - The observation mask is a PERSISTENT, range-limited LINE-OF-SIGHT disk graph induced by 2-D patrol
    positions (sectorized clusters -> quasi-static visibility), and the per-observer noise GROWS WITH
    DISTANCE (free-space SNR ~ 1/r^2; Cortes et al. coverage law). Positions are no longer inert.
  - Capacity-1 contention as in the body; guessed rank drawn at random per run.

Shows the categorical low-rank-vs-structure-free separation survives a physically-motivated instance.
Reuses latentswarm.run.run verbatim. Run from anywhere.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.stdout.reconfigure(encoding="utf-8")

from latentswarm import RunConfig
from latentswarm.run import run
from _results_io import save_results


def main():
    cfg = RunConfig(
        scenario="sensing_coalition",      # non-negative sensing-modality traits (STRATA-style)
        n_modes=5,                          # one specialist archetype per modality (d=5)
        mask_mode="line_of_sight",          # geometry-induced persistent disk-graph mask
        rho=0.5,                            # target line-of-sight density (sets the sensing radius R_s)
        noise_alpha=2.0,                    # per-observer noise variance ~ sigma_obs^2 (1 + (r/R_s)^2)
        capacity_one=True,                  # capacity-1 contention, as in the body
        seeds=list(range(16)),
        algorithms=["random", "ucb_indep", "mf_sgd", "swarm_cf"],
    )
    out = run(cfg)
    out["meta"]["experiment"] = ("X2 robotics-grounded instance: STRATA sensing-modality traits + "
                                 "line-of-sight geometry mask + distance-dependent noise")
    save_results("latentswarm_grounded", out, results_dir=os.path.join(ROOT, "results", "pilots"))
    print("saved -> results/pilots/latentswarm_grounded_<stamp>.json")


if __name__ == "__main__":
    main()
