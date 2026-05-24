"""Config-driven sweep drivers that reproduce the paper's headline analytical sweeps from the
LatentSwarm package, and emit the SAME JSON schema that experiments/make_figures.py already reads.

One mission "cell" is run through the genuine package path (RunConfig -> build_scenario -> ZKMRTAEnv
-> Policy -> metrics): the uniform-world scenario `uniform_cosine` (i.i.d. unit-sphere traits, the
no-types cosine world matching experiments/core.make_world(model="uniform")), the env's persistent
per-pair mask M_ik ~ Bernoulli(rho) (diag True), per-observer noise (own ~ N(0, sigma_own^2),
broadcast ~ N(0, sigma_obs^2)), a fresh per-round random size-`cand` menu, and the held-out unseen /
overall / state-uniqueness / anytime metrics from metrics.py. The guessed rank is drawn d_hat ~
Uniform{d..2d} per seed via RandomState(9000+seed), so every method at a seed shares the same d_hat.

Every comparison includes CLUB (clustering-of-bandits) and BiasModel (additive popularity) so the
discrete-clustering and additive controls appear in every figure and table.

Cells are independent and run in a PROCESS POOL (one mission per worker), so a full 16-seed sweep is
parallel across CPU cores.

Figure-schema compatibility: the per-figure JSON uses the ANALYTICAL method KEYS (RewardCF, PTF,
ESTR, BPMF, UCBIndep, Tabular, CLUB, BiasModel, ...) that make_figures.py / method_profiles.py look
up, mapped from the package algorithm names below. Sweeps emit exactly the keys each reader expects:

    bakeoff        -> c14_compare schema    (Table 3):     raw[rho][KEY] = {overall, unseen, uniq}
    crossover      -> c15_crossover schema  (F5/F22/F28):  raw[rho][KEY] = {overall, unseen, uniq}
    anytime        -> c16_anytime schema    (F6/F22/F28):  raw[rho][KEY] = [traj per seed]
    collab         -> collab schema         (F18a):        raw[rho][KEY] = {overall, unseen}
    scale_m        -> scale_m schema        (F18b):        raw[m][KEY]   = {overall, unseen}
    ranksweep      -> e246-style dhat sweep schema:        raw["dhat"][v][KEY] = {unseen, anytime}
    offersize      -> crossover+anytime at cand=20 and cand=n (F22/F28)
    iid_vs_persistent -> e12_iid_masking schema (F8):      rawA[mode][rho][KEY]={unseen,anytime,uniq}, rawB[mode][T]

IMPORTANT: this module writes ONLY to results/smoke/ by default (the development/validation sink);
results/pilots/ is the paper's figure-pipeline input and must not be polluted by smoke runs. Point
--out elsewhere only for a real (non-smoke) sweep.

CLI:  python -m latentswarm.sweeps --which crossover --out results/pilots [--smoke] [--workers N]
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from .config import RunConfig
from .registry import ALGORITHMS, METRICS, get
from .scenarios import build_scenario
from .env import ZKMRTAEnv

# package algorithm name -> analytical figure-JSON key (so make_figures.py / method_profiles.py find the series)
PKG2KEY = {
    "swarm_cf": "RewardCF",         # ours, online weighted-ALS
    "swarmcf_batch": "PTF",         # ours, batch warm-start + finetune (SwarmCF-batch)
    "estr": "ESTR",
    "bpmf": "BPMF",
    "mf_sgd": "MFSGD",
    "ucb_indep": "UCBIndep",
    "ucb_homo": "UCBHomo",
    "tabular": "Tabular",
    "club": "CLUB",                 # clustering-of-bandits (discrete-clustering control)
    "knn_cf": "KNNCF",              # memory-based user-user CF (model-free control)
    "soft_impute": "SoftImpute",    # convex nuclear-norm completion (convex-completion control)
    "bias_model": "BiasModel",      # additive popularity (rank<=2 control)
    "random": "Random",
    # SwarmCF-* refinement family (follow-up paper); keys match method_profiles.FAMILY code names
    "em_cf": "EMCF",
    "ard_em_cf": "ARD-EMCF",
    "active_cf": "ActiveCF",
    "coord_cf": "CoordCF",
    "contention_cf": "ContentionCF",
    "contention_ada_cf": "ContentionAdaCF",
    "choice_cf": "ChoiceCF",
    "both_cf": "BothCF",
    "unified_cf": "UnifiedCF",
}


def guessed_rank(cfg, seed):
    """Analytical-harness alignment: d_hat ~ Uniform{d .. 2d} drawn once per seed from a dedicated
    RandomState(9000+seed) (matches experiments/pilot_compare.guessed_rank), so all methods at a seed
    share d_hat. Honors a fixed cfg.rank_guess int if the user pins one."""
    if isinstance(cfg.rank_guess, int):
        return int(cfg.rank_guess)
    return int(np.random.RandomState(9000 + seed).randint(cfg.d, 2 * cfg.d + 1))


def _cfg_for(base: RunConfig, **over) -> RunConfig:
    d = base.to_dict(); d.update(over); return RunConfig.from_dict(d)


def run_cell(cfg: RunConfig, algo: str, seed: int, want_traj=False):
    """Run ONE mission of `algo` on a fresh world (seed) through the package path, and return its
    metrics. Mirrors pilot_c11_masking.run_masked / pilot_anytime semantics via the env.

    Returns dict: overall, unseen (held-out), uniq (state-uniqueness, nan if structure-free),
    and (if want_traj) real/oracle/random per-round SUM arrays for the anytime/regret/ttc metrics.
    """
    world_rng = np.random.RandomState(seed)
    P, U = build_scenario(cfg, world_rng).generate()
    d_guess = guessed_rank(cfg, seed)
    env = ZKMRTAEnv(cfg, P, U, d_guess, seed=seed)
    pol = get(ALGORITHMS, algo)(cfg, cfg.m, cfg.n, d_guess, seed=seed + 7)
    obs = env.reset()
    T = cfg.T
    real = np.zeros(T); orac = np.zeros(T); rnd = np.zeros(T)
    for t in range(T):
        actions = pol.act(obs)
        # per-round anytime accounting on the OFFERED menus (best-in-offer oracle, mean-in-offer floor)
        for i in range(cfg.m):
            off = np.where(obs[i]["offer"])[0]
            a = int(actions[i])
            if a >= 0:
                real[t] += float(env.R[i, a])
            if off.size:
                orac[t] += float(env.R[i, off].max()); rnd[t] += float(env.R[i, off].mean())
        obs, rewards, _ = env.step(actions)
        pol.observe(obs)
    pred = pol.predict_rows()
    engaged = env.engaged

    # held-out eval (fresh menus from never-engaged tasks + over all tasks), analytical protocol.
    overall = _overall_skill(env.R, pred, np.random.RandomState(seed + 555), cfg.offer_size)
    unseen = get(METRICS, "unseen_pair_skill_heldout")().compute(
        P=P, U=U, pred_rows=pred, engaged=engaged, rng=np.random.RandomState(seed + 556), offer_size=cfg.offer_size)
    uniq = get(METRICS, "state_uniqueness")().compute(policy=pol)
    out = {"overall": float(overall) if overall is not None else None,
           "unseen": float(unseen) if unseen is not None else None,
           "uniq": float(uniq)}
    if want_traj:
        out["real"] = real.tolist(); out["oracle"] = orac.tolist(); out["random"] = rnd.tolist()
    return out


def _overall_skill(R, pred, g, offer_size, reps=120):
    """Overall held-out skill over ALL tasks (run_masked's 'overall'): fresh size-min(c,20) menus from
    the full task set, greedy pick under the model, normalized to best-in-menu / mean-in-menu."""
    m, n = R.shape
    c = n if (not offer_size or offer_size <= 0) else int(offer_size)
    ev = min(c, 20)
    gg, oo, rr = [], [], []
    for _ in range(reps):
        for k in range(m):
            off = g.choice(n, size=ev, replace=False)
            if pred is not None:
                pick = off[int(np.argmax(pred[k, off]))]
            else:
                pick = off[int(g.randint(ev))]
            gg.append(R[k, pick]); oo.append(R[k, off].max()); rr.append(R[k, off].mean())
    gm, om, rm = np.mean(gg), np.mean(oo), np.mean(rr)
    return (gm - rm) / max(om - rm, 1e-6)


# ---------------------------------------------------------------------------------------------
# parallel cell runner: cells are independent (deterministic given cfg, algo, seed) -> process pool
# ---------------------------------------------------------------------------------------------
def _cell_job(args):
    cfg_dict, algo, seed, want_traj = args
    return run_cell(RunConfig.from_dict(cfg_dict), algo, seed, want_traj=want_traj)


def _workers(n=None):
    return n if n else max(1, min(6, (os.cpu_count() or 2)))


def _pmap(jobs, want_traj=False, workers=None):
    """jobs: list of (tag, cfg, algo, seed). Runs each cell in a process pool and returns a list of
    (tag, algo, seed, result) in the SAME ORDER as `jobs` (so per-(tag,algo) seed order is preserved)."""
    packed = [(cfg.to_dict(), algo, seed, want_traj) for (_, cfg, algo, seed) in jobs]
    results = [None] * len(jobs)
    w = _workers(workers)
    if w <= 1:
        for i, p in enumerate(packed):
            results[i] = _cell_job(p)
    else:
        with ProcessPoolExecutor(max_workers=w) as ex:
            futs = {ex.submit(_cell_job, packed[i]): i for i in range(len(packed))}
            for fut in as_completed(futs):
                results[futs[fut]] = fut.result()
    return [(jobs[i][0], jobs[i][2], jobs[i][3], results[i]) for i in range(len(jobs))]


# ---------------------------------------------------------------------------------------------
# defaults matching the headline regime (uniform_cosine world, persistent mask)
# ---------------------------------------------------------------------------------------------
def base_config(smoke=False, **over) -> RunConfig:
    """The headline regime as a RunConfig (uniform-world). smoke=True shrinks m/n/T/seeds for a quick
    validity check (NOT a paper run)."""
    if smoke:
        d = dict(m=8, n=40, d=5, T=10, n_types=5, offer_size=20,
                 scenario="uniform_cosine", jitter=0.15, mask_mode="persistent",
                 rho=0.5, sigma_own=0.10, sigma_obs=0.30,
                 als_sweeps=8, refit_every=3, epsilon=0.5, epsilon_decay=0.93, epsilon_min=0.05,
                 seeds=[0, 1])
    else:
        d = dict(m=30, n=240, d=5, T=50, n_types=10, offer_size=20,
                 scenario="uniform_cosine", jitter=0.15, mask_mode="persistent",
                 rho=0.5, sigma_own=0.10, sigma_obs=0.30,
                 als_sweeps=8, refit_every=3, epsilon=0.5, epsilon_decay=0.93, epsilon_min=0.05,
                 seeds=list(range(16)))
    d.update(over)
    return RunConfig(**d)


# canonical comparison sets -- CLUB and BiasModel are in every figure/table (the discrete-clustering
# and additive-popularity controls); RewardCF (=SwarmCF) is ours.
_BAKEOFF_METHODS = ["random", "ucb_indep", "ucb_homo", "tabular", "mf_sgd", "estr",
                    "swarmcf_batch", "bpmf", "club", "knn_cf", "soft_impute", "bias_model", "swarm_cf"]
_CROSSOVER_METHODS = ["ucb_indep", "tabular", "mf_sgd", "estr", "bpmf", "club", "knn_cf", "soft_impute",
                      "bias_model", "swarmcf_batch", "swarm_cf"]
_ANYTIME_METHODS = ["random", "ucb_indep", "tabular", "mf_sgd", "estr", "bpmf", "club",
                    "knn_cf", "soft_impute", "bias_model", "swarmcf_batch", "swarm_cf"]
_COLLAB_METHODS = ["swarm_cf", "swarmcf_batch", "club", "knn_cf", "soft_impute", "bias_model", "ucb_indep", "tabular"]
_RHOS = [1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.15, 0.1]


def sweep_bakeoff(cfg: RunConfig, methods=None, rhos=None):
    """Method bake-off at the three headline rhos (c14_compare schema, Table 3): every method's
    overall/unseen/uniq at rho in {1.0, 0.5, 0.25}. Feeds method_profiles.html_scorecard."""
    methods = methods or _BAKEOFF_METHODS
    rhos = rhos or [1.0, 0.5, 0.25]
    raw = {str(r): {PKG2KEY[a]: {"overall": [], "unseen": [], "uniq": []} for a in methods} for r in rhos}
    jobs = [(str(r), _cfg_for(cfg, rho=r), a, s) for r in rhos for a in methods for s in cfg.seeds]
    for tag, a, s, c in _pmap(jobs):
        cell = raw[tag][PKG2KEY[a]]
        cell["overall"].append(c["overall"]); cell["unseen"].append(c["unseen"]); cell["uniq"].append(c["uniq"])
    meta = _meta(cfg, experiment="C14 method bake-off (uniform world): overall/unseen/uniq vs rho",
                 methods=[PKG2KEY[a] for a in methods], rhos=rhos, cand=cfg.offer_size,
                 group={PKG2KEY[a]: _GROUP.get(a, "") for a in methods})
    return {"meta": meta, "raw": raw}, "c14_compare"


def sweep_crossover(cfg: RunConfig, methods=None, rhos=None):
    """Masking-robustness crossover: unseen/overall/uniq vs rho (c15_crossover schema, F5/F22/F28)."""
    methods = methods or _CROSSOVER_METHODS
    rhos = rhos or _RHOS
    raw = {str(r): {PKG2KEY[a]: {"overall": [], "unseen": [], "uniq": []} for a in methods} for r in rhos}
    jobs = [(str(r), _cfg_for(cfg, rho=r), a, s) for r in rhos for a in methods for s in cfg.seeds]
    for tag, a, s, c in _pmap(jobs):
        cell = raw[tag][PKG2KEY[a]]
        cell["overall"].append(c["overall"]); cell["unseen"].append(c["unseen"]); cell["uniq"].append(c["uniq"])
    meta = _meta(cfg, experiment="crossover (masking-robustness): unseen/overall vs rho",
                 methods=[PKG2KEY[a] for a in methods], rhos=rhos, cand=cfg.offer_size)
    return {"meta": meta, "raw": raw}, "c15_crossover"


def sweep_anytime(cfg: RunConfig, methods=None, rhos=None):
    """Anytime cumulative-reward trajectories per rho (c16_anytime schema, F6/F22/F28). Includes all
    bake-off methods so opmetrics can derive regret/ttc for every Table 3 row."""
    methods = methods or _ANYTIME_METHODS
    rhos = rhos if rhos is not None else [1.0, 0.25]
    raw = {str(r): {PKG2KEY[a]: [] for a in methods} for r in rhos}
    at = get(METRICS, "anytime_trajectory")()
    jobs = [(str(r), _cfg_for(cfg, rho=r), a, s) for r in rhos for a in methods for s in cfg.seeds]
    for tag, a, s, c in _pmap(jobs, want_traj=True):
        traj = at.compute(real=c["real"], oracle=c["oracle"], random=c["random"])["trajectory"]
        raw[tag][PKG2KEY[a]].append(traj)
    meta = _meta(cfg, experiment="anytime cumulative-reward AUC trajectories",
                 methods=[PKG2KEY[a] for a in methods], rhos=rhos, cand=cfg.offer_size)
    return {"meta": meta, "raw": raw}, "c16_anytime"


def sweep_collab(cfg: RunConfig, methods=None, rhos=None):
    """Collaboration value: unseen/overall vs broadcast rate rho INCLUDING rho=0 (isolated)
    (collab schema, F18a)."""
    methods = methods or _COLLAB_METHODS
    rhos = rhos or [0.0, 0.1, 0.25, 0.5, 1.0]
    raw = {str(r): {PKG2KEY[a]: {"overall": [], "unseen": []} for a in methods} for r in rhos}
    jobs = [(str(r), _cfg_for(cfg, rho=max(r, 1e-9)), a, s) for r in rhos for a in methods for s in cfg.seeds]
    for tag, a, s, c in _pmap(jobs):
        cell = raw[tag][PKG2KEY[a]]
        cell["overall"].append(c["overall"]); cell["unseen"].append(c["unseen"])
    meta = _meta(cfg, experiment="collaboration value: skill vs rho (incl rho=0 isolated)",
                 methods=[PKG2KEY[a] for a in methods], rhos=rhos)
    return {"meta": meta, "raw": raw}, "collab"


def sweep_scale_m(cfg: RunConfig, methods=None, ms=None):
    """Positive scaling with swarm size m at fixed n, T, rho (scale_m schema, F18b)."""
    methods = methods or _COLLAB_METHODS
    ms = ms or ([4, 8] if cfg.m <= 8 else [5, 10, 20, 40, 80])
    raw = {str(mm): {PKG2KEY[a]: {"overall": [], "unseen": []} for a in methods} for mm in ms}
    jobs = [(str(mm), _cfg_for(cfg, m=mm), a, s) for mm in ms for a in methods for s in cfg.seeds]
    for tag, a, s, c in _pmap(jobs):
        cell = raw[tag][PKG2KEY[a]]
        cell["overall"].append(c["overall"]); cell["unseen"].append(c["unseen"])
    meta = _meta(cfg, experiment="positive scaling with swarm size m", methods=[PKG2KEY[a] for a in methods], ms=ms)
    return {"meta": meta, "raw": raw}, "scale_m"


def sweep_ranksweep(cfg: RunConfig, methods=None, dhats=None):
    """Guessed-rank (d_hat) sweep: unseen + anytime vs d_hat (e246-scaling 'dhat' column schema)."""
    methods = methods or ["tabular", "ucb_indep", "club", "swarmcf_batch", "swarm_cf"]
    dhats = dhats or ([5, 8] if cfg.d <= 5 and len(cfg.seeds) <= 2 else [3, 5, 8, 10, 12])
    at = get(METRICS, "anytime_trajectory")()
    raw = {"dhat": {str(v): {PKG2KEY[a]: {"unseen": [], "anytime": []} for a in methods} for v in dhats}}
    jobs = [(str(v), _cfg_for(cfg, rank_guess=int(v)), a, s) for v in dhats for a in methods for s in cfg.seeds]
    for tag, a, s, c in _pmap(jobs, want_traj=True):
        fin = at.compute(real=c["real"], oracle=c["oracle"], random=c["random"])["final"]
        cell = raw["dhat"][tag][PKG2KEY[a]]
        cell["unseen"].append(c["unseen"]); cell["anytime"].append(fin)
    meta = _meta(cfg, experiment="guessed-rank d_hat sweep (unseen + anytime)",
                 methods=[PKG2KEY[a] for a in methods], sweeps={"dhat": dhats}, rho=cfg.rho)
    return {"meta": meta, "raw": raw}, "e246_scaling"


def sweep_offersize(cfg: RunConfig):
    """Offer-size robustness (F22/F28): the crossover + anytime sweeps at the body menu (cand=20) AND
    the all-tasks menu (cand=n). Returns a LIST of (payload, name) so the driver writes 4 files (each
    tagged by meta['cand'], which the figure reader keys on)."""
    outs = []
    for c in (20, cfg.n):
        cc = _cfg_for(cfg, offer_size=(0 if c >= cfg.n else c))   # 0 = all-tasks menu in the env
        px, nx = sweep_crossover(cc)
        px["meta"]["cand"] = c
        outs.append((px, nx))
        pa, na = sweep_anytime(cc)
        pa["meta"]["cand"] = c
        outs.append((pa, na))
    return outs


def sweep_iid_vs_persistent(cfg: RunConfig, methods=None, rhos=None, tgrid=None):
    """Persistent vs i.i.d. masking (e12_iid_masking schema, F8): Part A unseen/anytime/uniq vs rho in
    both modes; Part B state-uniqueness vs horizon T (RewardCF, rho=0.25)."""
    methods = methods or ["tabular", "ucb_indep", "club", "swarmcf_batch", "estr", "swarm_cf"]
    rhos = rhos or [1.0, 0.5, 0.25, 0.1]
    tgrid = tgrid or ([10, 20] if len(cfg.seeds) <= 2 else [25, 50, 100, 200])
    modes = ["persistent", "iid"]
    at = get(METRICS, "anytime_trajectory")()
    rawA = {mode: {str(r): {PKG2KEY[a]: {"unseen": [], "anytime": [], "uniq": []} for a in methods}
                   for r in rhos} for mode in modes}
    rawB = {mode: {str(T): {"uniq": [], "unseen": []} for T in tgrid} for mode in modes}
    mm = {"persistent": "persistent", "iid": "per_round"}        # env mask_mode for each label
    # Part A: tag = "mode|rho"
    jobsA = [("%s|%s" % (mode, r), _cfg_for(cfg, rho=r, mask_mode=mm[mode]), a, s)
             for mode in modes for r in rhos for a in methods for s in cfg.seeds]
    for tag, a, s, c in _pmap(jobsA, want_traj=True):
        mode, r = tag.split("|")
        fin = at.compute(real=c["real"], oracle=c["oracle"], random=c["random"])["final"]
        cell = rawA[mode][r][PKG2KEY[a]]
        cell["unseen"].append(c["unseen"]); cell["anytime"].append(fin); cell["uniq"].append(c["uniq"])
    # Part B: tag = "mode|T", swarm_cf only
    jobsB = [("%s|%s" % (mode, T), _cfg_for(cfg, rho=0.25, T=T, mask_mode=mm[mode]), "swarm_cf", s)
             for mode in modes for T in tgrid for s in cfg.seeds]
    for tag, a, s, c in _pmap(jobsB):
        mode, T = tag.split("|")
        rawB[mode][T]["uniq"].append(c["uniq"]); rawB[mode][T]["unseen"].append(c["unseen"])
    meta = _meta(cfg, experiment="persistent vs iid masking",
                 methods=[PKG2KEY[a] for a in methods], modes=modes, rhos=rhos, Tgrid=tgrid)
    return {"meta": meta, "rawA": rawA, "rawB": rawB}, "e12_iid_masking"


# ---------------------------------------------------------------------------------------------
# approximate-low-rank robustness (Appendix F, F27): unseen skill vs full-rank perturbation eps
# ---------------------------------------------------------------------------------------------
_APPROX_METHODS = ["ucb_indep", "club", "swarmcf_batch", "swarm_cf"]
_APPROX_EPS = [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]


def _eff_rank(R, thresh=0.99):
    sv = np.linalg.svd(R, compute_uv=False)
    cum = np.cumsum(sv ** 2) / max(float(np.sum(sv ** 2)), 1e-12)
    return int(np.searchsorted(cum, thresh) + 1)


def run_approx_cell(cfg: RunConfig, algo: str, seed: int):
    """One approx-low-rank mission (scenario=approx_lowrank): run the mission on R_eps and also report
    the effective rank of the realized reward (the same deterministic world rebuilt is cheap)."""
    out = run_cell(cfg, algo, seed)                       # overall, unseen on R_eps
    P, U = build_scenario(cfg, np.random.RandomState(seed)).generate()
    out["eff_rank"] = _eff_rank(P @ U.T)
    return out


def _approx_job(args):
    cfg_dict, algo, seed = args
    return run_approx_cell(RunConfig.from_dict(cfg_dict), algo, seed)


def sweep_approxrank(cfg: RunConfig, methods=None, epslist=None, workers=None):
    """Approximate low-rank robustness (x5_approxrank schema, F27): unseen + effective rank vs the
    full-rank perturbation strength eps, at the masked headline rho=0.25."""
    methods = methods or _APPROX_METHODS
    epslist = epslist or _APPROX_EPS
    raw = {("%.2f" % e): {PKG2KEY[a]: {"overall": [], "unseen": [], "eff_rank": []} for a in methods}
           for e in epslist}
    jobs = [(("%.2f" % e), _cfg_for(cfg, scenario="approx_lowrank", approx_eps=float(e), rho=0.25), a, s)
            for e in epslist for a in methods for s in cfg.seeds]
    packed = [(c.to_dict(), a, s) for (_, c, a, s) in jobs]
    results = [None] * len(jobs)
    w = _workers(workers)
    if w <= 1:
        for i, pk in enumerate(packed):
            results[i] = _approx_job(pk)
    else:
        with ProcessPoolExecutor(max_workers=w) as ex:
            futs = {ex.submit(_approx_job, packed[i]): i for i in range(len(packed))}
            for fut in as_completed(futs):
                results[futs[fut]] = fut.result()
    for i, (tag, c, a, s) in enumerate(jobs):
        r = results[i]; cell = raw[tag][PKG2KEY[a]]
        cell["overall"].append(r["overall"]); cell["unseen"].append(r["unseen"]); cell["eff_rank"].append(r["eff_rank"])
    meta = _meta(cfg, experiment="approximate low-rank robustness (full-rank perturbation eps)",
                 methods=[PKG2KEY[a] for a in methods], eps=epslist, rho=0.25)
    return {"meta": meta, "raw": raw}, "x5_approxrank"


# ---------------------------------------------------------------------------------------------
# contention sweep (follow-up paper, Section 4 / Figure 2): communication-free de-confliction
# ---------------------------------------------------------------------------------------------
from collections import defaultdict
from .metrics import hungarian_oracle_per_step  # reuse the Hungarian helper for the pool sub-matrix

_CONTENTION_METHODS = ["contention_ada_cf", "contention_cf", "swarm_cf", "active_cf",
                       "swarmcf_batch", "club", "ucb_indep", "random"]
_POOLS = [240, 60, 30, 15]

# group label for the bake-off meta (matches experiments/pilot_compare.GROUP)
_GROUP = {"random": "no-struct", "ucb_indep": "no-struct", "ucb_homo": "no-struct",
          "tabular": "no-struct", "mf_sgd": "low-rank", "estr": "low-rank",
          "swarmcf_batch": "low-rank", "bpmf": "low-rank", "bias_model": "additive",
          "club": "clustering", "swarm_cf": "low-rank(ours)"}


def _match_opt(Rsub):
    """Max-sum capacity-1 matching of distinct pool tasks to robots (the centralized ceiling)."""
    from scipy.optimize import linear_sum_assignment
    ri, ci = linear_sum_assignment(-Rsub)
    return float(Rsub[ri, ci].sum())


def _rand_earned(Rsub, rng, draws=30):
    """Expected earned reward of random picks with random collision resolution (the floor)."""
    m, p = Rsub.shape
    tot = 0.0
    for _ in range(draws):
        picks = rng.randint(0, p, m)
        by = defaultdict(list)
        for i, a in enumerate(picks):
            by[a].append(i)
        s = 0.0
        for a, cont in by.items():
            w = cont[rng.randint(len(cont))]
            s += Rsub[w, a]
        tot += s
    return tot / draws


def run_contention_cell(cfg: RunConfig, algo: str, pool: int, seed: int):
    """One contention mission of `algo` on a fresh world: a SHARED size-`pool` offer is posted each
    round, every robot picks one task, each contested task is awarded to one uniformly random
    contender (losers earn 0 and produce NO public engagement), and the broadcast carries winners'
    (action, outcome) only with masking rho on top. Faithful port of
    experiments/pilot_contention.run_contention. Returns (earned_skill, unseen_skill, collision_rate)
    where earned_skill is matching-normalized (Hungarian ceiling, random floor)."""
    world_rng = np.random.RandomState(seed)
    P, U = build_scenario(cfg, world_rng).generate()
    R = P @ U.T
    m, n = R.shape
    d_guess = guessed_rank(cfg, seed)
    rng = np.random.RandomState(seed + 999)
    Mask = rng.rand(m, m) < cfg.rho
    np.fill_diagonal(Mask, True)
    pol = get(ALGORITHMS, algo)(cfg, m, n, d_guess, seed=seed + 7)

    cum_real = cum_orac = cum_rand = 0.0
    collisions = 0; engagements = 0
    last_sel = np.full(m, -1, dtype=int)
    last_rew = np.zeros(m)
    engaged = [set() for _ in range(m)]

    def _obs(offer_idx, won):
        offer = np.zeros(n, dtype=bool); offer[offer_idx] = True
        out = []
        for i in range(m):
            sel = np.full(m, -1, dtype=int); rew = np.zeros(m)
            for k in range(m):
                if last_sel[k] == -1 or not won[k] or not Mask[i, k]:
                    continue
                sel[k] = last_sel[k]
                sig = cfg.sigma_own if k == i else cfg.sigma_obs
                rew[k] = last_rew[k] + (rng.normal(0.0, sig) if sig > 0 else 0.0)
            out.append({"offer": offer, "sel": sel, "rew": rew, "i": i})
        return out

    S = rng.choice(n, size=pool, replace=False)
    obs = _obs(S, np.zeros(m, bool))
    for t in range(cfg.T):
        actions = pol.act(obs)
        picks = np.array([int(actions[i]) if int(actions[i]) >= 0 else int(rng.choice(S)) for i in range(m)])
        for i in range(m):
            engaged[i].add(int(picks[i]))
        by = defaultdict(list)
        for i, aidx in enumerate(picks):
            by[aidx].append(i)
        won = np.zeros(m, bool)
        for aidx, cont in by.items():
            won[cont[rng.randint(len(cont))]] = True
            if len(cont) > 1:
                collisions += len(cont) - 1
            engagements += len(cont)
        true_r = np.array([R[i, picks[i]] for i in range(m)])
        cum_real += float(np.where(won, true_r, 0.0).sum())
        Rsub = R[:, S]
        cum_orac += _match_opt(Rsub)
        cum_rand += _rand_earned(Rsub, np.random.RandomState(seed * 131 + t))
        last_sel = picks.copy()
        last_rew = np.where(won, true_r, 0.0)
        if hasattr(pol, "set_lost"):
            pol.set_lost([0.0 if won[i] else 1.0 for i in range(m)])
        S = rng.choice(n, size=pool, replace=False)
        obs = _obs(S, won)
        pol.observe(obs)

    anytime = (cum_real - cum_rand) / max(cum_orac - cum_rand, 1e-9)
    pred = pol.predict_rows()
    unseen = get(METRICS, "unseen_pair_skill_heldout")().compute(
        P=P, U=U, pred_rows=pred, engaged=engaged, rng=np.random.RandomState(seed + 555),
        offer_size=cfg.offer_size)
    coll_rate = collisions / max(engagements, 1)
    return float(anytime), (float(unseen) if unseen is not None else None), float(coll_rate)


def _contention_job(args):
    cfg_dict, algo, pool, seed = args
    return run_contention_cell(RunConfig.from_dict(cfg_dict), algo, pool, seed)


def sweep_contention(cfg: RunConfig, methods=None, pools=None, workers=None):
    """De-confliction under capacity-1 contention (follow-up Figure 2 / Section 4): earned reward
    (matching-normalized), contention-free unseen skill, and collision rate vs the shared pool size
    (smaller = more contention). Faithful port of experiments/pilot_contention.main."""
    methods = methods or _CONTENTION_METHODS
    pools = pools or _POOLS
    raw = {str(p): {PKG2KEY[a]: {"anytime": [], "unseen": [], "coll": []} for a in methods} for p in pools}
    jobs = [(p, a, s) for p in pools for a in methods for s in cfg.seeds]
    packed = [(cfg.to_dict(), a, p, s) for (p, a, s) in jobs]
    results = [None] * len(jobs)
    w = _workers(workers)
    if w <= 1:
        for i, pk in enumerate(packed):
            results[i] = _contention_job(pk)
    else:
        with ProcessPoolExecutor(max_workers=w) as ex:
            futs = {ex.submit(_contention_job, packed[i]): i for i in range(len(packed))}
            for fut in as_completed(futs):
                results[futs[fut]] = fut.result()
    for i, (p, a, s) in enumerate(jobs):
        an, un, co = results[i]
        cell = raw[str(p)][PKG2KEY[a]]
        cell["anytime"].append(an); cell["unseen"].append(un); cell["coll"].append(co)
    meta = _meta(cfg, experiment="contention capacity-1 matching (de-confliction)",
                 methods=[PKG2KEY[a] for a in methods], pools=pools, rho=cfg.rho)
    return {"meta": meta, "raw": raw}, "contention"


def _meta(cfg: RunConfig, **extra):
    meta = dict(m=cfg.m, n=cfg.n, d=cfg.d, K=cfg.n_types, d_hat="random in [d,2d] per seed",
                T=cfg.T, cand=cfg.offer_size, sigma_own=cfg.sigma_own, sigma_obs=cfg.sigma_obs,
                seeds=list(cfg.seeds), scenario=cfg.scenario,
                source="latentswarm package (sweeps.py)")
    meta.update(extra)
    return meta


SWEEPS = {
    "bakeoff": sweep_bakeoff,          # Table 3 (c14_compare)
    "crossover": sweep_crossover,
    "anytime": sweep_anytime,
    "collab": sweep_collab,
    "scale_m": sweep_scale_m,
    "ranksweep": sweep_ranksweep,
    "offersize": sweep_offersize,
    "iid_vs_persistent": sweep_iid_vs_persistent,
    "approxrank": sweep_approxrank,   # Appendix F (F27)
    "contention": sweep_contention,   # follow-up paper Section 4 / Figure 2 (de-confliction)
}


_SAVE_SEQ = 0


def _save(payload, name, out_dir):
    """Save with a unique, chronologically-sortable stamp. A per-process sequence suffix prevents
    same-second collisions (e.g. offersize writes its cand=20 and cand=n files back-to-back)."""
    global _SAVE_SEQ
    os.makedirs(out_dir, exist_ok=True)
    payload = dict(payload)
    payload.setdefault("meta", {})
    payload["meta"]["saved_utc"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    payload["meta"]["name"] = name
    stamp = time.strftime("%Y%m%d_%H%M%S") + ("_%02d" % _SAVE_SEQ)
    _SAVE_SEQ += 1
    path = os.path.join(out_dir, f"{name}_{stamp}.json")
    json.dump(payload, open(path, "w"), indent=2, default=float)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--which", required=True, choices=sorted(SWEEPS), help="which sweep to run")
    ap.add_argument("--out", default="results/smoke",
                    help="output directory (default results/smoke; use results/pilots for paper runs)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny config (small m/n/T, 2 seeds) for a quick validity check, not a paper run")
    ap.add_argument("--seeds", type=int, default=None, help="override number of seeds")
    ap.add_argument("--cand", type=int, default=None,
                    help="override offer size c (0 = all-tasks menu); e.g. --cand 3 for the in-regime ablation")
    ap.add_argument("--workers", type=int, default=None, help="process-pool workers (default min(6,cpu))")
    args = ap.parse_args()

    cfg = base_config(smoke=args.smoke)
    if args.seeds is not None:
        cfg = _cfg_for(cfg, seeds=list(range(args.seeds)))
    if args.cand is not None:
        cfg = _cfg_for(cfg, offer_size=args.cand)
    t0 = time.time()
    fn = SWEEPS[args.which]
    result = fn(cfg)
    paths = []
    if isinstance(result, list):                 # offersize returns several files
        for payload, name in result:
            paths.append(_save(payload, name, args.out))
    else:
        payload, name = result
        paths.append(_save(payload, name, args.out))
    for p in paths:
        print("saved ->", p)
    print("done in %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
