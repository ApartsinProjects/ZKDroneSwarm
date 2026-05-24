"""Config-driven sweep drivers that reproduce the paper's headline analytical sweeps from the
LatentSwarm package, and emit the SAME JSON schema that experiments/make_figures.py already reads.

One mission "cell" is run through the genuine package path (RunConfig -> build_scenario -> ZKMRTAEnv
-> Policy -> metrics): the parity scenario `block_cosine` (the unit-cosine world matching
experiments/core.make_world), the env's persistent per-pair mask M_ik ~ Bernoulli(rho) (diag True),
per-observer noise (own ~ N(0, sigma_own^2), broadcast ~ N(0, sigma_obs^2)), a fresh per-round
random size-`cand` menu, and the held-out unseen / overall / state-uniqueness / anytime metrics from
metrics.py. The guessed rank is drawn d_hat ~ Uniform{d..2d} per seed via RandomState(9000+seed)
(the analytical-harness alignment), so every method at a given seed shares the same d_hat.

Figure-schema compatibility: the per-figure JSON uses the ANALYTICAL method KEYS (RewardCF, PTF,
ESTR, BPMF, UCBIndep, Tabular) that make_figures.py looks up, mapped from the package algorithm
names below. Sweeps emit exactly the keys each figure reader expects:

    crossover      -> c15_crossover schema  (F5/F22/F25):  raw[rho][KEY] = {overall, unseen, uniq}
    anytime        -> c16_anytime schema    (F6/F22/F24):  raw[rho][KEY] = [traj per seed]
    collab         -> collab schema         (F18a):        raw[rho][KEY] = {overall, unseen}
    scale_m        -> scale_m schema        (F18b):        raw[m][KEY]   = {overall, unseen}
    ranksweep      -> e246-style dhat sweep schema (F9 col): raw["dhat"][v][KEY] = {unseen, anytime}
    offersize      -> two crossover+anytime files, one at cand=20 and one at cand=n (F22/F25)
    iid_vs_persistent -> e12_iid_masking schema (F8):      rawA[mode][rho][KEY]={unseen,anytime,uniq}, rawB[mode][T]

IMPORTANT: this module writes ONLY to results/smoke/ by default (the development/validation sink);
results/pilots/ is the paper's figure-pipeline input and must not be polluted by smoke runs. Point
--out elsewhere only for a real (non-smoke) sweep.

CLI:  python -m latentswarm.sweeps --which crossover [--out results/smoke] [--smoke]
"""
import argparse
import json
import os
import time

import numpy as np

from .config import RunConfig
from .registry import ALGORITHMS, METRICS, get
from .scenarios import build_scenario
from .env import ZKMRTAEnv

# package algorithm name -> analytical figure-JSON key (so make_figures.py finds the series)
PKG2KEY = {
    "swarm_cf": "RewardCF",         # ours, online weighted-ALS
    "swarmcf_batch": "PTF",         # ours, batch warm-start + finetune (SwarmCF-batch)
    "estr": "ESTR",
    "bpmf": "BPMF",
    "mf_sgd": "MFSGD",
    "ucb_indep": "UCBIndep",
    "ucb_homo": "UCBHomo",
    "tabular": "Tabular",
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
    """Run ONE mission of `algo` on a fresh block_cosine world (seed) through the package path, and
    return its metrics. Mirrors pilot_c11_masking.run_masked / pilot_anytime semantics via the env.

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
    # overall and unseen each own an independent fresh stream (the package path does not reproduce
    # run_masked's single interleaved g exactly -> statistical, not bit-identical, parity).
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
# defaults matching the analytical headline regime (block_cosine, persistent mask)
# ---------------------------------------------------------------------------------------------
def base_config(smoke=False, **over) -> RunConfig:
    """The headline analytical regime as a RunConfig. smoke=True shrinks m/n/T/seeds for a quick
    validity check (NOT a paper run)."""
    if smoke:
        # tiny validity config (NOT a paper run). n is kept >= ev+engaged so the held-out unseen
        # eval (fresh size-min(c,20)=20 menus from never-engaged tasks) is well-defined at smoke size.
        d = dict(m=8, n=40, d=5, T=10, n_types=5, offer_size=20,
                 scenario="block_cosine", jitter=0.15, mask_mode="persistent",
                 rho=0.5, sigma_own=0.10, sigma_obs=0.30,
                 als_sweeps=8, refit_every=3, epsilon=0.5, epsilon_decay=0.93, epsilon_min=0.05,
                 seeds=[0, 1])
    else:
        d = dict(m=30, n=240, d=5, T=50, n_types=10, offer_size=20,
                 scenario="block_cosine", jitter=0.15, mask_mode="persistent",
                 rho=0.5, sigma_own=0.10, sigma_obs=0.30,
                 als_sweeps=8, refit_every=3, epsilon=0.5, epsilon_decay=0.93, epsilon_min=0.05,
                 seeds=list(range(16)))
    d.update(over)
    return RunConfig(**d)


_CROSSOVER_METHODS = ["ucb_indep", "mf_sgd", "estr", "bpmf", "swarmcf_batch", "swarm_cf"]
_ANYTIME_METHODS = ["random", "ucb_indep", "tabular", "estr", "swarmcf_batch", "swarm_cf"]
_COLLAB_METHODS = ["swarm_cf", "swarmcf_batch", "ucb_indep", "tabular"]
_RHOS = [1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.15, 0.1]


def sweep_crossover(cfg: RunConfig, methods=None, rhos=None):
    """Masking-robustness crossover: unseen/overall/uniq vs rho (c15_crossover schema, F5/F22/F25)."""
    methods = methods or _CROSSOVER_METHODS
    rhos = rhos or _RHOS
    raw = {str(r): {PKG2KEY[a]: {"overall": [], "unseen": [], "uniq": []} for a in methods} for r in rhos}
    for r in rhos:
        for a in methods:
            for s in cfg.seeds:
                c = run_cell(_cfg_for(cfg, rho=r), a, s)
                cell = raw[str(r)][PKG2KEY[a]]
                cell["overall"].append(c["overall"]); cell["unseen"].append(c["unseen"]); cell["uniq"].append(c["uniq"])
    meta = _meta(cfg, experiment="crossover (masking-robustness): unseen/overall vs rho",
                 methods=[PKG2KEY[a] for a in methods], rhos=rhos, cand=cfg.offer_size)
    return {"meta": meta, "raw": raw}, "c15_crossover"


def sweep_anytime(cfg: RunConfig, methods=None, rhos=None):
    """Anytime cumulative-reward trajectories per rho (c16_anytime schema, F6/F22/F24)."""
    methods = methods or _ANYTIME_METHODS
    rhos = rhos if rhos is not None else [1.0, 0.25]
    raw = {str(r): {PKG2KEY[a]: [] for a in methods} for r in rhos}
    at = get(METRICS, "anytime_trajectory")()
    for r in rhos:
        for a in methods:
            for s in cfg.seeds:
                c = run_cell(_cfg_for(cfg, rho=r), a, s, want_traj=True)
                traj = at.compute(real=c["real"], oracle=c["oracle"], random=c["random"])["trajectory"]
                raw[str(r)][PKG2KEY[a]].append(traj)
    meta = _meta(cfg, experiment="anytime cumulative-reward AUC trajectories",
                 methods=[PKG2KEY[a] for a in methods], rhos=rhos, cand=cfg.offer_size)
    return {"meta": meta, "raw": raw}, "c16_anytime"


def sweep_collab(cfg: RunConfig, methods=None, rhos=None):
    """Collaboration value: unseen/overall vs broadcast rate rho INCLUDING rho=0 (isolated)
    (collab schema, F18a)."""
    methods = methods or _COLLAB_METHODS
    rhos = rhos or [0.0, 0.1, 0.25, 0.5, 1.0]
    raw = {str(r): {PKG2KEY[a]: {"overall": [], "unseen": []} for a in methods} for r in rhos}
    for r in rhos:
        for a in methods:
            for s in cfg.seeds:
                c = run_cell(_cfg_for(cfg, rho=max(r, 1e-9)), a, s)   # rho=0 -> sees only self (diag True)
                cell = raw[str(r)][PKG2KEY[a]]
                cell["overall"].append(c["overall"]); cell["unseen"].append(c["unseen"])
    meta = _meta(cfg, experiment="collaboration value: skill vs rho (incl rho=0 isolated)",
                 methods=[PKG2KEY[a] for a in methods], rhos=rhos)
    return {"meta": meta, "raw": raw}, "collab"


def sweep_scale_m(cfg: RunConfig, methods=None, ms=None):
    """Positive scaling with swarm size m at fixed n, T, rho (scale_m schema, F18b)."""
    methods = methods or _COLLAB_METHODS
    ms = ms or ([4, 8] if cfg.m <= 8 else [5, 10, 20, 40, 80])
    raw = {str(mm): {PKG2KEY[a]: {"overall": [], "unseen": []} for a in methods} for mm in ms}
    for mm in ms:
        for a in methods:
            for s in cfg.seeds:
                # K1=min(n_types,m) so the latent rank stays = d for all m (as pilot_scale_m does)
                c = run_cell(_cfg_for(cfg, m=mm, n_types=min(cfg.n_types, mm)), a, s)
                cell = raw[str(mm)][PKG2KEY[a]]
                cell["overall"].append(c["overall"]); cell["unseen"].append(c["unseen"])
    meta = _meta(cfg, experiment="positive scaling with swarm size m", methods=[PKG2KEY[a] for a in methods], ms=ms)
    return {"meta": meta, "raw": raw}, "scale_m"


def sweep_ranksweep(cfg: RunConfig, methods=None, dhats=None):
    """Guessed-rank (d_hat) sweep: unseen + anytime vs d_hat (e246-scaling 'dhat' column schema, F9)."""
    methods = methods or ["tabular", "ucb_indep", "swarmcf_batch", "swarm_cf"]
    dhats = dhats or ([5, 8] if cfg.d <= 5 and len(cfg.seeds) <= 2 else [3, 5, 8, 10, 12])
    at = get(METRICS, "anytime_trajectory")()
    raw = {"dhat": {str(v): {PKG2KEY[a]: {"unseen": [], "anytime": []} for a in methods} for v in dhats}}
    for v in dhats:
        for a in methods:
            for s in cfg.seeds:
                c = run_cell(_cfg_for(cfg, rank_guess=int(v)), a, s, want_traj=True)
                fin = at.compute(real=c["real"], oracle=c["oracle"], random=c["random"])["final"]
                cell = raw["dhat"][str(v)][PKG2KEY[a]]
                cell["unseen"].append(c["unseen"]); cell["anytime"].append(fin)
    meta = _meta(cfg, experiment="guessed-rank d_hat sweep (unseen + anytime)",
                 methods=[PKG2KEY[a] for a in methods], sweeps={"dhat": dhats}, rho=cfg.rho)
    return {"meta": meta, "raw": raw}, "e246_scaling"


def sweep_offersize(cfg: RunConfig):
    """Offer-size robustness (F22/F25): the crossover + anytime sweeps at the body menu (cand=20) AND
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
    methods = methods or ["tabular", "ucb_indep", "swarmcf_batch", "estr", "swarm_cf"]
    rhos = rhos or [1.0, 0.5, 0.25, 0.1]
    tgrid = tgrid or ([10, 20] if len(cfg.seeds) <= 2 else [25, 50, 100, 200])
    modes = ["persistent", "iid"]
    at = get(METRICS, "anytime_trajectory")()
    rawA = {mode: {str(r): {PKG2KEY[a]: {"unseen": [], "anytime": [], "uniq": []} for a in methods}
                   for r in rhos} for mode in modes}
    rawB = {mode: {str(T): {"uniq": [], "unseen": []} for T in tgrid} for mode in modes}
    mm = {"persistent": "persistent", "iid": "per_round"}        # env mask_mode for each label
    for mode in modes:
        for r in rhos:
            for a in methods:
                for s in cfg.seeds:
                    c = run_cell(_cfg_for(cfg, rho=r, mask_mode=mm[mode]), a, s, want_traj=True)
                    fin = at.compute(real=c["real"], oracle=c["oracle"], random=c["random"])["final"]
                    cell = rawA[mode][str(r)][PKG2KEY[a]]
                    cell["unseen"].append(c["unseen"]); cell["anytime"].append(fin); cell["uniq"].append(c["uniq"])
    for mode in modes:
        for T in tgrid:
            for s in cfg.seeds:
                c = run_cell(_cfg_for(cfg, rho=0.25, T=T, mask_mode=mm[mode]), "swarm_cf", s)
                rawB[mode][str(T)]["uniq"].append(c["uniq"]); rawB[mode][str(T)]["unseen"].append(c["unseen"])
    meta = _meta(cfg, experiment="persistent vs iid masking (Theorem 4)",
                 methods=[PKG2KEY[a] for a in methods], modes=modes, rhos=rhos, Tgrid=tgrid)
    return {"meta": meta, "rawA": rawA, "rawB": rawB}, "e12_iid_masking"


# ---------------------------------------------------------------------------------------------
# contention sweep (follow-up paper, Section 4 / Figure 2): communication-free de-confliction
# ---------------------------------------------------------------------------------------------
from collections import defaultdict
from .metrics import hungarian_oracle_per_step  # reuse the Hungarian helper for the pool sub-matrix

_CONTENTION_METHODS = ["contention_ada_cf", "contention_cf", "swarm_cf", "active_cf",
                       "swarmcf_batch", "ucb_indep", "random"]
_POOLS = [240, 60, 30, 15]


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
    """One contention mission of `algo` on a fresh block_cosine world: a SHARED size-`pool` offer is
    posted each round, every robot picks one task, each contested task is awarded to one uniformly
    random contender (losers earn 0 and produce NO public engagement), and the broadcast carries
    winners' (action, outcome) only with masking rho on top. Faithful port of
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
        """Build per-robot observation dicts: shared offer = offer_idx; broadcast carries WINNERS only
        (losers masked as NO_OP), with the persistent mask + per-observer noise applied (matches the
        env's _observe semantics for the winners-only broadcast)."""
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

    # first observation has no prior engagements (won all-False)
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
        # update last_sel/last_rew for the NEXT broadcast (winners only)
        last_sel = picks.copy()
        last_rew = np.where(won, true_r, 0.0)
        # exact loss signal for the de-confliction policies (lost iff engaged but did not win)
        if hasattr(pol, "set_lost"):
            pol.set_lost([0.0 if won[i] else 1.0 for i in range(m)])
        S = rng.choice(n, size=pool, replace=False)               # next round's shared pool
        obs = _obs(S, won)
        pol.observe(obs)

    anytime = (cum_real - cum_rand) / max(cum_orac - cum_rand, 1e-9)
    pred = pol.predict_rows()
    unseen = get(METRICS, "unseen_pair_skill_heldout")().compute(
        P=P, U=U, pred_rows=pred, engaged=engaged, rng=np.random.RandomState(seed + 555),
        offer_size=cfg.offer_size)
    coll_rate = collisions / max(engagements, 1)
    return float(anytime), (float(unseen) if unseen is not None else None), float(coll_rate)


def sweep_contention(cfg: RunConfig, methods=None, pools=None):
    """De-confliction under capacity-1 contention (follow-up Figure 2 / Section 4): earned reward
    (matching-normalized), contention-free unseen skill, and collision rate vs the shared pool size
    (smaller = more contention). rho is held at cfg.rho (use 1.0 to isolate contention from masking).
    Faithful port of experiments/pilot_contention.main; the private-offset methods (SwarmCF-D+,
    SwarmCF-D) should be best-or-tied at every pool and roughly double the no-offset methods at the
    most contended pool."""
    methods = methods or _CONTENTION_METHODS
    pools = pools or _POOLS
    raw = {str(p): {PKG2KEY[a]: {"anytime": [], "unseen": [], "coll": []} for a in methods} for p in pools}
    for p in pools:
        for a in methods:
            for s in cfg.seeds:
                an, un, co = run_contention_cell(cfg, a, p, s)
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
    "crossover": sweep_crossover,
    "anytime": sweep_anytime,
    "collab": sweep_collab,
    "scale_m": sweep_scale_m,
    "ranksweep": sweep_ranksweep,
    "offersize": sweep_offersize,
    "iid_vs_persistent": sweep_iid_vs_persistent,
    "contention": sweep_contention,   # follow-up paper Section 4 / Figure 2 (de-confliction)
}


def _save(payload, name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    payload = dict(payload)
    payload.setdefault("meta", {})
    payload["meta"]["saved_utc"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    payload["meta"]["name"] = name
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"{name}_{stamp}.json")
    json.dump(payload, open(path, "w"), indent=2, default=float)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--which", required=True, choices=sorted(SWEEPS),
                    help="which sweep to run")
    ap.add_argument("--out", default="results/smoke",
                    help="output directory (default results/smoke; do NOT use results/pilots for smoke runs)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny config (small m/n/T, 2 seeds) for a quick validity check, not a paper run")
    ap.add_argument("--seeds", type=int, default=None, help="override number of seeds")
    args = ap.parse_args()

    cfg = base_config(smoke=args.smoke)
    if args.seeds is not None:
        cfg = _cfg_for(cfg, seeds=list(range(args.seeds)))
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


if __name__ == "__main__":
    main()
