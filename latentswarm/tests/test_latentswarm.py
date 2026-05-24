"""Smoke + contract tests for the latentswarm package.

Run either way:
    python -m latentswarm.tests.test_latentswarm     # plain runner (no pytest needed)
    pytest latentswarm/tests
"""
import numpy as np

from latentswarm import RunConfig
from latentswarm.registry import SCENARIOS, ALGORITHMS, METRICS, get
from latentswarm.scenarios import build_scenario
from latentswarm.env import ZKMRTAEnv
from latentswarm.metrics import hungarian_oracle_per_step, EarnedSkill, UnseenPairSkill, bootstrap_ci
from latentswarm.run import run_mission


def _cfg(**kw):
    base = dict(m=8, n=20, d=3, T=10, seeds=[0, 1], rank_guess=4, n_modes=3)
    base.update(kw)
    return RunConfig(**base)


def test_registries_populated():
    assert {"gaussian_mixture", "iid_gaussian"} <= set(SCENARIOS)
    assert {"random", "ucb_indep", "mf_sgd", "swarm_cf"} <= set(ALGORITHMS)
    assert {"earned_skill", "unseen_pair_skill"} <= set(METRICS)


def test_scenario_shapes():
    cfg = _cfg()
    P, U = build_scenario(cfg, np.random.RandomState(0)).generate()
    assert P.shape == (cfg.m, cfg.d) and U.shape == (cfg.n, cfg.d)


def test_env_contract_and_capacity_one():
    cfg = _cfg()
    P, U = build_scenario(cfg, np.random.RandomState(0)).generate()
    env = ZKMRTAEnv(cfg, P, U, d_guess=4, seed=0)
    obs = env.reset()
    assert len(obs) == cfg.m
    assert obs[0]["offer"].shape == (cfg.n,) and obs[0]["sel"].shape == (cfg.m,)
    assert obs[0]["offer"].all()                      # all-tasks menu by default
    actions = np.zeros(cfg.m, dtype=int)              # everyone picks task 0
    _, rew, info = env.step(actions)
    assert rew.shape == (cfg.m,)
    assert (rew != 0).sum() <= 1                      # capacity-1: at most one winner
    assert info["collisions"] >= cfg.m - 1


def test_mask_persistent_and_self_visible():
    cfg = _cfg(rho=0.5, mask_mode="persistent")
    P, U = build_scenario(cfg, np.random.RandomState(0)).generate()
    env = ZKMRTAEnv(cfg, P, U, 4, seed=0)
    env.reset()
    m1 = env.mask.copy()
    env.step(np.zeros(cfg.m, dtype=int))
    assert np.array_equal(env.mask, m1)              # persistent: unchanged across rounds
    assert env.mask.diagonal().all()                # a robot always observes itself


def test_algorithms_run_and_predict_rows():
    cfg = _cfg()
    P, U = build_scenario(cfg, np.random.RandomState(0)).generate()
    for name in ["random", "ucb_indep", "mf_sgd", "swarm_cf"]:
        env = ZKMRTAEnv(cfg, P, U, 4, seed=0)
        pol = get(ALGORITHMS, name)(cfg, cfg.m, cfg.n, 4, seed=1)
        per_round, engaged = run_mission(env, pol, cfg.T)
        assert len(per_round) == cfg.T and len(engaged) == cfg.m
        pr = pol.predict_rows()
        if name in ("mf_sgd", "swarm_cf"):
            assert pr is not None and pr.shape == (cfg.m, cfg.n)
        else:
            assert pr is None                       # structure-free: no unseen model


def test_metrics():
    cfg = _cfg()
    P, U = build_scenario(cfg, np.random.RandomState(0)).generate()
    assert np.isfinite(hungarian_oracle_per_step(P, U))
    m, lo, hi = bootstrap_ci([0.1, 0.2, 0.3, 0.4])
    assert lo <= m <= hi
    es = EarnedSkill().compute(mean_reward=0.5, random_mean=0.0, oracle_mean=1.0)
    assert abs(es - 0.5) < 1e-9


def test_rank_guess_random_in_range():
    cfg = _cfg(rank_guess="random", rank_lo=5, rank_hi=10)
    rng = np.random.RandomState(0)
    for _ in range(50):
        assert 5 <= cfg.rank_for_run(rng) <= 10


def test_swarmcf_beats_random_smoke():
    """End-to-end sanity: SwarmCF earns more than random on a small block-model mission."""
    cfg = _cfg(m=16, n=40, d=3, T=20, n_modes=3, rank_guess=4)
    P, U = build_scenario(cfg, np.random.RandomState(0)).generate()
    out = {}
    for name in ["random", "swarm_cf"]:
        env = ZKMRTAEnv(cfg, P, U, 4, seed=0)
        pol = get(ALGORITHMS, name)(cfg, cfg.m, cfg.n, 4, seed=1)
        per_round, _ = run_mission(env, pol, cfg.T)
        out[name] = float(np.mean(per_round))
    assert out["swarm_cf"] > out["random"]


# ---- new components (unification): baselines, parity scenario, metrics, configurability ----

def test_baselines_registered_and_run():
    """The ported baselines register and obey the predict_rows contract (None for structure-free)."""
    assert {"tabular", "ucb_homo", "estr", "swarmcf_batch", "bpmf", "club", "knn_cf", "soft_impute", "bias_model"} <= set(ALGORITHMS)
    cfg = _cfg(m=8, n=20, d=5, T=10, scenario="block_cosine", n_types=5, jitter=0.15,
               offer_size=20, rank_guess=6, rho=0.5, sigma_own=0.10, sigma_obs=0.30)
    P, U = build_scenario(cfg, np.random.RandomState(0)).generate()
    structure_free = {"tabular", "ucb_homo"}
    for name in ["tabular", "ucb_homo", "estr", "swarmcf_batch", "bpmf", "club", "knn_cf", "soft_impute", "bias_model"]:
        env = ZKMRTAEnv(cfg, P, U, 6, seed=0)
        pol = get(ALGORITHMS, name)(cfg, cfg.m, cfg.n, 6, seed=7)
        per_round, engaged = run_mission(env, pol, cfg.T)
        assert len(per_round) == cfg.T and len(engaged) == cfg.m
        pr = pol.predict_rows()
        if name in structure_free:
            assert pr is None
        else:
            assert pr is not None and pr.shape == (cfg.m, cfg.n)


def test_club_clusters_and_beats_floor():
    """CLUB (clustering of bandits) transfers across teammates via hard clusters, so on a clustered
    (block_cosine) world it generalizes to unseen pairs ABOVE the structure-free floor. Confirms the
    decentralized cluster-pool + global-popularity fallback predicts a full [m, n] row."""
    cfg = _cfg(m=8, n=60, d=4, T=14, scenario="block_cosine", n_types=4, jitter=0.15,
               offer_size=20, rank_guess=5, rho=0.5, sigma_own=0.10, sigma_obs=0.30)
    P, U = build_scenario(cfg, np.random.RandomState(0)).generate()
    env = ZKMRTAEnv(cfg, P, U, 5, seed=0)
    pol = get(ALGORITHMS, "club")(cfg, cfg.m, cfg.n, 5, seed=7)
    _, engaged = run_mission(env, pol, cfg.T)
    pr = pol.predict_rows()
    assert pr is not None and pr.shape == (cfg.m, cfg.n) and np.isfinite(pr).all()
    metric = get(METRICS, "unseen_pair_skill_heldout")()
    cf = metric.compute(P=P, U=U, pred_rows=pr, engaged=engaged,
                        rng=np.random.RandomState(1), offer_size=20)
    floor = metric.compute(P=P, U=U, pred_rows=None, engaged=engaged,
                           rng=np.random.RandomState(1), offer_size=20)
    assert cf is not None and floor is not None
    assert cf > floor                                     # clustering transfer lifts above the floor


def test_block_cosine_matches_core_make_world():
    """block_cosine is bit-identical to experiments/core.make_world(signed=True) and is a unit-cosine
    world (R in [-1, 1]) -- the linchpin for analytical parity."""
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "experiments"))
    from core import make_world
    m, n, d, K = 12, 30, 4, 5
    Pc, Uc, Rc, _ = make_world(m, n, d, K, K, within=0.15, seed=3, signed=True, model="block")
    cfg = _cfg(m=m, n=n, d=d, n_types=K, jitter=0.15, scenario="block_cosine")
    Pl, Ul = build_scenario(cfg, np.random.RandomState(3)).generate()
    assert np.array_equal(Pc, Pl) and np.array_equal(Uc, Ul)
    R = Pl @ Ul.T
    assert R.min() >= -1.0001 and R.max() <= 1.0001


def test_uniform_cosine_scenario():
    """uniform_cosine: i.i.d. unit-sphere traits (no types) -> signed cosine reward in [-1,1]."""
    assert "uniform_cosine" in SCENARIOS
    cfg = _cfg(m=10, n=30, d=4, scenario="uniform_cosine")
    P, U = build_scenario(cfg, np.random.RandomState(0)).generate()
    assert P.shape == (10, 4) and U.shape == (30, 4)
    assert np.allclose(np.linalg.norm(P, axis=1), 1.0) and np.allclose(np.linalg.norm(U, axis=1), 1.0)
    R = P @ U.T
    assert R.min() >= -1.0001 and R.max() <= 1.0001


def test_approx_lowrank_scenario():
    """approx_lowrank: eps=0 is exactly rank d; eps>0 raises the effective rank (factored R_eps)."""
    from latentswarm.sweeps import _eff_rank
    c0 = _cfg(m=20, n=60, d=4, scenario="approx_lowrank", approx_eps=0.0)
    P0, U0 = build_scenario(c0, np.random.RandomState(0)).generate()
    assert P0.shape == (20, 4)
    er0 = _eff_rank(P0 @ U0.T)
    c1 = _cfg(m=20, n=60, d=4, scenario="approx_lowrank", approx_eps=1.0)
    P1, U1 = build_scenario(c1, np.random.RandomState(0)).generate()
    er1 = _eff_rank(P1 @ U1.T)
    assert er0 <= 4 and er1 > er0      # the full-rank perturbation raises the effective rank


def test_new_metrics_registered_and_compute():
    from latentswarm.metrics import (AnytimeTrajectory, CumulativeRegret, TimeToCompetence,
                                      StateUniqueness, UnseenPairSkillHeldout)
    assert {"anytime_trajectory", "cumulative_regret", "time_to_competence",
            "state_uniqueness", "unseen_pair_skill_heldout"} <= set(METRICS)
    real = np.array([1.0, 2, 3, 4, 5]); orac = np.full(5, 10.0); rnd = np.full(5, 2.0)
    at = AnytimeTrajectory().compute(real=real, oracle=orac, random=rnd)
    assert len(at["trajectory"]) == 5 and np.isfinite(at["final"])
    assert CumulativeRegret().compute(real=real, oracle=orac, random=rnd) >= 0
    ttc = TimeToCompetence().compute(real=real, oracle=orac, random=rnd, frac=0.1)
    assert ttc is None or (1 <= ttc <= 5)
    # state-uniqueness: identical (non-constant) predicted matrices -> ~0 (same model, no divergence)
    base = np.arange(24, dtype=float).reshape(4, 6)
    same = [base.copy() for _ in range(4)]
    assert abs(StateUniqueness().compute(full_mats=same)) < 1e-6
    # distinct random matrices -> clearly positive uniqueness
    rng = np.random.RandomState(0)
    diff = [rng.randn(4, 6) for _ in range(4)]
    assert StateUniqueness().compute(full_mats=diff) > 0.3


def test_unseen_heldout_floor_vs_lowrank():
    """Held-out unseen: a low-rank model (swarm_cf) beats the structure-free floor (pred_rows=None)."""
    cfg = _cfg(m=8, n=60, d=4, T=12, scenario="block_cosine", n_types=4, jitter=0.15,
               offer_size=20, rank_guess=5, rho=0.5, sigma_own=0.10, sigma_obs=0.30)
    P, U = build_scenario(cfg, np.random.RandomState(0)).generate()
    env = ZKMRTAEnv(cfg, P, U, 5, seed=0)
    pol = get(ALGORITHMS, "swarm_cf")(cfg, cfg.m, cfg.n, 5, seed=7)
    _, engaged = run_mission(env, pol, cfg.T)
    metric = get(METRICS, "unseen_pair_skill_heldout")()
    cf = metric.compute(P=P, U=U, pred_rows=pol.predict_rows(), engaged=engaged,
                        rng=np.random.RandomState(1), offer_size=20)
    floor = metric.compute(P=P, U=U, pred_rows=None, engaged=engaged,
                           rng=np.random.RandomState(1), offer_size=20)
    assert cf is not None and floor is not None
    assert cf > floor                                     # low-rank lifts above the categorical floor


def test_config_defaults_lifted_literals():
    """The lifted literals keep their original defaults (configurability must not change defaults)."""
    c = RunConfig()
    assert c.factor_init_scale == 0.1 and c.mf_ridge == 1e-2 and c.buffer_window == 6000
    assert c.sensing_base_competence == 0.15 and c.sensing_specialty == 1.0
    assert c.n_types == 10


# ---- the SwarmCF-* refinement family (follow-up paper) ----

def _refine_cfg(**kw):
    base = dict(m=8, n=40, d=4, T=12, scenario="block_cosine", n_types=5, jitter=0.15,
                offer_size=20, rank_guess=6, rho=0.5, sigma_own=0.10, sigma_obs=0.30)
    base.update(kw)
    return RunConfig(**base)


def test_refinements_registered():
    """All nine SwarmCF-* refinements register as algorithms (and effective_rank as a metric)."""
    assert {"em_cf", "ard_em_cf", "active_cf", "coord_cf", "contention_cf", "contention_ada_cf",
            "choice_cf", "both_cf", "unified_cf"} <= set(ALGORITHMS)
    assert "effective_rank" in METRICS


def test_refinements_run_and_predict_rows():
    """Every refinement runs end-to-end and completes unseen entries (predict_rows -> [m, n])."""
    cfg = _refine_cfg()
    P, U = build_scenario(cfg, np.random.RandomState(0)).generate()
    for name in ["em_cf", "ard_em_cf", "active_cf", "coord_cf", "contention_cf",
                 "contention_ada_cf", "choice_cf", "both_cf", "unified_cf"]:
        env = ZKMRTAEnv(cfg, P, U, 6, seed=0)
        pol = get(ALGORITHMS, name)(cfg, cfg.m, cfg.n, 6, seed=7)
        per_round, engaged = run_mission(env, pol, cfg.T)
        assert len(per_round) == cfg.T and len(engaged) == cfg.m
        pr = pol.predict_rows()
        assert pr is not None and pr.shape == (cfg.m, cfg.n)


def test_ard_self_determines_rank_below_guess():
    """SwarmCF-B-ARD prunes the effective rank below an over-guessed d-hat (rank self-determination),
    while plain SwarmCF-B keeps the full guessed rank. Modest (non-paper) regime, but unambiguous."""
    cfg = _refine_cfg(m=20, n=120, d=3, T=40, n_types=8, rank_guess=12, rho=1.0,
                      em_sweeps=8, em_refit_every=2)
    P, U = build_scenario(cfg, np.random.RandomState(1)).generate()
    eff = {}
    for name in ["em_cf", "ard_em_cf"]:
        env = ZKMRTAEnv(cfg, P, U, 12, seed=1)
        pol = get(ALGORITHMS, name)(cfg, cfg.m, cfg.n, 12, seed=8)
        run_mission(env, pol, cfg.T)
        eff[name] = get(METRICS, "effective_rank")().compute(policy=pol)
    assert abs(eff["em_cf"] - 12.0) < 1e-6          # no ARD: every column survives the flat prior
    assert eff["ard_em_cf"] < 12.0 - 1.0            # ARD self-prunes toward the identifiable rank


def test_contention_sweep_cell_deconflicts():
    """The contention sweep cell (shared pool, capacity-1) runs and the private-offset method earns at
    least as much as plain SwarmCF under severe contention (de-confliction helps)."""
    from latentswarm.sweeps import run_contention_cell, base_config, _cfg_for
    cfg = _cfg_for(base_config(smoke=True), seeds=[0])
    an_ada, _, _ = run_contention_cell(cfg, "contention_ada_cf", 8, 0)
    an_cf, _, _ = run_contention_cell(cfg, "contention_cf", 8, 0)
    an_plain, _, _ = run_contention_cell(cfg, "swarm_cf", 8, 0)
    assert np.isfinite(an_ada) and np.isfinite(an_cf) and np.isfinite(an_plain)
    assert max(an_ada, an_cf) >= an_plain - 1e-9    # the offset does not hurt under contention


_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    for t in _TESTS:
        t()
        print("ok  ", t.__name__)
    print("all %d tests passed" % len(_TESTS))
