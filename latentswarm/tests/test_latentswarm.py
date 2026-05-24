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


_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    for t in _TESTS:
        t()
        print("ok  ", t.__name__)
    print("all %d tests passed" % len(_TESTS))
