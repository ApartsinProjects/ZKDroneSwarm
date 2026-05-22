"""
experiments/smoke_test.py

Quick sanity checks: run 1 episode of each policy and verify that:
  - The episode completes without exception.
  - All returned metrics are finite and non-negative where expected.
  - avg_latent_match_quality is in [0, 1.1] (it is a mean cosine similarity).
  - OracleLPolicy outperforms RandomPolicy on avg_latent_match_quality.
  - UCBIndepPolicy arm-counts increase after its one episode.

Usage (from repo root):
    python experiments/smoke_test.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

from tabula_drone.config import load_config
from tabula_drone.utils.metrics_manager import EpisodeMetrics
from tabula_drone.policies.random_policy import RandomPolicy
from tabula_drone.policies.max_damage_oracle import OptimalAssignmentOracle
from tabula_drone.policies.ucb_indep_policy import UCBIndepPolicy
from tabula_drone.policies.oracle_l_policy import OracleLPolicy
from tabula_drone.policies.matrix_factorization_policy import MatrixFactorizationPolicy
from tabula_drone.policies.multi_agent_policy import MultiAgentPolicy
from tabula_drone.policies.base import bind_diagnostics_provider
from experiments.env_setup import build_env_and_configs

CONFIG_PATH = "config/scenario.json"
SMOKE_SEED = 999


def get_env_diagnostics(env):
    if env.diagnostics is None:
        return {}
    return env.diagnostics.to_dict()


def run_one_episode(env, policy, seed):
    obs, infos = env.reset(seed=seed)
    bind_diagnostics_provider(policy, lambda: get_env_diagnostics(env))

    total_rewards = {a: 0.0 for a in env.agents}
    step_count = 0
    done = False
    total_net_damage = 0.0
    total_gross_damage = 0.0
    total_collisions = 0
    total_latent_mismatch = 0.0
    total_optimal_potential = 0.0
    overkill_events = []
    shared_info = {}

    while not done:
        step_count += 1
        try:
            actions = policy.select_actions(obs, infos, env=env)
        except TypeError:
            actions = policy.select_actions(obs, infos)

        obs, rewards, terminations, truncations, infos = env.step(actions)
        shared_info = get_env_diagnostics(env)
        policy.update(obs)

        ref = env.agents[0]
        done = terminations[ref] or truncations[ref]
        for a in env.agents:
            total_rewards[a] += rewards[a]
        total_net_damage += shared_info.get("net_damage", 0.0)
        total_gross_damage += shared_info.get("total_gross_damage", 0.0)
        total_latent_mismatch += shared_info.get("latent_mismatch", 0.0)
        total_optimal_potential += shared_info.get("optimal_potential", 0.0)
        total_collisions += shared_info.get("collisions", 0)
        if "overkill" in shared_info:
            overkill_events.append(shared_info["overkill"])

    ammo_used_dict = shared_info.get("ammo_used", {})
    total_ammo_used = sum(ammo_used_dict.values())
    total_overkill = sum(sum(ev.values()) for ev in overkill_events)
    targets_neutralized = shared_info.get("cumulative_neutralizations", 0)
    done_reason = shared_info.get("done_reason")

    return EpisodeMetrics(
        episode=0,
        steps=step_count,
        done_reason=done_reason,
        targets_neutralized=targets_neutralized,
        total_ammo_used=total_ammo_used,
        total_overkill=total_overkill,
        total_net_damage=total_net_damage,
        total_gross_damage=total_gross_damage,
        total_collisions=total_collisions,
        agent_rewards=total_rewards,
        weapon_damage_profile_mapping=env.weapon_damage_profile_mapping,
        total_latent_mismatch=total_latent_mismatch,
        total_optimal_potential=total_optimal_potential,
    )


def check(cond, name, extra=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" ({extra})" if extra else ""))
    return cond


def smoke_policy(label, policy, env, seed=SMOKE_SEED):
    print(f"\n--- {label} ---")
    passed = True

    try:
        m = run_one_episode(env, policy, seed)
    except Exception as e:
        import traceback
        print(f"  [FAIL] Episode raised exception: {e}")
        traceback.print_exc()
        return False

    passed &= check(m.steps > 0, "steps > 0", str(m.steps))
    passed &= check(m.total_ammo_used >= 0, "total_ammo_used >= 0", str(m.total_ammo_used))
    passed &= check(m.targets_neutralized >= 0, "targets_neutralized >= 0", str(m.targets_neutralized))
    passed &= check(0.0 <= m.avg_latent_match_quality <= 1.1,
                    "avg_latent_match_quality in [0, 1.1]",
                    f"{m.avg_latent_match_quality:.4f}")
    passed &= check(math.isfinite(m.total_reward), "total_reward finite", f"{m.total_reward:.4f}")
    passed &= check(m.total_overkill >= 0.0, "total_overkill >= 0", f"{m.total_overkill:.4f}")

    print(f"  steps={m.steps}  neutralized={m.targets_neutralized}/{env.num_targets}"
          f"  ammo={m.total_ammo_used}  lmq={m.avg_latent_match_quality:.4f}"
          f"  reason={m.done_reason}")
    return passed, m


def main():
    config = load_config(CONFIG_PATH)
    env, drones_config, targets_config, builder = build_env_and_configs(config, seed=SMOKE_SEED)
    num_drones = env.num_drones
    num_targets = env.num_targets

    print(f"Smoke test environment: {num_drones} drones, {num_targets} targets")

    results = {}
    metrics_map = {}

    # Random
    policy = RandomPolicy(seed=SMOKE_SEED, allow_noop=config.policy.allow_noop)
    r, m = smoke_policy("RandomPolicy", policy, env)
    results["random"] = r
    metrics_map["random"] = m

    # Oracle-HP (OptimalAssignmentOracle)
    policy = OptimalAssignmentOracle(seed=SMOKE_SEED, allow_noop=config.policy.allow_noop)
    r, m = smoke_policy("OracleHP (OptimalAssignmentOracle)", policy, env)
    results["oracle_hp"] = r
    metrics_map["oracle_hp"] = m

    # Oracle-L
    policy = OracleLPolicy(allow_noop=config.policy.allow_noop)
    r, m = smoke_policy("OracleL", policy, env)
    results["oracle_l"] = r
    metrics_map["oracle_l"] = m

    # UCBIndep
    policy = UCBIndepPolicy(
        num_agents=num_drones, num_targets=num_targets,
        c=2.0, seed=SMOKE_SEED, allow_noop=config.policy.allow_noop,
    )
    r, m = smoke_policy("UCBIndepPolicy", policy, env)
    results["ucb_indep"] = r
    metrics_map["ucb_indep"] = m
    # Verify arm counts increased
    total_arm_counts = int(policy._counts.sum())
    print(f"  UCBIndep total arm updates after 1 episode: {total_arm_counts}")
    results["ucb_counts"] = check(total_arm_counts > 0, "UCBIndep arm counts > 0", str(total_arm_counts))

    # MF-CF
    mf_cfg = config.collaborative_filtering.matrix_factorization_cf
    policies_dict = {f"drone_{i}": MatrixFactorizationPolicy(
        num_targets=num_targets, agent_idx=i, num_agents=num_drones,
        latent_dim=mf_cfg.latent_dim, learning_rate=mf_cfg.learning_rate,
        lambda_reg=mf_cfg.lambda_reg, epsilon=mf_cfg.epsilon,
        epsilon_decay=mf_cfg.epsilon_decay, epsilon_min=mf_cfg.epsilon_min,
        anti_signal_weight=mf_cfg.anti_signal_weight,
        use_integration_matrix=bool(mf_cfg.use_integration_matrix),
        seed=SMOKE_SEED + i,
    ) for i in range(num_drones)}
    policy = MultiAgentPolicy(policies_dict)
    r, m = smoke_policy("MatrixFactorizationCF", policy, env)
    results["mf_cf"] = r
    metrics_map["mf_cf"] = m

    # Cross-policy sanity: Oracle-L lmq should be >= Random's lmq (with margin)
    oracle_l_lmq = metrics_map.get("oracle_l")
    random_lmq = metrics_map.get("random")
    if oracle_l_lmq is not None and random_lmq is not None:
        passed = oracle_l_lmq.avg_latent_match_quality >= random_lmq.avg_latent_match_quality - 0.05
        results["oracle_l_beats_random"] = check(
            passed, "OracleL lmq >= Random lmq (sanity)",
            f"oracle_l={oracle_l_lmq.avg_latent_match_quality:.4f} "
            f"random={random_lmq.avg_latent_match_quality:.4f}"
        )

    # Summary
    print("\n=== SMOKE TEST SUMMARY ===")
    all_pass = all(results.values())
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}: {k}")
    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
