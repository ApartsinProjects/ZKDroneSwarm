import json

from main_zk_mrta import run_episode
from tabula_drone.envs.drone_engage_zk_mrta_v0 import DroneEngageZKMRTA
from tabula_drone.logging import EnvironmentLogger
from tabula_drone.policies.matrix_factorization_policy import MatrixFactorizationPolicy
from tabula_drone.policies.max_damage_oracle import OptimalAssignmentOracle
from tabula_drone.policies.min_ttk_oracle import OracleTimeToKillPolicy
from tabula_drone.policies.multi_agent_policy import MultiAgentPolicy
from tabula_drone.policies.random_policy import RandomPolicy


def build_test_env() -> DroneEngageZKMRTA:
    return DroneEngageZKMRTA(
        world_size=(100.0, 100.0),
        max_steps=3,
        drones_config=[
            {"position": (10.0, 10.0), "weapon_type": "light"},
            {"position": (20.0, 20.0), "weapon_type": "medium"},
        ],
        targets_config=[
            {"position": (30.0, 30.0), "class_type": "A"},
            {"position": (40.0, 40.0), "class_type": "B"},
        ],
        scenario_id="runner_test",
        class_attribute_mapping={
            "A": {"hp": 10.0},
            "B": {"hp": 12.0},
        },
        weapon_damage_profile_mapping={
            "light": {"hp": 3.0},
            "medium": {"hp": 4.0},
        },
        mode="episodic",
    )


def test_run_episode_uses_env_diagnostics_with_random_policy() -> None:
    env = build_test_env()
    policy = RandomPolicy(seed=1)

    metrics = run_episode(env, policy, episode_num=1, seed=7)

    assert metrics["episode"] == 1
    assert metrics["steps"] >= 1
    assert metrics["total_ammo_used"] >= 0
    assert "done_reason" in metrics
    assert set(metrics["agent_rewards"].keys()) == {"drone_0", "drone_1"}


def test_run_episode_uses_env_diagnostics_with_ttk_oracle() -> None:
    env = build_test_env()
    policy = OracleTimeToKillPolicy(
        agent_weapon_profiles={
            "drone_0": {"hp": 3.0},
            "drone_1": {"hp": 4.0},
        },
        allow_noop=True,
    )

    metrics = run_episode(env, policy, episode_num=1, seed=7)

    assert metrics["episode"] == 1
    assert metrics["steps"] >= 1
    assert metrics["total_ammo_used"] >= 0
    assert "done_reason" in metrics
    assert set(metrics["agent_rewards"].keys()) == {"drone_0", "drone_1"}


def test_run_episode_uses_env_diagnostics_with_max_damage_oracle() -> None:
    env = build_test_env()
    policy = OptimalAssignmentOracle(
        agent_weapon_profiles={
            "drone_0": {"hp": 3.0},
            "drone_1": {"hp": 4.0},
        },
        allow_noop=True,
    )

    metrics = run_episode(env, policy, episode_num=1, seed=7)

    assert metrics["episode"] == 1
    assert metrics["steps"] >= 1
    assert metrics["total_ammo_used"] >= 0
    assert "done_reason" in metrics
    assert set(metrics["agent_rewards"].keys()) == {"drone_0", "drone_1"}


def test_run_episode_uses_env_diagnostics_with_matrix_factorization_policy() -> None:
    env = build_test_env()
    policy = MultiAgentPolicy(
        {
            "drone_0": MatrixFactorizationPolicy(
                num_targets=2,
                agent_idx=0,
                num_agents=2,
                latent_dim=2,
                epsilon=0.0,
                seed=1,
            ),
            "drone_1": MatrixFactorizationPolicy(
                num_targets=2,
                agent_idx=1,
                num_agents=2,
                latent_dim=2,
                epsilon=0.0,
                seed=2,
            ),
        }
    )

    metrics = run_episode(env, policy, episode_num=1, seed=7)

    assert metrics["episode"] == 1
    assert metrics["steps"] >= 1
    assert metrics["total_ammo_used"] >= 0
    assert "done_reason" in metrics
    assert set(metrics["agent_rewards"].keys()) == {"drone_0", "drone_1"}


def test_run_episode_uses_environment_logger_boundary(tmp_path) -> None:
    env = build_test_env()
    policy = RandomPolicy(seed=1)
    environment_logger = EnvironmentLogger(
        output_dir=str(tmp_path),
        scenario_id="runner_test",
        mode="episodic",
    )
    environment_logger.start_policy("random_policy", is_deterministic=True)

    metrics = run_episode(
        env,
        policy,
        episode_num=1,
        seed=7,
        environment_logger=environment_logger,
    )
    environment_logger.persist_episode_outputs(episode_num=1, steps=metrics["steps"])
    result = environment_logger.finalize_policy()

    episode_path = (
        tmp_path / "runner_test" / "random_policy" / "episodes" / "episode_first_ep01.json"
    )
    environment_path = tmp_path / "runner_test" / "environment.json"

    assert metrics["episode"] == 1
    assert result["steps"] == {"first": metrics["steps"]}
    assert episode_path.is_file()
    assert environment_path.is_file()

    episode_data = json.loads(episode_path.read_text())
    environment_data = json.loads(environment_path.read_text())

    assert episode_data["environment_path"] == "../../environment.json"
    assert episode_data["config"] == {"policy_type": "random_policy"}
    assert "scenario" not in episode_data
    assert environment_data["scenario_id"] == "runner_test"
    assert "policy_type" not in environment_data["config"]
