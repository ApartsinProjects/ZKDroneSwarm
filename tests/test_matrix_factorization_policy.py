import numpy as np

from tabula_drone.envs.drone_engage_latent_mrta import DroneEngageLatentMRTA
from tabula_drone.policies.matrix_factorization_policy import MatrixFactorizationPolicy


def build_integration_env() -> DroneEngageLatentMRTA:
    return DroneEngageLatentMRTA(
        world_size=(100.0, 100.0),
        max_steps=3,
        drones_config=[
            {"position": (10.0, 10.0), "latent_vector": [1.0, 0.0, 0.0], "mode_id": 0},
            {"position": (20.0, 20.0), "latent_vector": [0.0, 1.0, 0.0], "mode_id": 1},
        ],
        targets_config=[
            {"position": (30.0, 30.0), "latent_vector": [1.0, 1.0, 0.0], "mode_id": 0, "hp": 1.0},
        ],
        scenario_id="integration_matrix_test",
    )


def test_env_observation_omits_activity_mask_and_keeps_compatibility_rewards() -> None:
    env = build_integration_env()

    observations, _ = env.reset(seed=42)
    first_observation = observations["drone_0"]

    assert "target_was_active_at_engagement" not in first_observation

    step_observations, rewards, _, _, _ = env.step({"drone_0": 1, "drone_1": 1})
    shared_observation = step_observations["drone_0"]

    expected_reward = 1.0 / np.sqrt(2.0)
    assert "target_was_active_at_engagement" not in shared_observation
    assert int(np.count_nonzero(shared_observation["selected_targets"] == 1)) == 2
    assert all(np.isclose(reward, expected_reward) for reward in rewards.values())


def test_integration_matrix_accumulates_all_observed_rewards() -> None:
    policy = MatrixFactorizationPolicy(
        num_targets=2,
        agent_idx=0,
        num_agents=2,
        latent_dim=2,
        learning_rate=0.0,
        use_integration_matrix=True,
        seed=1,
    )

    observation = {
        "targets": np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "selected_targets": np.array([1, 2], dtype=np.int32),
        "observed_rewards": np.array([0.5, 0.25], dtype=np.float32),
    }

    policy.update_from_observation(observation)

    assert policy.M_count[0, 0] == 1.0
    assert policy.M_sum[0, 0] == 0.5
    assert policy.M_count[1, 1] == 1.0
    assert policy.M_sum[1, 1] == 0.25


def test_non_integration_mode_still_uses_observed_reward_directly() -> None:
    policy = MatrixFactorizationPolicy(
        num_targets=1,
        agent_idx=0,
        num_agents=1,
        latent_dim=1,
        learning_rate=0.1,
        lambda_reg=0.0,
        anti_signal_weight=1.0,
        use_integration_matrix=False,
        seed=1,
    )
    policy.P = np.array([[0.5]], dtype=np.float64)
    policy.U = np.array([[0.4]], dtype=np.float64)

    observation = {
        "targets": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "selected_targets": np.array([1], dtype=np.int32),
        "observed_rewards": np.array([0.1], dtype=np.float32),
    }

    policy.update_from_observation(observation)

    assert np.allclose(policy.P, np.array([[0.492]], dtype=np.float64))
    assert np.allclose(policy.U, np.array([[0.39]], dtype=np.float64))


def test_learning_state_exposes_predicted_integration_matrix() -> None:
    policy = MatrixFactorizationPolicy(
        num_targets=2,
        agent_idx=0,
        num_agents=2,
        latent_dim=2,
        use_integration_matrix=True,
        seed=1,
    )
    policy.P = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    policy.U = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    policy.M_sum = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float64)
    policy.M_count = np.array([[2.0, 4.0], [3.0, 4.0]], dtype=np.float64)

    state = policy.get_learning_state()

    integration_matrix = state["integration_matrix"]
    assert integration_matrix["M_avg"] == [[1.0, 1.0], [2.0, 2.0]]
    assert integration_matrix["M_pred"] == [[19.0, 22.0], [43.0, 50.0]]
    assert integration_matrix["M_count"] == [[2.0, 4.0], [3.0, 4.0]]
