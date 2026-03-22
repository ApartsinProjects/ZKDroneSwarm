from io import StringIO

import numpy as np

from main_zk_mrta import compute_initial_oracle_reward_matrix, extract_mf_self_view_matrix
from tabula_drone.policies.matrix_factorization_policy import MatrixFactorizationPolicy
from tabula_drone.policies.multi_agent_policy import MultiAgentPolicy
from tabula_drone.utils.console_rendering import ConsolePrinter


def test_compute_initial_oracle_reward_matrix_matches_attribute_alignment_math() -> None:
    oracle = compute_initial_oracle_reward_matrix(
        drones_config=[
            {"weapon_type": "light"},
            {"weapon_type": "heavy"},
        ],
        targets_config=[
            {"class_type": "A"},
            {"class_type": "B"},
        ],
        class_attribute_mapping={
            "A": {"armor": 6.0, "shield": 2.0},
            "B": {"armor": 1.0, "shield": 7.0},
        },
        weapon_damage_profile_mapping={
            "light": {"armor": 4.0, "shield": 0.0},
            "heavy": {"armor": 0.0, "shield": 5.0},
        },
        reward_mode="ATTRIBUTE_ALIGNMENT",
    )

    expected = np.array(
        [
            [(4.0 / 4.0) * (6.0 / np.sqrt(40.0)), (1.0 / 4.0) * (1.0 / np.sqrt(50.0))],
            [(2.0 / 5.0) * (2.0 / np.sqrt(40.0)), (5.0 / 5.0) * (7.0 / np.sqrt(50.0))],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(oracle, expected, rtol=1e-9, atol=1e-9)


def test_extract_mf_self_view_matrix_uses_each_agent_local_row() -> None:
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

    policy.policies["drone_0"].P = np.array([[1.0, 2.0], [8.0, 8.0]], dtype=np.float64)
    policy.policies["drone_0"].U = np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    policy.policies["drone_1"].P = np.array([[9.0, 9.0], [7.0, 11.0]], dtype=np.float64)
    policy.policies["drone_1"].U = np.array([[1.0, 2.0], [10.0, 20.0]], dtype=np.float64)

    learned = extract_mf_self_view_matrix(policy)

    expected = np.array(
        [
            [13.0, 16.0],
            [117.0, 234.0],
        ],
        dtype=np.float64,
    )

    assert learned is not None
    np.testing.assert_allclose(learned, expected, rtol=1e-9, atol=1e-9)


def test_optimal_engagement_prediction_can_use_external_score_matrix() -> None:
    buffer = StringIO()
    printer = ConsolePrinter(stream=buffer)

    printer.optimal_engagement_prediction(
        drones_config=[
            {"weapon_type": "light"},
            {"weapon_type": "heavy"},
        ],
        targets_config=[
            {"class_type": "A"},
            {"class_type": "B"},
        ],
        class_attribute_mapping={
            "A": {"armor": 9.0, "shield": 1.0},
            "B": {"armor": 1.0, "shield": 9.0},
        },
        weapon_damage_profile_mapping={
            "light": {"armor": 9.0, "shield": 0.0},
            "heavy": {"armor": 0.0, "shield": 9.0},
        },
        score_matrix=[
            [0.1, 0.9],
            [0.8, 0.2],
        ],
    )

    output = buffer.getvalue()
    assert "Optimal Engagement Prediction (Greedy):" in output
    assert "D0" in output and "T1" in output
    assert "D1" in output and "T0" in output


def test_optimal_engagement_prediction_can_render_score_style_output() -> None:
    buffer = StringIO()
    printer = ConsolePrinter(stream=buffer)

    printer.optimal_engagement_prediction(
        drones_config=[
            {"weapon_type": "light"},
            {"weapon_type": "heavy"},
        ],
        targets_config=[
            {"class_type": "A"},
            {"class_type": "B"},
        ],
        class_attribute_mapping={
            "A": {"armor": 9.0, "shield": 1.0},
            "B": {"armor": 1.0, "shield": 9.0},
        },
        weapon_damage_profile_mapping={
            "light": {"armor": 9.0, "shield": 0.0},
            "heavy": {"armor": 0.0, "shield": 9.0},
        },
        title="Learned Engagement Prediction (Greedy)",
        score_matrix=[
            [0.1234, 0.9234],
            [0.8456, 0.2456],
        ],
        value_header="Score",
        value_matrix=[
            [0.1234, 0.9234],
            [0.8456, 0.2456],
        ],
        value_precision=3,
    )

    output = buffer.getvalue()
    assert "Learned Engagement Prediction (Greedy):" in output
    assert "Score" in output
    assert "0.923" in output
    assert "0.846" in output
