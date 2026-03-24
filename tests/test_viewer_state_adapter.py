import json

from viewer.state_adapter import extract_initial_state


def test_extract_initial_state_resolves_external_environment_payload(tmp_path) -> None:
    run_dir = tmp_path / "run_sample"
    episodes_dir = run_dir / "random" / "episodes"
    episodes_dir.mkdir(parents=True)

    environment_path = run_dir / "environment.json"
    environment_path.write_text(
        json.dumps(
            {
                "version": "1.2",
                "scenario_id": "run_sample",
                "config": {
                    "world_size": [100.0, 200.0],
                    "max_steps": 5,
                    "scenario_id": "run_sample",
                    "class_attribute_mapping": {
                        "A": {"hp": 10.0},
                    },
                    "weapon_damage_profile_mapping": {
                        "light": {"hp": 3.0},
                    },
                },
                "scenario": {
                    "num_drones": 1,
                    "num_targets": 1,
                    "drone_positions": [[1.0, 2.0]],
                    "target_positions": [[3.0, 4.0]],
                    "weapon_assignments": {"drone_0": "light"},
                    "target_classes": ["A"],
                },
            }
        )
    )

    episode_path = episodes_dir / "episode_first_ep01.json"
    episode_path.write_text(
        json.dumps(
            {
                "version": "1.2",
                "episode_id": "abc12345",
                "scenario_id": "run_sample",
                "episode_num": 1,
                "total_episodes": 1,
                "timestamp": "2026-03-23T00:00:00Z",
                "rng_seed": 7,
                "environment_path": "../../environment.json",
                "config": {"policy_type": "random"},
                "steps": [],
                "summary": None,
            }
        )
    )

    state = extract_initial_state(
        json.loads(episode_path.read_text()),
        episode_path=str(episode_path),
    )

    assert state["world_size"] == (100.0, 200.0)
    assert state["policy_type"] == "random"
    assert state["scenario_id"] == "run_sample"
    assert state["drones"] == [
        {"id": "drone_0", "position": (1.0, 2.0), "weapon_type": "light"}
    ]
    assert state["targets"] == [
        {"id": "target_0", "position": (3.0, 4.0), "class_type": "A", "hp": 10.0}
    ]


def test_extract_initial_state_normalizes_single_snapshot_learning_state(tmp_path) -> None:
    run_dir = tmp_path / "run_sample"
    policy_dir = run_dir / "matrix_factorization_cf"
    episodes_dir = policy_dir / "episodes"
    learning_state_dir = policy_dir / "learning_state"
    episodes_dir.mkdir(parents=True)
    learning_state_dir.mkdir(parents=True)

    environment_path = run_dir / "environment.json"
    environment_path.write_text(
        json.dumps(
            {
                "version": "1.2",
                "scenario_id": "run_sample",
                "config": {
                    "world_size": [100.0, 200.0],
                    "scenario_id": "run_sample",
                    "class_attribute_mapping": {"A": {"hp": 10.0}},
                    "weapon_damage_profile_mapping": {"light": {"hp": 3.0}},
                },
                "scenario": {
                    "drone_positions": [[1.0, 2.0]],
                    "target_positions": [[3.0, 4.0]],
                    "weapon_assignments": {"drone_0": "light"},
                    "target_classes": ["A"],
                },
            }
        )
    )

    episode_path = episodes_dir / "episode_first_ep02.json"
    episode_path.write_text(
        json.dumps(
            {
                "version": "1.2",
                "episode_num": 2,
                "environment_path": "../../environment.json",
                "config": {"policy_type": "matrix_factorization_cf"},
                "steps": [],
            }
        )
    )

    (learning_state_dir / "learning_state_ep02.json").write_text(
        json.dumps(
            {
                "episode_num": 2,
                "episode_state": {
                    "agents": [{"agent_idx": 0, "agent_lv": [1.0, 2.0], "target_lv": [[3.0, 4.0]]}]
                },
            }
        )
    )
    (learning_state_dir / "learning_state_ep01.json").write_text(
        json.dumps(
            {
                "episode_num": 1,
                "episode_state": {
                    "agents": [{"agent_idx": 0, "agent_lv": [0.0, 1.0], "target_lv": [[2.0, 3.0]]}]
                },
            }
        )
    )

    state = extract_initial_state(
        json.loads(episode_path.read_text()),
        episode_path=str(episode_path),
    )

    assert state["decentralized_learning_state"]["episode_state_agents"] == [
        {"agent_idx": 0, "agent_lv": [1.0, 2.0], "target_lv": [[3.0, 4.0]]}
    ]
    assert state["decentralized_learning_state_ep1"]["episode_state_agents"] == [
        {"agent_idx": 0, "agent_lv": [0.0, 1.0], "target_lv": [[2.0, 3.0]]}
    ]
