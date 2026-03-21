from tabula_drone.envs.drone_engage_zk_mrta_v0 import DroneEngageZKMRTA


def build_test_env() -> DroneEngageZKMRTA:
    return DroneEngageZKMRTA(
        world_size=(100.0, 100.0),
        max_steps=5,
        drones_config=[
            {"position": (10.0, 10.0), "weapon_type": "light"},
            {"position": (20.0, 20.0), "weapon_type": "medium"},
        ],
        targets_config=[
            {"position": (30.0, 30.0), "class_type": "A"},
            {"position": (40.0, 40.0), "class_type": "B"},
        ],
        scenario_id="diagnostics_test",
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


def test_reset_exposes_diagnostics_and_minimal_infos() -> None:
    env = build_test_env()

    observations, infos = env.reset(seed=42)

    assert set(observations.keys()) == set(env.agents)
    assert set(infos.keys()) == set(env.agents)
    assert all(info == {} for info in infos.values())

    diagnostics = env.diagnostics
    assert diagnostics is not None

    payload = diagnostics.to_dict()
    assert payload["step_index"] == 0
    assert payload["scenario_id"] == "diagnostics_test"
    assert payload["actions"] == {}
    assert payload["ammo_used"] == {"drone_0": 0, "drone_1": 0}
    assert payload["weapon_types"] == ["light", "medium"]
    assert payload["target_hps"] == [10.0, 12.0]
    assert payload["target_classes"] == ["A", "B"]
    assert payload["target_active"] == [True, True]
    assert "processing_order" not in payload


def test_step_updates_diagnostics_and_keeps_infos_minimal() -> None:
    env = build_test_env()

    env.reset(seed=42)
    actions = {agent_id: 0 for agent_id in env.agents}

    observations, rewards, terminations, truncations, infos = env.step(actions)

    assert set(observations.keys()) == set(env.agents)
    assert set(rewards.keys()) == set(env.agents)
    assert set(terminations.keys()) == set(env.agents)
    assert set(truncations.keys()) == set(env.agents)
    assert set(infos.keys()) == set(env.agents)
    assert all(info == {} for info in infos.values())

    diagnostics = env.diagnostics
    assert diagnostics is not None

    payload = diagnostics.to_dict()
    assert payload["step_index"] == 1
    assert payload["actions"] == actions
    assert payload["effective_damage"] == {"drone_0": 0.0, "drone_1": 0.0}
    assert payload["neutralizations_this_step"] == 0
    assert payload["cumulative_neutralizations"] == 0
    assert payload["collisions"] == 0
    assert payload["target_selections"] == {}
    assert "processing_order" in payload
    assert sorted(payload["processing_order"]) == ["drone_0", "drone_1"]
