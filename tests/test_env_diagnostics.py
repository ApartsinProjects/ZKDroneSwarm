from tabula_drone.envs.drone_engage_latent_mrta import DroneEngageLatentMRTA


def build_test_env() -> DroneEngageLatentMRTA:
    return DroneEngageLatentMRTA(
        world_size=(100.0, 100.0),
        max_steps=5,
        drones_config=[
            {"position": (10.0, 10.0), "latent_vector": [1.0, 0.0, 0.0], "mode_id": 0},
            {"position": (20.0, 20.0), "latent_vector": [0.0, 1.0, 0.0], "mode_id": 1},
        ],
        targets_config=[
            {"position": (30.0, 30.0), "latent_vector": [1.0, 0.0, 0.0], "mode_id": 0, "hp": 10.0},
            {"position": (40.0, 40.0), "latent_vector": [0.0, 1.0, 0.0], "mode_id": 1, "hp": 12.0},
        ],
        scenario_id="diagnostics_test",
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
    assert payload["weapon_types"] == ["mode_0", "mode_1"]
    assert payload["target_hps"] == [1.0, 1.0]  # Latent env normalizes HP to 1.0
    assert payload["target_classes"] == ["mode_0", "mode_1"]
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
    assert payload["net_damage"] == 0.0
    assert payload["neutralizations_this_step"] == 0
    assert payload["cumulative_neutralizations"] == 0
    assert payload["collisions"] == 0
    assert payload["target_selections"] == {}
    assert "processing_order" in payload
    assert sorted(payload["processing_order"]) == ["drone_0", "drone_1"]
