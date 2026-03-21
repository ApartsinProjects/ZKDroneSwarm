from pettingzoo.test import parallel_api_test

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
        scenario_id="parallel_api_test",
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


def test_parallel_env_api_compliance() -> None:
    env = build_test_env()
    parallel_api_test(env, num_cycles=10)
