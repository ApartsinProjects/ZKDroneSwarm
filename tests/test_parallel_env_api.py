from pettingzoo.test import parallel_api_test

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
        scenario_id="parallel_api_test",
        mode="episodic",
    )


def test_parallel_env_api_compliance() -> None:
    env = build_test_env()
    parallel_api_test(env, num_cycles=10)
