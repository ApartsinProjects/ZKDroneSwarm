"""
experiments/env_setup.py

Shared helper for constructing the DroneEngageLatentMRTA environment
from a loaded ScenarioConfig. Used by smoke_test.py, run_experiment.py, etc.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tabula_drone.config import load_config
from tabula_drone.envs.drone_engage_latent_mrta import DroneEngageLatentMRTA
from tabula_drone.scenarios.latent_scenario_builder import LatentScenarioBuilder


def build_env_and_configs(config, seed=None):
    """
    Build LatentScenarioBuilder + DroneEngageLatentMRTA from a ScenarioConfig.

    Returns:
        (env, drones_config, targets_config, builder)
    """
    if config.latent_world is None:
        raise ValueError("latent_world config section is required")

    builder = LatentScenarioBuilder(
        world_size=config.world.size,
        config=config.latent_world,
        target_hp=config.targets.target_hp,
        seed=seed if seed is not None else config.seed,
    )
    builder.with_drones(
        count=config.drones.count,
        region=config.drones.region,
        min_distance_between_drones=config.drones.min_distance_between_drones,
    )
    builder.with_targets(
        count=config.targets.count,
        region=config.targets.region,
        min_distance_from_drones=config.targets.min_distance_from_drones,
        min_distance_between_targets=config.targets.min_distance_between_targets,
    )
    drones_config, targets_config = builder.build()

    # Resolve noise settings
    cf_config = config.collaborative_filtering
    mf_config = cf_config.matrix_factorization_cf if cf_config else None

    if mf_config and mf_config.reward_noise is not None:
        reward_noise = mf_config.reward_noise
    elif cf_config and cf_config.reward_noise is not None:
        reward_noise = cf_config.reward_noise
    else:
        reward_noise = 0.0

    observation_noise = cf_config.observation_noise if cf_config else 0.0

    latent_world_dict = {
        "mode": builder.mode,
        "latent_dim": builder.latent_dim,
        "num_modes": builder.num_modes,
        "drone_variance": builder.drone_variance,
        "target_variance": builder.target_variance,
        "center_mode": builder.center_mode,
    }

    env = DroneEngageLatentMRTA(
        world_size=config.world.size,
        max_steps=config.environment.max_steps,
        drones_config=drones_config,
        targets_config=targets_config,
        scenario_id=config.environment.scenario_id,
        reward_noise=reward_noise,
        observation_noise=observation_noise,
        builder=builder,
        latent_world=latent_world_dict,
        target_hp=config.targets.target_hp,
    )

    return env, drones_config, targets_config, builder
