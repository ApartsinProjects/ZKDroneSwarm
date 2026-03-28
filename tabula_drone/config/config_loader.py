"""
Configuration loader for TabulaDrone scenarios.

Provides dataclasses for typed configuration and a load function
that reads and validates JSON configuration files.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple


@dataclass
class WorldConfig:
    """World configuration."""
    size: Tuple[float, float]


@dataclass
class DronesConfig:
    """Drone configuration."""
    count: int
    region: Tuple[Tuple[float, float], Tuple[float, float]]
    min_distance_between_drones: float
    weapon_distribution: Dict[str, float]


@dataclass
class TargetsConfig:
    """Target configuration."""
    count: int
    region: Tuple[Tuple[float, float], Tuple[float, float]]
    class_distribution: Dict[str, float]
    min_distance_from_drones: float
    min_distance_between_targets: float


@dataclass
class EpisodicModeConfig:
    """Episodic mode configuration."""
    num_episodes: int


@dataclass
class ContinuousModeConfig:
    """Continuous mode configuration."""
    logging_interval_steps: int


@dataclass
class EnvironmentConfig:
    """Environment configuration."""
    max_steps: int
    mode: str
    verbose: bool
    scenario_id: str
    episodic: Optional[EpisodicModeConfig] = None
    continuous: Optional[ContinuousModeConfig] = None


@dataclass
class PolicyConfig:
    """Policy configuration."""
    type: List[str]
    allow_noop: bool





@dataclass
class LoggingConfig:
    """Logging configuration."""
    output_dir: str


@dataclass
class CustomWorldConfig:
    """Custom-world generation configuration."""
    mappings_file: str


@dataclass
class LatentWorldConfig:
    """Latent-world benchmark generation configuration."""
    latent_dim: int
    num_modes: int
    drone_variance: float
    target_variance: float
    target_hp: float


@dataclass
class MappingsConfig:
    """Mappings configuration for class attributes and weapon damage profiles."""
    class_attribute_mapping: Dict[str, Dict[str, float]]
    weapon_damage_profile_mapping: Dict[str, Dict[str, float]]


@dataclass
class EpGreedyCFConfig:
    """ε-Greedy Collaborative Filtering policy hyperparameters."""
    latent_dim: Optional[int] = None
    learning_rate: Optional[float] = None
    epsilon: Optional[float] = None
    epsilon_decay: Optional[float] = None
    epsilon_min: Optional[float] = None
    social_trust_factor: Optional[float] = None
    divergence_threshold: Optional[float] = None
    confidence_threshold: Optional[float] = None
    social_reward_clip_min: Optional[float] = None
    max_episodes: Optional[int] = None


@dataclass
class UCBCFConfig:
    """UCB Collaborative Filtering policy hyperparameters."""
    latent_dim: Optional[int] = None
    learning_rate: Optional[float] = None
    ucb_c: Optional[float] = None


@dataclass
class MFPolicyConfig:
    """Matrix Factorization policy hyperparameters."""
    latent_dim: Optional[int] = None
    learning_rate: Optional[float] = None
    lambda_reg: Optional[float] = None
    epsilon: Optional[float] = None
    epsilon_decay: Optional[float] = None
    epsilon_min: Optional[float] = None
    reward_noise: Optional[float] = None
    selection_noise: Optional[float] = None
    anti_signal_weight: Optional[float] = None


@dataclass
class CollaborativeFilteringConfig:
    """Collaborative filtering policy configuration."""
    reward_noise: float
    ep_greedy_cf: Optional[EpGreedyCFConfig] = None
    ucb_cf: Optional[UCBCFConfig] = None
    coordinated_ep_greedy_cf: Optional[EpGreedyCFConfig] = None
    selfish_ep_greedy_cf: Optional[EpGreedyCFConfig] = None
    matrix_factorization_cf: Optional[MFPolicyConfig] = None


@dataclass
class ScenarioConfig:
    """Root configuration containing all scenario settings."""
    seed: int
    world_model: str
    world: WorldConfig
    drones: DronesConfig
    targets: TargetsConfig
    environment: EnvironmentConfig
    policy: PolicyConfig
    logging: LoggingConfig
    custom_world: Optional[CustomWorldConfig] = None
    mappings: Optional[MappingsConfig] = None
    collaborative_filtering: CollaborativeFilteringConfig = None
    latent_world: Optional[LatentWorldConfig] = None


def _validate_required_keys(data: dict, required_keys: List[str], context: str) -> None:
    """Validate that all required keys are present in a dictionary."""
    missing = [key for key in required_keys if key not in data]
    if missing:
        raise ValueError(f"Missing required keys in {context}: {missing}")


def _parse_world_config(data: dict) -> WorldConfig:
    """Parse world configuration section."""
    _validate_required_keys(data, ["size"], "world")
    size = data["size"]
    if not isinstance(size, list) or len(size) != 2:
        raise ValueError("world.size must be a list of two numbers [width, height]")
    return WorldConfig(size=tuple(size))


def _parse_custom_world_config(data: dict) -> CustomWorldConfig:
    """Parse custom-world configuration section."""
    if not isinstance(data, dict):
        raise ValueError("custom world_model requires a custom_world configuration object")

    _validate_required_keys(data, ["mappings_file"], "custom_world")
    mappings_file = data["mappings_file"]
    if not isinstance(mappings_file, str) or not mappings_file:
        raise ValueError("custom_world.mappings_file must be a non-empty string")

    return CustomWorldConfig(mappings_file=mappings_file)


def _parse_drones_config(data: dict) -> DronesConfig:
    """Parse drones configuration section."""
    _validate_required_keys(
        data,
        ["count", "region", "min_distance_between_drones", "weapon_distribution"],
        "drones"
    )
    
    count = data["count"]
    if not isinstance(count, int) or count <= 0:
        raise ValueError("drones.count must be a positive integer")
    
    region_data = data["region"]
    x_fraction = tuple(region_data["x_fraction"])
    y_fraction = tuple(region_data["y_fraction"])
    region = (x_fraction, y_fraction)
    
    weapon_distribution = data["weapon_distribution"]
    if not isinstance(weapon_distribution, dict):
        raise ValueError("drones.weapon_distribution must be a dictionary")
    
    return DronesConfig(
        count=count,
        region=region,
        min_distance_between_drones=float(data["min_distance_between_drones"]),
        weapon_distribution=weapon_distribution
    )


def _parse_targets_config(data: dict) -> TargetsConfig:
    """Parse targets configuration section."""
    _validate_required_keys(
        data,
        ["count", "region", "class_distribution", "min_distance_from_drones", "min_distance_between_targets"],
        "targets"
    )
    
    count = data["count"]
    if not isinstance(count, int) or count <= 0:
        raise ValueError("targets.count must be a positive integer")
    
    region_data = data["region"]
    x_fraction = tuple(region_data["x_fraction"])
    y_fraction = tuple(region_data["y_fraction"])
    region = (x_fraction, y_fraction)
    
    class_distribution = data["class_distribution"]
    if not isinstance(class_distribution, dict):
        raise ValueError("targets.class_distribution must be a dictionary")
    
    return TargetsConfig(
        count=count,
        region=region,
        class_distribution=class_distribution,
        min_distance_from_drones=float(data["min_distance_from_drones"]),
        min_distance_between_targets=float(data["min_distance_between_targets"])
    )


def _parse_environment_config(data: dict) -> EnvironmentConfig:
    """Parse environment configuration section."""
    _validate_required_keys(data, ["max_steps", "verbose"], "environment")
    
    max_steps = data["max_steps"]
    if not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("environment.max_steps must be a positive integer")
    
    verbose = bool(data["verbose"])
    scenario_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Handle Mode Toggling
    mode = data.get("mode", "episodic")
    if mode not in ["episodic", "continuous"]:
        raise ValueError(f"environment.mode must be 'episodic' or 'continuous', got '{mode}'")

    episodic_config = None
    continuous_config = None

    if mode == "episodic":
        # Check for nested config or flat num_episodes (backward compat)
        if "episodic" in data:
            e_data = data["episodic"]
            _validate_required_keys(e_data, ["num_episodes"], "environment.episodic")
            episodic_config = EpisodicModeConfig(num_episodes=int(e_data["num_episodes"]))
        elif "num_episodes" in data:
            # Fallback for flat structure
            episodic_config = EpisodicModeConfig(num_episodes=int(data["num_episodes"]))
        else:
            raise ValueError("Episodic mode requires 'num_episodes' (either nested or flat)")
            
    elif mode == "continuous":
        if "continuous" not in data:
            raise ValueError("Continuous mode requires 'continuous' configuration object")
        c_data = data["continuous"]
        _validate_required_keys(c_data, ["logging_interval_steps"], "environment.continuous")
        continuous_config = ContinuousModeConfig(
            logging_interval_steps=int(c_data["logging_interval_steps"])
        )

    return EnvironmentConfig(
        max_steps=max_steps,
        mode=mode,
        verbose=verbose,
        scenario_id=scenario_id,
        episodic=episodic_config,
        continuous=continuous_config
    )


def _parse_policy_config(data: dict) -> PolicyConfig:
    """Parse policy configuration section."""
    _validate_required_keys(data, ["type", "allow_noop"], "policy")
    policy_types = data["type"]
    if not isinstance(policy_types, list) or not policy_types:
        raise ValueError("policy.type must be a non-empty list of policy types")
    valid_types = {"random", "min_ttk_oracle", "max_damage_oracle", "ep_greedy_cf", "ucb_cf", "selfish_ep_greedy_cf", "coordinated_ep_greedy_cf", "matrix_factorization_cf"}
    for pt in policy_types:
        if pt not in valid_types:
            raise ValueError(f"policy.type contains invalid type '{pt}', must be one of {valid_types}")
    return PolicyConfig(type=policy_types, allow_noop=bool(data["allow_noop"]))


def _parse_ep_greedy_cf_config(data: dict) -> EpGreedyCFConfig:
    """Parse ε-Greedy CF policy configuration section (optional).
    
    Validates bounds and logs when defaults are used.
    """
    if data is None:
        return None
    
    # Extract values with defaults
    latent_dim = data.get("latent_dim")
    learning_rate = data.get("learning_rate")
    epsilon = data.get("epsilon")
    epsilon_decay = data.get("epsilon_decay")
    epsilon_min = data.get("epsilon_min")
    social_trust_factor = data.get("social_trust_factor")
    divergence_threshold = data.get("divergence_threshold")
    confidence_threshold = data.get("confidence_threshold")
    social_reward_clip_min = data.get("social_reward_clip_min")
    max_episodes = data.get("max_episodes")
    
    # Log defaults
    if latent_dim is None:
        print("Note, using default value 2 for hyperparameter latent_dim (ep_greedy_cf)")
    if learning_rate is None:
        print("Note, using default value 0.01 for hyperparameter learning_rate (ep_greedy_cf)")
    if epsilon is None:
        print("Note, using default value 0.3 for hyperparameter epsilon (ep_greedy_cf)")
    if epsilon_decay is None:
        print("Note, using default value 0.99 for hyperparameter epsilon_decay (ep_greedy_cf)")
    if epsilon_min is None:
        print("Note, using default value 0.05 for hyperparameter epsilon_min (ep_greedy_cf)")
    
    # Validate bounds
    if latent_dim is not None:
        if not isinstance(latent_dim, int) or latent_dim < 1:
            raise ValueError("ep_greedy_cf.latent_dim must be an integer >= 1")
    if learning_rate is not None:
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0 or learning_rate > 1:
            raise ValueError("ep_greedy_cf.learning_rate must be in (0, 1]")
    if epsilon is not None:
        if not isinstance(epsilon, (int, float)) or epsilon < 0 or epsilon > 1:
            raise ValueError("ep_greedy_cf.epsilon must be in [0, 1]")
    if epsilon_decay is not None:
        if not isinstance(epsilon_decay, (int, float)) or epsilon_decay < 0 or epsilon_decay > 1:
            raise ValueError("ep_greedy_cf.epsilon_decay must be in [0, 1]")
    if epsilon_min is not None:
        if not isinstance(epsilon_min, (int, float)) or epsilon_min < 0 or epsilon_min > 1:
            raise ValueError("ep_greedy_cf.epsilon_min must be in [0, 1]")
    
    return EpGreedyCFConfig(
        latent_dim=latent_dim,
        learning_rate=float(learning_rate) if learning_rate is not None else None,
        epsilon=float(epsilon) if epsilon is not None else None,
        epsilon_decay=float(epsilon_decay) if epsilon_decay is not None else None,
        epsilon_min=float(epsilon_min) if epsilon_min is not None else None,
        social_trust_factor=float(social_trust_factor) if social_trust_factor is not None else None,
        divergence_threshold=float(divergence_threshold) if divergence_threshold is not None else None,
        confidence_threshold=float(confidence_threshold) if confidence_threshold is not None else None,
        social_reward_clip_min=float(social_reward_clip_min) if social_reward_clip_min is not None else None,
        max_episodes=int(max_episodes) if max_episodes is not None else None,
    )


def _parse_selfish_ep_greedy_cf_config(data: dict) -> EpGreedyCFConfig:
    """Parse Selfish ε-Greedy CF policy configuration section (optional).
    
    Validates bounds and logs when defaults are used.
    """
    if data is None:
        return None
    
    # Extract values with defaults
    latent_dim = data.get("latent_dim")
    learning_rate = data.get("learning_rate")
    epsilon = data.get("epsilon")
    epsilon_decay = data.get("epsilon_decay")
    epsilon_min = data.get("epsilon_min")
    social_trust_factor = data.get("social_trust_factor")
    divergence_threshold = data.get("divergence_threshold")
    confidence_threshold = data.get("confidence_threshold")
    social_reward_clip_min = data.get("social_reward_clip_min")
    max_episodes = data.get("max_episodes")
    
    # Log defaults
    if latent_dim is None:
        print("Note, using default value 2 for hyperparameter latent_dim (selfish_ep_greedy_cf)")
    if learning_rate is None:
        print("Note, using default value 0.01 for hyperparameter learning_rate (selfish_ep_greedy_cf)")
    if epsilon is None:
        print("Note, using default value 0.3 for hyperparameter epsilon (selfish_ep_greedy_cf)")
    if epsilon_decay is None:
        print("Note, using default value 0.99 for hyperparameter epsilon_decay (selfish_ep_greedy_cf)")
    if epsilon_min is None:
        print("Note, using default value 0.05 for hyperparameter epsilon_min (selfish_ep_greedy_cf)")
    if social_trust_factor is None:
        print("Note, using default value 0.3 for hyperparameter social_trust_factor (selfish_ep_greedy_cf)")
    if divergence_threshold is None:
        print("Note, using default value 0.5 for hyperparameter divergence_threshold (selfish_ep_greedy_cf)")
    if confidence_threshold is None:
        print("Note, using default value 0.8 for hyperparameter confidence_threshold (selfish_ep_greedy_cf)")
    if social_reward_clip_min is None:
        print("Note, using default value -0.5 for hyperparameter social_reward_clip_min (selfish_ep_greedy_cf)")
    if max_episodes is None:
        print("Note, using default value 100 for hyperparameter max_episodes (selfish_ep_greedy_cf)")
    
    # Validate bounds
    if latent_dim is not None:
        if not isinstance(latent_dim, int) or latent_dim < 1:
            raise ValueError("selfish_ep_greedy_cf.latent_dim must be an integer >= 1")
    if learning_rate is not None:
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0 or learning_rate > 1:
            raise ValueError("selfish_ep_greedy_cf.learning_rate must be in (0, 1]")
    if epsilon is not None:
        if not isinstance(epsilon, (int, float)) or epsilon < 0 or epsilon > 1:
            raise ValueError("selfish_ep_greedy_cf.epsilon must be in [0, 1]")
    if epsilon_decay is not None:
        if not isinstance(epsilon_decay, (int, float)) or epsilon_decay < 0 or epsilon_decay > 1:
            raise ValueError("selfish_ep_greedy_cf.epsilon_decay must be in [0, 1]")
    if epsilon_min is not None:
        if not isinstance(epsilon_min, (int, float)) or epsilon_min < 0 or epsilon_min > 1:
            raise ValueError("selfish_ep_greedy_cf.epsilon_min must be in [0, 1]")
    if social_trust_factor is not None:
        if not isinstance(social_trust_factor, (int, float)) or social_trust_factor < 0 or social_trust_factor > 1:
            raise ValueError("selfish_ep_greedy_cf.social_trust_factor must be in [0, 1]")
    if divergence_threshold is not None:
        if not isinstance(divergence_threshold, (int, float)) or divergence_threshold < 0 or divergence_threshold > 1:
            raise ValueError("selfish_ep_greedy_cf.divergence_threshold must be in [0, 1]")
    if confidence_threshold is not None:
        if not isinstance(confidence_threshold, (int, float)) or confidence_threshold < 0 or confidence_threshold > 1:
            raise ValueError("selfish_ep_greedy_cf.confidence_threshold must be in [0, 1]")
    if social_reward_clip_min is not None:
        if not isinstance(social_reward_clip_min, (int, float)):
            raise ValueError("selfish_ep_greedy_cf.social_reward_clip_min must be a number")
    if max_episodes is not None:
        if not isinstance(max_episodes, int) or max_episodes < 1:
            raise ValueError("selfish_ep_greedy_cf.max_episodes must be an integer >= 1")
    
    return EpGreedyCFConfig(
        latent_dim=latent_dim,
        learning_rate=float(learning_rate) if learning_rate is not None else None,
        epsilon=float(epsilon) if epsilon is not None else None,
        epsilon_decay=float(epsilon_decay) if epsilon_decay is not None else None,
        epsilon_min=float(epsilon_min) if epsilon_min is not None else None,
        social_trust_factor=float(social_trust_factor) if social_trust_factor is not None else None,
        divergence_threshold=float(divergence_threshold) if divergence_threshold is not None else None,
        confidence_threshold=float(confidence_threshold) if confidence_threshold is not None else None,
        social_reward_clip_min=float(social_reward_clip_min) if social_reward_clip_min is not None else None,
        max_episodes=int(max_episodes) if max_episodes is not None else None,
    )


def _parse_coordinated_ep_greedy_cf_config(data: dict) -> EpGreedyCFConfig:
    """Parse Coordinated ε-Greedy CF policy configuration section (optional).
    
    Validates bounds and logs when defaults are used.
    """
    if data is None:
        return None
    
    # Extract values with defaults
    latent_dim = data.get("latent_dim")
    learning_rate = data.get("learning_rate")
    epsilon = data.get("epsilon")
    epsilon_decay = data.get("epsilon_decay")
    epsilon_min = data.get("epsilon_min")
    social_trust_factor = data.get("social_trust_factor")
    divergence_threshold = data.get("divergence_threshold")
    confidence_threshold = data.get("confidence_threshold")
    social_reward_clip_min = data.get("social_reward_clip_min")
    max_episodes = data.get("max_episodes")
    
    # Log defaults
    if latent_dim is None:
        print("Note, using default value 2 for hyperparameter latent_dim (coordinated_ep_greedy_cf)")
    if learning_rate is None:
        print("Note, using default value 0.01 for hyperparameter learning_rate (coordinated_ep_greedy_cf)")
    if epsilon is None:
        print("Note, using default value 0.3 for hyperparameter epsilon (coordinated_ep_greedy_cf)")
    if epsilon_decay is None:
        print("Note, using default value 0.99 for hyperparameter epsilon_decay (coordinated_ep_greedy_cf)")
    if epsilon_min is None:
        print("Note, using default value 0.05 for hyperparameter epsilon_min (coordinated_ep_greedy_cf)")
    if social_trust_factor is None:
        print("Note, using default value 0.3 for hyperparameter social_trust_factor (coordinated_ep_greedy_cf)")
    if divergence_threshold is None:
        print("Note, using default value 0.5 for hyperparameter divergence_threshold (coordinated_ep_greedy_cf)")
    if confidence_threshold is None:
        print("Note, using default value 0.8 for hyperparameter confidence_threshold (coordinated_ep_greedy_cf)")
    if social_reward_clip_min is None:
        print("Note, using default value -0.5 for hyperparameter social_reward_clip_min (coordinated_ep_greedy_cf)")
    if max_episodes is None:
        print("Note, using default value 100 for hyperparameter max_episodes (coordinated_ep_greedy_cf)")
    
    # Validate bounds
    if latent_dim is not None:
        if not isinstance(latent_dim, int) or latent_dim < 1:
            raise ValueError("coordinated_ep_greedy_cf.latent_dim must be an integer >= 1")
    if learning_rate is not None:
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0 or learning_rate > 1:
            raise ValueError("coordinated_ep_greedy_cf.learning_rate must be in (0, 1]")
    if epsilon is not None:
        if not isinstance(epsilon, (int, float)) or epsilon < 0 or epsilon > 1:
            raise ValueError("coordinated_ep_greedy_cf.epsilon must be in [0, 1]")
    if epsilon_decay is not None:
        if not isinstance(epsilon_decay, (int, float)) or epsilon_decay < 0 or epsilon_decay > 1:
            raise ValueError("coordinated_ep_greedy_cf.epsilon_decay must be in [0, 1]")
    if epsilon_min is not None:
        if not isinstance(epsilon_min, (int, float)) or epsilon_min < 0 or epsilon_min > 1:
            raise ValueError("coordinated_ep_greedy_cf.epsilon_min must be in [0, 1]")
    if social_trust_factor is not None:
        if not isinstance(social_trust_factor, (int, float)) or social_trust_factor < 0 or social_trust_factor > 1:
            raise ValueError("coordinated_ep_greedy_cf.social_trust_factor must be in [0, 1]")
    if divergence_threshold is not None:
        if not isinstance(divergence_threshold, (int, float)) or divergence_threshold < 0 or divergence_threshold > 1:
            raise ValueError("coordinated_ep_greedy_cf.divergence_threshold must be in [0, 1]")
    if confidence_threshold is not None:
        if not isinstance(confidence_threshold, (int, float)) or confidence_threshold < 0 or confidence_threshold > 1:
            raise ValueError("coordinated_ep_greedy_cf.confidence_threshold must be in [0, 1]")
    if social_reward_clip_min is not None:
        if not isinstance(social_reward_clip_min, (int, float)):
            raise ValueError("coordinated_ep_greedy_cf.social_reward_clip_min must be a number")
    if max_episodes is not None:
        if not isinstance(max_episodes, int) or max_episodes < 1:
            raise ValueError("coordinated_ep_greedy_cf.max_episodes must be an integer >= 1")
    
    return EpGreedyCFConfig(
        latent_dim=latent_dim,
        learning_rate=float(learning_rate) if learning_rate is not None else None,
        epsilon=float(epsilon) if epsilon is not None else None,
        epsilon_decay=float(epsilon_decay) if epsilon_decay is not None else None,
        epsilon_min=float(epsilon_min) if epsilon_min is not None else None,
        social_trust_factor=float(social_trust_factor) if social_trust_factor is not None else None,
        divergence_threshold=float(divergence_threshold) if divergence_threshold is not None else None,
        confidence_threshold=float(confidence_threshold) if confidence_threshold is not None else None,
        social_reward_clip_min=float(social_reward_clip_min) if social_reward_clip_min is not None else None,
        max_episodes=int(max_episodes) if max_episodes is not None else None,
    )


def _parse_ucb_cf_config(data: dict) -> UCBCFConfig:
    """Parse UCB CF policy configuration section (optional).
    
    Validates bounds and logs when defaults are used.
    """
    if data is None:
        return None
    
    # Extract values with defaults
    latent_dim = data.get("latent_dim")
    learning_rate = data.get("learning_rate")
    ucb_c = data.get("ucb_c")
    
    # Log defaults
    if latent_dim is None:
        print("Note, using default value 2 for hyperparameter latent_dim (ucb_cf)")
    if learning_rate is None:
        print("Note, using default value 0.01 for hyperparameter learning_rate (ucb_cf)")
    if ucb_c is None:
        print("Note, using default value 0.5 for hyperparameter ucb_c (ucb_cf)")
    
    # Validate bounds
    if latent_dim is not None:
        if not isinstance(latent_dim, int) or latent_dim < 1:
            raise ValueError("ucb_cf.latent_dim must be an integer >= 1")
    if learning_rate is not None:
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0 or learning_rate > 1:
            raise ValueError("ucb_cf.learning_rate must be in (0, 1]")
    if ucb_c is not None:
        if not isinstance(ucb_c, (int, float)) or ucb_c < 0:
            raise ValueError("ucb_cf.ucb_c must be >= 0")
    
    return UCBCFConfig(
        latent_dim=latent_dim,
        learning_rate=float(learning_rate) if learning_rate is not None else None,
        ucb_c=float(ucb_c) if ucb_c is not None else None,
    )


def _parse_mf_policy_config(data: dict) -> MFPolicyConfig:
    """Parse Matrix Factorization policy configuration section (optional).

    Validates bounds and logs when defaults are used.
    """
    if data is None:
        return None

    latent_dim = data.get("latent_dim")
    learning_rate = data.get("learning_rate")
    lambda_reg = data.get("lambda_reg")
    epsilon = data.get("epsilon")
    epsilon_decay = data.get("epsilon_decay")
    epsilon_min = data.get("epsilon_min")
    reward_noise = data.get("reward_noise")
    selection_noise = data.get("selection_noise")
    anti_signal_weight = data.get("anti_signal_weight")

    # Log defaults
    if latent_dim is None:
        print("Note, using default value 8 for hyperparameter latent_dim (matrix_factorization_cf)")
    if learning_rate is None:
        print("Note, using default value 0.01 for hyperparameter learning_rate (matrix_factorization_cf)")
    if lambda_reg is None:
        print("Note, using default value 0.02 for hyperparameter lambda_reg (matrix_factorization_cf)")
    if epsilon is None:
        print("Note, using default value 0.20 for hyperparameter epsilon (matrix_factorization_cf)")
    if epsilon_decay is None:
        print("Note, using default value 1.0 for hyperparameter epsilon_decay (matrix_factorization_cf)")
    if epsilon_min is None:
        print("Note, using default value 0.02 for hyperparameter epsilon_min (matrix_factorization_cf)")
    if reward_noise is None:
        print("Note, reward_noise not specified for matrix_factorization_cf, will inherit from collaborative_filtering section")
    if selection_noise is None:
        print("Note, using default value 0.0 for hyperparameter selection_noise (matrix_factorization_cf)")
    if anti_signal_weight is None:
        print("Note, using default value 0.1 for hyperparameter anti_signal_weight (matrix_factorization_cf)")

    # Validate bounds
    if latent_dim is not None:
        if not isinstance(latent_dim, int) or latent_dim < 1:
            raise ValueError("matrix_factorization_cf.latent_dim must be an integer >= 1")
    if learning_rate is not None:
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0 or learning_rate > 1:
            raise ValueError("matrix_factorization_cf.learning_rate must be in (0, 1]")
    if lambda_reg is not None:
        if not isinstance(lambda_reg, (int, float)) or lambda_reg < 0:
            raise ValueError("matrix_factorization_cf.lambda_reg must be >= 0")
    if epsilon is not None:
        if not isinstance(epsilon, (int, float)) or epsilon < 0 or epsilon > 1:
            raise ValueError("matrix_factorization_cf.epsilon must be in [0, 1]")
    if epsilon_decay is not None:
        if not isinstance(epsilon_decay, (int, float)) or epsilon_decay < 0 or epsilon_decay > 1:
            raise ValueError("matrix_factorization_cf.epsilon_decay must be in [0, 1]")
    if epsilon_min is not None:
        if not isinstance(epsilon_min, (int, float)) or epsilon_min < 0 or epsilon_min > 1:
            raise ValueError("matrix_factorization_cf.epsilon_min must be in [0, 1]")
    if reward_noise is not None:
        if not isinstance(reward_noise, (int, float)) or reward_noise < 0:
            raise ValueError("matrix_factorization_cf.reward_noise must be >= 0")
    if selection_noise is not None:
        if not isinstance(selection_noise, (int, float)) or selection_noise < 0:
            raise ValueError("matrix_factorization_cf.selection_noise must be >= 0")
    if anti_signal_weight is not None:
        if not isinstance(anti_signal_weight, (int, float)) or anti_signal_weight < 0:
            raise ValueError("matrix_factorization_cf.anti_signal_weight must be >= 0")

    return MFPolicyConfig(
        latent_dim=latent_dim,
        learning_rate=float(learning_rate) if learning_rate is not None else None,
        lambda_reg=float(lambda_reg) if lambda_reg is not None else None,
        epsilon=float(epsilon) if epsilon is not None else None,
        epsilon_decay=float(epsilon_decay) if epsilon_decay is not None else None,
        epsilon_min=float(epsilon_min) if epsilon_min is not None else None,
        reward_noise=float(reward_noise) if reward_noise is not None else None,
        selection_noise=float(selection_noise) if selection_noise is not None else None,
        anti_signal_weight=float(anti_signal_weight) if anti_signal_weight is not None else None,
    )


def _parse_collaborative_filtering_config(data: dict) -> CollaborativeFilteringConfig:
    """Parse collaborative filtering configuration section (optional)."""
    if data is None:
        return None
    _validate_required_keys(data, ["reward_noise"], "collaborative_filtering")
    
    # Parse nested policy configs
    ep_greedy_cf = _parse_ep_greedy_cf_config(data.get("ep_greedy_cf"))
    ucb_cf = _parse_ucb_cf_config(data.get("ucb_cf"))
    selfish_ep_greedy_cf = _parse_selfish_ep_greedy_cf_config(data.get("selfish_ep_greedy_cf"))
    coordinated_ep_greedy_cf = _parse_coordinated_ep_greedy_cf_config(data.get("coordinated_ep_greedy_cf"))
    matrix_factorization_cf = _parse_mf_policy_config(data.get("matrix_factorization_cf"))
    
    return CollaborativeFilteringConfig(
        reward_noise=float(data["reward_noise"]),
        ep_greedy_cf=ep_greedy_cf,
        ucb_cf=ucb_cf,
        selfish_ep_greedy_cf=selfish_ep_greedy_cf,
        coordinated_ep_greedy_cf=coordinated_ep_greedy_cf,
        matrix_factorization_cf=matrix_factorization_cf,
    )





def _parse_logging_config(data: dict) -> LoggingConfig:
    """Parse logging configuration section."""
    _validate_required_keys(data, ["output_dir"], "logging")
    return LoggingConfig(output_dir=str(data["output_dir"]))


def _parse_latent_world_config(data: dict) -> LatentWorldConfig:
    """Parse latent-world benchmark generation settings."""
    if data is None:
        raise ValueError("latent world_model requires a latent_world configuration object")

    _validate_required_keys(
        data,
        ["latent_dim", "num_modes", "drone_variance", "target_variance", "target_hp"],
        "latent_world",
    )

    latent_dim = data["latent_dim"]
    num_modes = data["num_modes"]
    drone_variance = data["drone_variance"]
    target_variance = data["target_variance"]
    target_hp = data["target_hp"]

    if not isinstance(latent_dim, int) or latent_dim < 1:
        raise ValueError("latent_world.latent_dim must be an integer >= 1")
    if not isinstance(num_modes, int) or num_modes < 1:
        raise ValueError("latent_world.num_modes must be an integer >= 1")
    if not isinstance(drone_variance, (int, float)) or drone_variance < 0:
        raise ValueError("latent_world.drone_variance must be >= 0")
    if not isinstance(target_variance, (int, float)) or target_variance < 0:
        raise ValueError("latent_world.target_variance must be >= 0")
    if not isinstance(target_hp, (int, float)) or target_hp <= 0:
        raise ValueError("latent_world.target_hp must be > 0")

    return LatentWorldConfig(
        latent_dim=latent_dim,
        num_modes=num_modes,
        drone_variance=float(drone_variance),
        target_variance=float(target_variance),
        target_hp=float(target_hp),
    )


def _parse_mappings_config(data: dict) -> MappingsConfig:
    """
    Parse and validate mappings configuration.
    
    Validates:
    - Both required sections are present
    - All values are dicts of dicts with float values
    - Cross-validation: weapon attribute names must be subset of target attribute names
    
    Args:
        data: Raw mappings data from JSON
        
    Returns:
        MappingsConfig with validated mappings
        
    Raises:
        ValueError: If validation fails
    """
    _validate_required_keys(
        data,
        ["class_attribute_mapping", "weapon_damage_profile_mapping"],
        "mappings"
    )
    
    class_mapping = data["class_attribute_mapping"]
    weapon_mapping = data["weapon_damage_profile_mapping"]
    
    if not isinstance(class_mapping, dict):
        raise ValueError("class_attribute_mapping must be a dictionary")
    if not isinstance(weapon_mapping, dict):
        raise ValueError("weapon_damage_profile_mapping must be a dictionary")
    
    if not class_mapping:
        raise ValueError("class_attribute_mapping must not be empty")
    if not weapon_mapping:
        raise ValueError("weapon_damage_profile_mapping must not be empty")
    
    # Validate class_attribute_mapping structure
    all_target_attributes = set()
    for class_type, attributes in class_mapping.items():
        if not isinstance(attributes, dict):
            raise ValueError(
                f"class_attribute_mapping['{class_type}'] must be a dictionary of attributes"
            )
        for attr_name, value in attributes.items():
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"class_attribute_mapping['{class_type}']['{attr_name}'] must be a number"
                )
            all_target_attributes.add(attr_name)
    
    # Validate weapon_damage_profile_mapping structure
    all_weapon_attributes = set()
    for weapon_type, damage_profile in weapon_mapping.items():
        if not isinstance(damage_profile, dict):
            raise ValueError(
                f"weapon_damage_profile_mapping['{weapon_type}'] must be a dictionary of damage values"
            )
        for attr_name, value in damage_profile.items():
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"weapon_damage_profile_mapping['{weapon_type}']['{attr_name}'] must be a number"
                )
            all_weapon_attributes.add(attr_name)
    
    # Cross-validation: weapon attributes must be subset of target attributes
    invalid_weapon_attrs = all_weapon_attributes - all_target_attributes
    if invalid_weapon_attrs:
        raise ValueError(
            f"Weapon damage profiles reference attributes not defined in any target class: "
            f"{invalid_weapon_attrs}. Valid attributes: {all_target_attributes}"
        )
    
    return MappingsConfig(
        class_attribute_mapping=class_mapping,
        weapon_damage_profile_mapping=weapon_mapping
    )


def load_mappings(path: str) -> MappingsConfig:
    """
    Load and validate mappings configuration from a JSON file.
    
    Args:
        path: Path to the mappings JSON file
        
    Returns:
        MappingsConfig with validated mappings
        
    Raises:
        FileNotFoundError: If the mappings file does not exist
        ValueError: If the mappings are invalid
    """
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Mappings file not found: {path}. "
            f"Please create a mappings.json file in the config directory."
        )
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in mappings file {path}: {e}")
    
    return _parse_mappings_config(data)


def load_config(path: str) -> ScenarioConfig:
    """
    Load and validate a scenario configuration from a JSON file.
    
    Also loads mappings.json from the same directory as the scenario config.
    
    Args:
        path: Path to the JSON configuration file
        
    Returns:
        ScenarioConfig with all parsed and validated settings including mappings
        
    Raises:
        FileNotFoundError: If the configuration file or mappings file does not exist
        ValueError: If the configuration is invalid or missing required fields
        json.JSONDecodeError: If the file contains invalid JSON
    """
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Configuration file not found: {path}. "
            f"Please create a configuration file at this location."
        )
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file {path}: {e}")
    
    world_model = str(data.get("world_model", "custom"))
    if world_model == "legacy":
        world_model = "custom"
    if world_model not in {"custom", "latent"}:
        raise ValueError("world_model must be 'custom' or 'latent' ('legacy' is still accepted as an alias for 'custom')")
    if "mappings_file" in data:
        raise ValueError("Root-level 'mappings_file' is no longer supported; use custom_world.mappings_file")

    _validate_required_keys(
        data,
        ["seed", "world", "drones", "targets", "environment", "policy", "logging"],
        "root"
    )

    seed = data["seed"]
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    custom_world = None
    latent_world = None
    mappings = None
    config_dir = os.path.dirname(path)

    if world_model == "custom":
        custom_world = _parse_custom_world_config(data.get("custom_world"))
        mappings_file = custom_world.mappings_file
        mappings_path = os.path.join(config_dir, mappings_file)
        if not os.path.exists(mappings_path):
            raise FileNotFoundError(f"Mappings file not found: {mappings_path}")
        mappings = load_mappings(mappings_path)
    else:
        latent_world = _parse_latent_world_config(data.get("latent_world"))
    
    # Parse optional collaborative_filtering config
    cf_config = _parse_collaborative_filtering_config(data.get("collaborative_filtering"))
    
    # Parse required configs
    world_config = _parse_world_config(data["world"])
    drones_config = _parse_drones_config(data["drones"])
    targets_config = _parse_targets_config(data["targets"])
    env_config = _parse_environment_config(data["environment"])
    policy_config = _parse_policy_config(data["policy"])
    logging_config = _parse_logging_config(data["logging"])
    
    return ScenarioConfig(
        seed=seed,
        world_model=world_model,
        world=world_config,
        drones=drones_config,
        targets=targets_config,
        environment=env_config,
        policy=policy_config,
        logging=logging_config,
        custom_world=custom_world,
        mappings=mappings,
        collaborative_filtering=cf_config,
        latent_world=latent_world,
    )
