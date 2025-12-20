"""
Configuration loader for TabulaDrone scenarios.

Provides dataclasses for typed configuration and a load function
that reads and validates JSON configuration files.
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple


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
class EnvironmentConfig:
    """Environment configuration."""
    max_steps: int
    scenario_id: str


@dataclass
class PolicyConfig:
    """Policy configuration."""
    type: str
    allow_noop: bool


@dataclass
class ExecutionConfig:
    """Execution configuration."""
    num_episodes: int
    verbose: bool


@dataclass
class LoggingConfig:
    """Logging configuration."""
    output_dir: str


@dataclass
class MappingsConfig:
    """Mappings configuration for class attributes and weapon damage profiles."""
    class_attribute_mapping: Dict[str, Dict[str, float]]
    weapon_damage_profile_mapping: Dict[str, Dict[str, float]]


@dataclass
class ScenarioConfig:
    """Root configuration containing all scenario settings."""
    seed: int
    world: WorldConfig
    drones: DronesConfig
    targets: TargetsConfig
    environment: EnvironmentConfig
    policy: PolicyConfig
    execution: ExecutionConfig
    logging: LoggingConfig
    mappings: MappingsConfig


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
    _validate_required_keys(data, ["max_steps", "scenario_id"], "environment")
    
    max_steps = data["max_steps"]
    if not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("environment.max_steps must be a positive integer")
    
    return EnvironmentConfig(
        max_steps=max_steps,
        scenario_id=str(data["scenario_id"])
    )


def _parse_policy_config(data: dict) -> PolicyConfig:
    """Parse policy configuration section."""
    _validate_required_keys(data, ["allow_noop"], "policy")
    policy_type = data.get("type", "random")
    if policy_type not in ("random", "oracle"):
        raise ValueError(f"policy.type must be 'random' or 'oracle', got '{policy_type}'")
    return PolicyConfig(type=policy_type, allow_noop=bool(data["allow_noop"]))


def _parse_execution_config(data: dict) -> ExecutionConfig:
    """Parse execution configuration section."""
    _validate_required_keys(data, ["num_episodes", "verbose"], "execution")
    
    num_episodes = data["num_episodes"]
    if not isinstance(num_episodes, int) or num_episodes <= 0:
        raise ValueError("execution.num_episodes must be a positive integer")
    
    return ExecutionConfig(
        num_episodes=num_episodes,
        verbose=bool(data["verbose"])
    )


def _parse_logging_config(data: dict) -> LoggingConfig:
    """Parse logging configuration section."""
    _validate_required_keys(data, ["output_dir"], "logging")
    return LoggingConfig(output_dir=str(data["output_dir"]))


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
    
    _validate_required_keys(
        data,
        ["seed", "world", "drones", "targets", "environment", "policy", "execution", "logging"],
        "root"
    )
    
    seed = data["seed"]
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    
    # Load mappings from same directory as scenario config
    config_dir = os.path.dirname(path)
    mappings_path = os.path.join(config_dir, "mappings.json")
    mappings = load_mappings(mappings_path)
    
    return ScenarioConfig(
        seed=seed,
        world=_parse_world_config(data["world"]),
        drones=_parse_drones_config(data["drones"]),
        targets=_parse_targets_config(data["targets"]),
        environment=_parse_environment_config(data["environment"]),
        policy=_parse_policy_config(data["policy"]),
        execution=_parse_execution_config(data["execution"]),
        logging=_parse_logging_config(data["logging"]),
        mappings=mappings
    )
