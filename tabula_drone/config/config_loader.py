"""
Configuration loader for TabulaDrone scenarios.

Provides dataclasses for typed configuration and a load function
that reads and validates JSON configuration files.
"""

import json
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
    return PolicyConfig(allow_noop=bool(data["allow_noop"]))


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


def load_config(path: str) -> ScenarioConfig:
    """
    Load and validate a scenario configuration from a JSON file.
    
    Args:
        path: Path to the JSON configuration file
        
    Returns:
        ScenarioConfig with all parsed and validated settings
        
    Raises:
        FileNotFoundError: If the configuration file does not exist
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
    
    return ScenarioConfig(
        seed=seed,
        world=_parse_world_config(data["world"]),
        drones=_parse_drones_config(data["drones"]),
        targets=_parse_targets_config(data["targets"]),
        environment=_parse_environment_config(data["environment"]),
        policy=_parse_policy_config(data["policy"]),
        execution=_parse_execution_config(data["execution"]),
        logging=_parse_logging_config(data["logging"])
    )
