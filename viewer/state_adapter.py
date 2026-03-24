"""
State adapter for loading and extracting data from episode logs.
"""

import json
import os
from typing import Dict, Any, List, Tuple, Optional


def load_episode(path: str) -> Dict[str, Any]:
    """
    Load episode log from JSON file.
    
    Args:
        path: Path to episode log JSON file
        
    Returns:
        Episode data dict
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    with open(path, "r") as f:
        return json.load(f)


def extract_initial_state(
    episode_data: Dict[str, Any],
    episode_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract initial world state for visualization.
    
    Handles both v1.0 (no config) and v1.1 (with config) episode logs.
    For decentralized policies, loads external learning_state.json if available.
    
    Args:
        episode_data: Episode data dict from load_episode()
        episode_path: Path to episode log file (needed to resolve learning_state_folder)
        
    Returns:
        Dict with:
            - world_size: (width, height) tuple
            - drones: list of {id, position, weapon_type}
            - targets: list of {id, position, class_type}
            - version: episode log version
            - class_attribute_mapping: dict mapping class types to attribute dicts
            - weapon_damage_profile_mapping: dict mapping weapon types to damage profile dicts
            - decentralized_learning_state: normalized learning-state payload (if decentralized)
            - learning_state_folder: relative path to learning state folder (if decentralized)
    """
    version = episode_data.get("version", "1.0")
    if version != "1.2":
        raise ValueError(
            f"Unsupported episode log version '{version}'. Viewer supports version '1.2' only."
        )
    scenario, config = _resolve_environment_payload(episode_data, episode_path)
    
    # Extract world size (v1.1 has it in config, v1.0 needs fallback)
    if "world_size" in config:
        world_size = tuple(config["world_size"])
    else:
        # Fallback: infer from positions or use default
        world_size = _infer_world_size(scenario)
    
    # Extract drones
    drone_positions = scenario.get("drone_positions", [])
    weapon_assignments = scenario.get("weapon_assignments", {})
    drones = []
    for i, pos in enumerate(drone_positions):
        drone_id = f"drone_{i}"
        drones.append({
            "id": drone_id,
            "position": tuple(pos),
            "weapon_type": weapon_assignments.get(drone_id, "unknown"),
        })
    
    # Extract targets
    target_positions = scenario.get("target_positions", [])
    target_classes = scenario.get("target_classes", [])
    class_attribute_mapping = config.get("class_attribute_mapping", {})
    targets = []
    for i, pos in enumerate(target_positions):
        target_id = f"target_{i}"
        class_type = target_classes[i] if i < len(target_classes) else "unknown"
        # Calculate initial HP as sum of all integrity attributes
        attributes = class_attribute_mapping.get(class_type, {})
        hp = sum(attributes.values()) if attributes else 0.0
        targets.append({
            "id": target_id,
            "position": tuple(pos),
            "class_type": class_type,
            "hp": hp,
        })
    
    # Extract class attribute mapping
    class_attribute_mapping = config.get("class_attribute_mapping", {})
    
    # Extract weapon damage profile mapping
    weapon_damage_profile_mapping = config.get("weapon_damage_profile_mapping", {})
    
    summary = episode_data.get("summary", None)
    hp_history = extract_hp_history(episode_data)
    active_targets_history = extract_active_targets_history(episode_data)
    seed = episode_data.get("rng_seed", None)
    scenario_id = config.get("scenario_id", None)
    policy_type = config.get("policy_type", None)
    
    steps = episode_data.get("steps", [])
    learning_path = episode_data.get("learning_path", None)
    total_episodes = episode_data.get("total_episodes", None)
    
    # Extract decentralized learning state if available
    decentralized_learning_state = None
    decentralized_learning_state_ep1 = None
    
    if episode_path:
        # New structure: learning_state is in sibling folder
        # Episode path: logs/run_*/policy/episodes/episode_*.json
        # Learning state: logs/run_*/policy/learning_state/learning_state_ep{NN}.json
        episodes_dir = os.path.dirname(episode_path)
        policy_dir = os.path.dirname(episodes_dir)
        learning_state_dir = os.path.join(policy_dir, "learning_state")
        
        if os.path.isdir(learning_state_dir):
            episode_num = episode_data.get("episode_num", 1)
            
            # Load learning state for current episode
            learning_state_path = os.path.join(
                learning_state_dir, f"learning_state_ep{episode_num:02d}.json"
            )
            if os.path.exists(learning_state_path):
                try:
                    with open(learning_state_path, "r") as f:
                        decentralized_learning_state = _normalize_learning_state_payload(
                            json.load(f)
                        )
                except (json.JSONDecodeError, IOError):
                    decentralized_learning_state = None
            
            # Load episode 1's learning state for comparison (random initialization)
            ep1_path = os.path.join(learning_state_dir, "learning_state_ep01.json")
            if os.path.exists(ep1_path):
                try:
                    with open(ep1_path, "r") as f:
                        decentralized_learning_state_ep1 = _normalize_learning_state_payload(
                            json.load(f)
                        )
                except (json.JSONDecodeError, IOError):
                    decentralized_learning_state_ep1 = None
    
    return {
        "world_size": world_size,
        "drones": drones,
        "targets": targets,
        "version": version,
        "class_attribute_mapping": class_attribute_mapping,
        "weapon_damage_profile_mapping": weapon_damage_profile_mapping,
        "summary": summary,
        "hp_history": hp_history,
        "active_targets_history": active_targets_history,
        "seed": seed,
        "scenario_id": scenario_id,
        "policy_type": policy_type,
        "steps": steps,
        "learning_path": learning_path,
        "total_episodes": total_episodes,
        "decentralized_learning_state": decentralized_learning_state,
        "decentralized_learning_state_ep1": decentralized_learning_state_ep1,
    }


def _normalize_learning_state_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a learning-state artifact to the viewer's single-snapshot contract.
    """
    normalized = dict(payload)
    snapshot = payload.get("episode_state")
    if not isinstance(snapshot, dict):
        return normalized
    agents = snapshot.get("agents", []) if isinstance(snapshot, dict) else []

    normalized["episode_state"] = snapshot
    normalized["episode_state_agents"] = agents
    return normalized


def _resolve_environment_payload(
    episode_data: Dict[str, Any],
    episode_path: Optional[str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Resolve shared environment data for both normalized and legacy episodes."""
    scenario = episode_data.get("scenario", {})
    config = episode_data.get("config", {})

    if scenario and "world_size" in config:
        return scenario, config

    environment_ref = episode_data.get("environment_path")
    if not environment_ref or not episode_path:
        return scenario, config

    environment_path = os.path.normpath(
        os.path.join(os.path.dirname(episode_path), environment_ref)
    )
    with open(environment_path, "r") as f:
        environment_data = json.load(f)

    merged_config = dict(environment_data.get("config", {}))
    merged_config.update(config)
    return environment_data.get("scenario", {}), merged_config


def extract_hp_history(episode_data: Dict[str, Any]) -> List[float]:
    """
    Extract aggregated total HP per step from episode data.
    
    Args:
        episode_data: Episode data dict from load_episode()
        
    Returns:
        List of total HP values (sum of all targets) per step.
        Returns empty list if steps data is missing or invalid.
    """
    steps = episode_data.get("steps", [])
    if not steps:
        return []
    
    hp_history = []
    for step in steps:
        info = step.get("info", {})
        target_hps = info.get("target_hps", [])
        if target_hps:
            hp_history.append(sum(target_hps))
        else:
            # If target_hps missing, skip this step
            break
    
    return hp_history


def extract_active_targets_history(episode_data: Dict[str, Any]) -> List[int]:
    """
    Extract count of active targets (HP > 0) per step from episode data.
    
    Args:
        episode_data: Episode data dict from load_episode()
        
    Returns:
        List of active target counts per step.
        Returns empty list if steps data is missing or invalid.
    """
    steps = episode_data.get("steps", [])
    if not steps:
        return []
    
    active_targets_history = []
    for step in steps:
        info = step.get("info", {})
        target_hps = info.get("target_hps", [])
        if target_hps:
            active_count = len([hp for hp in target_hps if hp > 0])
            active_targets_history.append(active_count)
        else:
            break
    
    return active_targets_history


def _infer_world_size(scenario: Dict[str, Any]) -> Tuple[float, float]:
    """
    Infer world size from entity positions (fallback for v1.0 logs).
    
    Args:
        scenario: Scenario dict from episode log
        
    Returns:
        (width, height) tuple with 20% padding
    """
    all_positions = []
    
    for pos in scenario.get("drone_positions", []):
        all_positions.append(pos)
    for pos in scenario.get("target_positions", []):
        all_positions.append(pos)
    
    if not all_positions:
        return (1000.0, 1000.0)  # Default
    
    max_x = max(p[0] for p in all_positions)
    max_y = max(p[1] for p in all_positions)
    
    # Add 20% padding
    return (max_x * 1.2, max_y * 1.2)
