"""
EngagementLogger for capturing drone POV and target POV engagement data.

Produces analysis JSON files with:
- Drone POV: What each drone did (actions, targets, damage, rewards)
- Target POV: How each target was engaged (attackers, damage, elimination)
- Summary: Engagement counts per drone, elimination attribution
"""

import json
from typing import Dict, Any, List, Optional


class EngagementLogger:
    """
    Logger for capturing engagement analysis data to JSON files.
    
    Managed internally by EpisodeLogger. Produces a separate analysis JSON
    file alongside the episode JSON, containing drone-centric and target-centric
    views of all engagements.
    
    JSON Schema (version 1.1):
        {
            "version": "1.1",
            "episode_id": "<uuid>",
            "timestamp": "<ISO8601>",
            "drone_pov": {
                "<drone_id>": [
                    {
                        "step": <int>,
                        "action": "fire" | "noop",
                        "target": {
                            "target_id": "<string>",
                            "class": "<string>",
                            "status": "active" | "eliminated"
                        } | null,
                        "damage": {"<attr>": <float>, ...} | null,
                        "reward": <float>
                    },
                    ...
                ]
            },
            "target_pov": {
                "<target_id>": [
                    {
                        "step": <int>,
                        "attacker": "<drone_id>",
                        "damage": {"<attr>": <float>, ...},
                        "hp_before": <float>,
                        "hp_after": <float>,
                        "eliminated": <bool>,
                        "eliminator": "<drone_id>" | null  (only if eliminated)
                    },
                    ...
                ]
            },
            "summary": {
                "drone_engagement_counts": {
                    "<drone_id>": {"<target_id>": <int>, ...}
                },
                "eliminations": {
                    "<target_id>": "<drone_id>" | null
                }
            }
        }
    """
    
    VERSION = "1.1"
    
    def __init__(self):
        """Initialize EngagementLogger."""
        self._episode_id: Optional[str] = None
        self._timestamp: Optional[str] = None
        self._drone_pov: Dict[str, List[Dict[str, Any]]] = {}
        self._target_pov: Dict[str, List[Dict[str, Any]]] = {}
        self._drone_damage_profiles: Dict[str, Dict[str, float]] = {}
        self._target_initial_attributes: Dict[str, Dict[str, float]] = {}
        self._engagement_counts: Dict[str, Dict[str, int]] = {}
        self._eliminations: Dict[str, Optional[str]] = {}
        self._target_classes: Dict[str, str] = {}
    
    def start_episode(
        self,
        env: Any,
        episode_id: str,
        timestamp: str,
    ) -> None:
        """
        Initialize engagement tracking for a new episode.
        
        Args:
            env: The environment instance (for drone/target info)
            episode_id: Unique episode identifier
            timestamp: ISO8601 timestamp
        """
        self._episode_id = episode_id
        self._timestamp = timestamp
        
        # Initialize drone POV with empty lists
        self._drone_pov = {drone.id: [] for drone in env.drones}
        
        # Initialize target POV with empty lists
        self._target_pov = {target.id: [] for target in env.targets}
        
        # Store drone damage profiles for later use
        self._drone_damage_profiles = {}
        for drone in env.drones:
            if hasattr(drone, 'damage_profile'):
                self._drone_damage_profiles[drone.id] = dict(drone.damage_profile)
            elif hasattr(drone, 'latent_vector'):
                self._drone_damage_profiles[drone.id] = {
                    f"d{i}": v for i, v in enumerate(drone.latent_vector)
                }
            else:
                self._drone_damage_profiles[drone.id] = {}
        
        # Store target initial attributes for reference
        self._target_initial_attributes = {}
        for target in env.targets:
            if hasattr(target, 'attributes') and hasattr(target.attributes, 'initial_values'):
                self._target_initial_attributes[target.id] = dict(target.attributes.initial_values)
            elif hasattr(target, 'latent_vector'):
                self._target_initial_attributes[target.id] = {
                    f"d{i}": v for i, v in enumerate(target.latent_vector)
                }
            else:
                self._target_initial_attributes[target.id] = {}
        
        # Store target classes for drone POV logging
        self._target_classes = {}
        for target in env.targets:
            if hasattr(target, 'class_type'):
                self._target_classes[target.id] = target.class_type
            elif hasattr(target, 'mode_id'):
                self._target_classes[target.id] = f"mode_{target.mode_id}"
            else:
                self._target_classes[target.id] = "unknown"
        
        # Initialize engagement counts per drone
        self._engagement_counts = {drone.id: {} for drone in env.drones}
        
        # Initialize eliminations tracking (None = not eliminated yet)
        self._eliminations = {target.id: None for target in env.targets}

    def flush(self) -> None:
        """
        Clear accumulated step data to free memory.
        Maintains episode metadata and elimination status.
        """
        for drone_id in self._drone_pov:
            self._drone_pov[drone_id] = []
        for target_id in self._target_pov:
            self._target_pov[target_id] = []
    
    def log_engagement(
        self,
        step_num: int,
        actions: Dict[str, int],
        rewards: Dict[str, float],
        info: Dict[str, Any],
    ) -> None:
        """
        Log engagement data for a single step.
        
        Args:
            step_num: Current step number (1-indexed)
            actions: Dict of {agent_id: action} (1-indexed, 0=noop)
            rewards: Dict of {agent_id: reward}
            info: Info dict from env.step()
        """
        target_active = info.get("target_active", [])
        target_hps = info.get("target_hps", [])
        
        for drone_id, action in actions.items():
            reward = rewards.get(drone_id, 0.0)
            
            if action == 0:
                # Noop action
                drone_entry = {
                    "step": step_num,
                    "action": "noop",
                    "target": None,
                    "damage": None,
                    "reward": reward,
                }
                self._drone_pov[drone_id].append(drone_entry)
            else:
                # Fire action (action is 1-indexed, target_idx is 0-indexed)
                target_idx = action - 1
                target_id = f"target_{target_idx}"
                damage_profile = self._drone_damage_profiles.get(drone_id, {})
                
                # Get target state after damage
                is_active = target_active[target_idx] if target_idx < len(target_active) else True
                hp_after = target_hps[target_idx] if target_idx < len(target_hps) else 0.0
                
                # Determine target status
                target_status = "active" if is_active else "eliminated"
                
                # Build drone POV entry
                drone_entry = {
                    "step": step_num,
                    "action": "fire",
                    "target": {
                        "target_id": target_id,
                        "class": self._target_classes.get(target_id),
                        "status": target_status,
                    },
                    "damage": dict(damage_profile),
                    "reward": reward,
                }
                self._drone_pov[drone_id].append(drone_entry)
                
                # Update engagement count
                if target_id not in self._engagement_counts[drone_id]:
                    self._engagement_counts[drone_id][target_id] = 0
                self._engagement_counts[drone_id][target_id] += 1
                
                # Build target POV entry (only if target was actually hit, not wasted)
                # Calculate hp_before by adding damage to hp_after
                # For latent world, damage is scalar; for ZK world, sum damage profile
                total_damage = sum(damage_profile.values()) if damage_profile else 0.0
                hp_before = hp_after + total_damage
                
                # Check if this shot eliminated the target
                eliminated = not is_active and self._eliminations.get(target_id) is None
                
                target_entry = {
                    "step": step_num,
                    "attacker": drone_id,
                    "damage": dict(damage_profile),
                    "hp_before": hp_before,
                    "hp_after": hp_after,
                    "eliminated": eliminated,
                }
                
                # Record eliminator if this shot eliminated the target
                if eliminated:
                    target_entry["eliminator"] = drone_id
                    self._eliminations[target_id] = drone_id
                
                self._target_pov[target_id].append(target_entry)
    
    def end_episode(self) -> None:
        """Finalize episode and build summary."""
        # Summary is built dynamically in to_dict() from accumulated data
        # No additional finalization needed
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Return engagement data as a dictionary.
        
        Returns:
            Complete engagement analysis structure
        """
        # Build summary from accumulated data
        summary = {
            "drone_engagement_counts": {
                drone_id: dict(counts) 
                for drone_id, counts in self._engagement_counts.items()
                if counts  # Only include drones that fired at something
            },
            "eliminations": dict(self._eliminations),
        }
        
        return {
            "version": self.VERSION,
            "episode_id": self._episode_id,
            "timestamp": self._timestamp,
            "drone_pov": self._drone_pov,
            "target_pov": self._target_pov,
            "summary": summary,
        }
    
    def save(self, filepath: str) -> str:
        """
        Write engagement data to JSON file.
        
        Args:
            filepath: Path for the analysis JSON file
        
        Returns:
            Filepath of the saved JSON file
        """
        data = self.to_dict()
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        return filepath
