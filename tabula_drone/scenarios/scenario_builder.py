"""
ScenarioBuilder for randomized drone and target configuration generation.

Provides a fluent API for generating coordinated drone and target configurations
with spatial constraints and weighted random distributions.
"""

import random
from typing import Dict, List, Any, Optional, Tuple, TypedDict



class DroneParams(TypedDict):
    """Type definition for drone configuration parameters."""
    count: int
    region: Tuple[Tuple[float, float], Tuple[float, float]]
    min_distance_between_drones: float
    weapon_distribution: Dict[str, float]


class TargetParams(TypedDict):
    """Type definition for target configuration parameters."""
    count: int
    region: Tuple[Tuple[float, float], Tuple[float, float]]
    class_distribution: Dict[str, float]
    min_distance_from_drones: float
    min_distance_between_targets: float


class ScenarioBuilder:
    """
    Builder for generating randomized drone and target configurations.
    
    Uses a single random seed to ensure reproducible scenario generation.
    Coordinates drone weapon assignment and target placement with spatial
    constraints (minimum distances from drones and between targets).
    
    Example:
        >>> builder = ScenarioBuilder(world_size=(1000.0, 1000.0), seed=42)
        >>> builder.with_drones(
        ...     count=2,
        ...     region=((0.0, 0.3), (0.0, 0.5)),
        ...     min_distance_between_drones=50.0,
        ...     weapon_distribution={"light": 0.2, "medium": 0.5, "heavy": 0.3}
        ... )
        >>> builder.with_targets(
        ...     count=3,
        ...     region=((0.5, 1.0), (0.5, 1.0)),
        ...     class_distribution={"A": 0.3, "B": 0.4, "C": 0.3},
        ...     min_distance_from_drones=100.0,
        ...     min_distance_between_targets=80.0
        ... )
        >>> drones_config, targets_config = builder.build()
    """
    
    def __init__(
        self,
        world_size: Tuple[float, float],
        seed: Optional[int] = None,
        class_attribute_mapping: Dict[str, Dict[str, float]] = None,
        weapon_damage_profile_mapping: Dict[str, Dict[str, float]] = None,
    ):
        """
        Initialize ScenarioBuilder.
        
        Args:
            world_size: (width, height) bounds of 2D world
            seed: Random seed for reproducibility. If None, uses system randomness.
            class_attribute_mapping: Dict mapping class types to attribute dicts.
                Required - must be provided.
            weapon_damage_profile_mapping: Dict mapping weapon types to damage profile dicts.
                Required - must be provided.
        """
        self.world_size = world_size
        self._rng = random.Random(seed)
        self._drone_params: Optional[DroneParams] = None
        self._target_params: Optional[TargetParams] = None
        
        # Validate and store mappings (required)
        if class_attribute_mapping is None:
            raise ValueError("class_attribute_mapping is required")
        if weapon_damage_profile_mapping is None:
            raise ValueError("weapon_damage_profile_mapping is required")
        self.class_attribute_mapping = class_attribute_mapping
        self.weapon_damage_profile_mapping = weapon_damage_profile_mapping
    
    def with_drones(
        self,
        count: int,
        region: Tuple[Tuple[float, float], Tuple[float, float]],
        min_distance_between_drones: float,
        weapon_distribution: Dict[str, float]
    ) -> "ScenarioBuilder":
        """
        Configure drone parameters for the scenario.
        
        Args:
            count: Number of drones to generate. Must be positive.
            region: Tuple of ((x_min_frac, x_max_frac), (y_min_frac, y_max_frac))
                defining the spawn region as fractions of world size.
            min_distance_between_drones: Minimum Euclidean distance between any two
                drone positions. Must be non-negative.
            weapon_distribution: Dict mapping weapon types to probability weights.
                Keys must be valid weapon types ('light', 'medium', 'heavy').
                Values must be non-negative. Weights will be normalized automatically.
                Example: {"light": 0.2, "medium": 0.5, "heavy": 0.3}
        
        Returns:
            Self for method chaining
        
        Raises:
            ValueError: If count, region, or weapon_distribution is invalid
        """
        # Validate count
        if not isinstance(count, int) or count <= 0:
            raise ValueError("Drone count must be a positive integer")
        
        # Validate min_distance_between_drones
        if min_distance_between_drones < 0:
            raise ValueError("min_distance_between_drones must be non-negative")
        
        # Get valid weapon types
        valid_weapon_types = set(self.weapon_damage_profile_mapping.keys())
        
        # Validate weapon_distribution keys
        invalid_types = set(weapon_distribution.keys()) - valid_weapon_types
        if invalid_types:
            raise ValueError(
                f"Invalid weapon types in distribution: {invalid_types}. "
                f"Valid types: {valid_weapon_types}"
            )
        
        # Validate weapon_distribution values are non-negative
        for weapon_type, weight in weapon_distribution.items():
            if weight < 0:
                raise ValueError(
                    f"Weight for weapon type '{weapon_type}' must be non-negative, "
                    f"got {weight}"
                )
        
        # Check that at least one weight is positive
        if sum(weapon_distribution.values()) == 0:
            raise ValueError("All weapon distribution weights are zero")
        
        # Store parameters
        self._drone_params = {
            "count": count,
            "region": region,
            "min_distance_between_drones": min_distance_between_drones,
            "weapon_distribution": weapon_distribution
        }
        
        return self
    
    def with_targets(
        self,
        count: int,
        region: Tuple[Tuple[float, float], Tuple[float, float]],
        class_distribution: Dict[str, float],
        min_distance_from_drones: float,
        min_distance_between_targets: float
    ) -> "ScenarioBuilder":
        """
        Configure target parameters for the scenario.
        
        Args:
            count: Number of targets to generate. Must be positive.
            region: Tuple of ((x_min_frac, x_max_frac), (y_min_frac, y_max_frac))
                defining the spawn region as fractions of world size.
            class_distribution: Dict mapping target class types to probability weights.
                Keys must be valid class types (exist in DEFAULT_CLASS_ATTRIBUTE_MAPPING).
                Values must be non-negative. Weights will be normalized automatically.
                Example: {"A": 0.3, "B": 0.4, "C": 0.3}
            min_distance_from_drones: Minimum Euclidean distance between any target
                and any drone position. Must be non-negative.
            min_distance_between_targets: Minimum Euclidean distance between any two
                target positions. Must be non-negative.
        
        Returns:
            Self for method chaining
        
        Raises:
            ValueError: If count(ed), class_distribution, or distance constraints are invalid
        """
        # Validate count
        if count <= 0:
            raise ValueError(
                f"Target count must be positive, got {count}"
            )
        
        # Get valid class types (use instance attribute mapping)
        valid_class_types = set(self.class_attribute_mapping.keys())
        
        # Validate class_distribution keys
        invalid_types = set(class_distribution.keys()) - valid_class_types
        if invalid_types:
            raise ValueError(
                f"Invalid class types in distribution: {invalid_types}. "
                f"Valid types: {valid_class_types}"
            )
        
        # Validate class_distribution values are non-negative
        for class_type, weight in class_distribution.items():
            if weight < 0:
                raise ValueError(
                    f"Weight for class type '{class_type}' must be non-negative, "
                    f"got {weight}"
                )
        
        # Check that at least one weight is positive
        if sum(class_distribution.values()) == 0:
            raise ValueError("All class distribution weights are zero")
        
        # Validate distance constraints
        if min_distance_from_drones < 0:
            raise ValueError(
                f"min_distance_from_drones must be non-negative, "
                f"got {min_distance_from_drones}"
            )
        
        if min_distance_between_targets < 0:
            raise ValueError(
                f"min_distance_between_targets must be non-negative, "
                f"got {min_distance_between_targets}"
            )
        
        # Store parameters
        self._target_params = {
            "count": count,
            "region": region,
            "class_distribution": class_distribution,
            "min_distance_from_drones": min_distance_from_drones,
            "min_distance_between_targets": min_distance_between_targets
        }
        
        return self
    
    def _is_position_valid(
        self,
        pos: Tuple[float, float],
        drone_positions: List[Tuple[float, float]],
        existing_target_positions: List[Tuple[float, float]],
        min_dist_drones: float,
        min_dist_targets: float
    ) -> bool:
        """
        Check if a position satisfies all spatial constraints.
        
        Validates that the position is:
        1. Within world bounds
        2. At least min_dist_drones away from all drone positions
        3. At least min_dist_targets away from all existing target positions
        
        Args:
            pos: Position to validate (x, y)
            drone_positions: List of drone positions
            existing_target_positions: List of already-placed target positions
            min_dist_drones: Minimum distance from drones
            min_dist_targets: Minimum distance from other targets
        
        Returns:
            True if position is valid, False otherwise
        """
        x, y = pos
        world_width, world_height = self.world_size
        
        # Check 1: Within world bounds
        if not (0 <= x <= world_width and 0 <= y <= world_height):
            return False
        
        # Check 2: Distance from all drone positions
        for drone_pos in drone_positions:
            dx = x - drone_pos[0]
            dy = y - drone_pos[1]
            distance = (dx * dx + dy * dy) ** 0.5  # Euclidean distance
            if distance < min_dist_drones:
                return False
        
        # Check 3: Distance from all existing target positions
        for target_pos in existing_target_positions:
            dx = x - target_pos[0]
            dy = y - target_pos[1]
            distance = (dx * dx + dy * dy) ** 0.5  # Euclidean distance
            if distance < min_dist_targets:
                return False
        
        return True
    
    def _generate_drone_positions(
        self,
        count: int,
        region: Tuple[Tuple[float, float], Tuple[float, float]],
        min_dist_drones: float
    ) -> List[Tuple[float, float]]:
        """
        Generate valid drone positions using rejection sampling.
        
        Uses rejection sampling with a maximum of 1000 attempts per position.
        If a valid position cannot be found within the attempt limit, raises
        an error indicating the configuration is likely infeasible.
        
        Args:
            count: Number of drone positions to generate
            region: Tuple of ((x_min_frac, x_max_frac), (y_min_frac, y_max_frac))
                defining the spawn region as fractions of world size.
            min_dist_drones: Minimum distance between drones
        
        Returns:
            List of valid drone positions
        
        Raises:
            ValueError: If unable to place all drones (infeasible configuration)
        """
        world_width, world_height = self.world_size
        x_min = region[0][0] * world_width
        x_max = region[0][1] * world_width
        y_min = region[1][0] * world_height
        y_max = region[1][1] * world_height
        drone_positions: List[Tuple[float, float]] = []
        max_attempts_per_position = 1000
        
        for drone_idx in range(count):
            position_found = False
            
            for attempt in range(max_attempts_per_position):
                # Generate random position within region bounds
                x = self._rng.uniform(x_min, x_max)
                y = self._rng.uniform(y_min, y_max)
                candidate_pos = (x, y)
                
                # Check distance from all existing drone positions
                is_valid = True
                for existing_pos in drone_positions:
                    dx = x - existing_pos[0]
                    dy = y - existing_pos[1]
                    distance = (dx * dx + dy * dy) ** 0.5
                    if distance < min_dist_drones:
                        is_valid = False
                        break
                
                if is_valid:
                    drone_positions.append(candidate_pos)
                    position_found = True
                    break
            
            # If we couldn't find a valid position after max attempts
            if not position_found:
                raise ValueError(
                    f"Unable to place drone {drone_idx + 1} of {count} after "
                    f"{max_attempts_per_position} attempts. Configuration may be infeasible. "
                    f"Consider: reducing drone count, reducing minimum distance, "
                    f"or increasing region size. "
                    f"Current: world_size={self.world_size}, "
                    f"min_dist_between_drones={min_dist_drones}"
                )
        
        return drone_positions
    
    def _generate_target_positions(
        self,
        count: int,
        region: Tuple[Tuple[float, float], Tuple[float, float]],
        drone_positions: List[Tuple[float, float]],
        min_dist_drones: float,
        min_dist_targets: float
    ) -> List[Tuple[float, float]]:
        """
        Generate valid target positions using rejection sampling.
        
        Uses rejection sampling with a maximum of 1000 attempts per position.
        If a valid position cannot be found within the attempt limit, raises
        an error indicating the configuration is likely infeasible.
        
        Args:
            count: Number of target positions to generate
            region: Tuple of ((x_min_frac, x_max_frac), (y_min_frac, y_max_frac))
                defining the spawn region as fractions of world size.
            drone_positions: List of drone positions to avoid
            min_dist_drones: Minimum distance from drones
            min_dist_targets: Minimum distance between targets
        
        Returns:
            List of valid target positions
        
        Raises:
            ValueError: If unable to place all targets (infeasible configuration)
        """
        world_width, world_height = self.world_size
        x_min = region[0][0] * world_width
        x_max = region[0][1] * world_width
        y_min = region[1][0] * world_height
        y_max = region[1][1] * world_height
        target_positions: List[Tuple[float, float]] = []
        max_attempts_per_position = 1000
        
        for target_idx in range(count):
            position_found = False
            
            for attempt in range(max_attempts_per_position):
                # Generate random position within region bounds
                x = self._rng.uniform(x_min, x_max)
                y = self._rng.uniform(y_min, y_max)
                candidate_pos = (x, y)
                
                # Check if position is valid
                if self._is_position_valid(
                    candidate_pos,
                    drone_positions,
                    target_positions,  # Already-placed targets
                    min_dist_drones,
                    min_dist_targets
                ):
                    target_positions.append(candidate_pos)
                    position_found = True
                    break
            
            # If we couldn't find a valid position after max attempts
            if not position_found:
                raise ValueError(
                    f"Unable to place target {target_idx + 1} of {count} after "
                    f"{max_attempts_per_position} attempts. Configuration may be infeasible. "
                    f"Consider: reducing target count, reducing minimum distances, "
                    f"or increasing world size. "
                    f"Current: world_size={self.world_size}, {len(drone_positions)} drones, "
                    f"min_dist_from_drones={min_dist_drones}, "
                    f"min_dist_between_targets={min_dist_targets}"
                )
        
        return target_positions
    
    def _assign_weapons_internal(
        self,
        positions: List[Tuple[float, float]],
        weapon_distribution: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Assign weapon types to drones based on weighted distribution.
        
        Uses stratified assignment to guarantee proportional distribution:
        1. Calculate guaranteed count for each weapon based on weights
        2. Assign remaining slots probabilistically
        3. Shuffle final assignment to randomize positions
        
        Args:
            positions: List of drone positions
            weapon_distribution: Dict mapping weapon types to probability weights
        
        Returns:
            List of drone configuration dicts, each with "position" and "weapon_type"
        """
        num_drones = len(positions)
        
        # Normalize distribution weights to sum to 1.0
        total_weight = sum(weapon_distribution.values())
        normalized_dist = {
            weapon_type: weight / total_weight
            for weapon_type, weight in weapon_distribution.items()
        }
        
        # Calculate guaranteed counts using floor of (weight * count)
        weapon_types = list(normalized_dist.keys())
        guaranteed_counts = {
            wt: int(normalized_dist[wt] * num_drones)
            for wt in weapon_types
        }
        
        # Build list of weapon assignments with guaranteed counts
        weapon_assignments: List[str] = []
        for wt in weapon_types:
            weapon_assignments.extend([wt] * guaranteed_counts[wt])
        
        # Fill remaining slots probabilistically
        remaining = num_drones - len(weapon_assignments)
        if remaining > 0:
            weights = [normalized_dist[wt] for wt in weapon_types]
            additional = self._rng.choices(weapon_types, weights=weights, k=remaining)
            weapon_assignments.extend(additional)
        
        # Shuffle to randomize which positions get which weapon
        self._rng.shuffle(weapon_assignments)
        
        # Generate configs
        drone_configs = []
        for position, weapon_type in zip(positions, weapon_assignments):
            config = {
                "position": position,
                "weapon_type": weapon_type
            }
            drone_configs.append(config)
        
        return drone_configs
    
    def _assign_classes_internal(
        self,
        positions: List[Tuple[float, float]],
        class_distribution: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Assign class types to targets based on weighted distribution.
        
        Uses stratified assignment to guarantee proportional distribution:
        1. Calculate guaranteed count for each class based on weights
        2. Assign remaining slots probabilistically
        3. Shuffle final assignment to randomize positions
        
        Args:
            positions: List of target positions
            class_distribution: Dict mapping class types to probability weights
        
        Returns:
            List of target configuration dicts, each with "position" and "class_type"
        """
        num_targets = len(positions)
        
        # Normalize distribution weights to sum to 1.0
        total_weight = sum(class_distribution.values())
        normalized_dist = {
            class_type: weight / total_weight
            for class_type, weight in class_distribution.items()
        }
        
        # Calculate guaranteed counts using floor of (weight * count)
        class_types = list(normalized_dist.keys())
        guaranteed_counts = {
            ct: int(normalized_dist[ct] * num_targets)
            for ct in class_types
        }
        
        # Build list of class assignments with guaranteed counts
        class_assignments: List[str] = []
        for ct in class_types:
            class_assignments.extend([ct] * guaranteed_counts[ct])
        
        # Fill remaining slots probabilistically
        remaining = num_targets - len(class_assignments)
        if remaining > 0:
            weights = [normalized_dist[ct] for ct in class_types]
            additional = self._rng.choices(class_types, weights=weights, k=remaining)
            class_assignments.extend(additional)
        
        # Shuffle to randomize which positions get which class
        self._rng.shuffle(class_assignments)
        
        # Generate configs
        target_configs = []
        for position, class_type in zip(positions, class_assignments):
            config = {
                "position": position,
                "class_type": class_type,
            }
            target_configs.append(config)
        
        return target_configs
    
    def build(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Build and return drone and target configurations.
        
        Orchestrates the generation of both drone and target configurations
        using the parameters set via with_drones() and with_targets().
        
        Generation order (important for determinism):
        1. Generate drone positions (respecting spatial constraints)
        2. Assign weapons to drones
        3. Generate target positions (respecting spatial constraints)
        4. Assign classes to targets
        
        Returns:
            Tuple of (drones_config, targets_config) where:
            - drones_config: List of drone configuration dicts
            - targets_config: List of target configuration dicts
        
        Raises:
            ValueError: If with_drones() or with_targets() not called before build()
        """
        # Validate that builder has been fully configured
        if self._drone_params is None:
            raise ValueError(
                "Builder not configured for drones. "
                "Call with_drones() before build()."
            )
        
        if self._target_params is None:
            raise ValueError(
                "Builder not configured for targets. "
                "Call with_targets() before build()."
            )
        
        # Extract drone parameters
        drone_count = self._drone_params["count"]
        drone_region = self._drone_params["region"]
        min_dist_between_drones = self._drone_params["min_distance_between_drones"]
        weapon_distribution = self._drone_params["weapon_distribution"]
        
        # Extract target parameters
        target_count = self._target_params["count"]
        target_region = self._target_params["region"]
        class_distribution = self._target_params["class_distribution"]
        min_dist_drones = self._target_params["min_distance_from_drones"]
        min_dist_targets = self._target_params["min_distance_between_targets"]
        
        # Step 1: Generate drone positions with spatial constraints
        drone_positions = self._generate_drone_positions(
            count=drone_count,
            region=drone_region,
            min_dist_drones=min_dist_between_drones
        )
        
        # Step 2: Generate drone configurations with weapons
        drones_config = self._assign_weapons_internal(
            positions=drone_positions,
            weapon_distribution=weapon_distribution
        )
        
        # Step 3: Generate target positions with spatial constraints
        target_positions = self._generate_target_positions(
            count=target_count,
            region=target_region,
            drone_positions=drone_positions,
            min_dist_drones=min_dist_drones,
            min_dist_targets=min_dist_targets
        )
        
        # Step 4: Generate target configurations with classes
        targets_config = self._assign_classes_internal(
            positions=target_positions,
            class_distribution=class_distribution
        )
        
        return drones_config, targets_config

    def respawn_target(
        self,
        drone_positions: List[Tuple[float, float]],
        existing_target_positions: List[Tuple[float, float]],
    ) -> Optional[Dict[str, Any]]:
        """
        Attempt to generate a single new target configuration.
        
        Args:
            drone_positions: List of current drone positions
            existing_target_positions: List of current (active) target positions
            
        Returns:
            Target config dict or None if respawn failed after max attempts
        """
        if self._target_params is None:
            return None
            
        region = self._target_params["region"]
        class_distribution = self._target_params["class_distribution"]
        min_dist_drones = self._target_params["min_distance_from_drones"]
        min_dist_targets = self._target_params["min_distance_between_targets"]
        
        world_width, world_height = self.world_size
        x_min = region[0][0] * world_width
        x_max = region[0][1] * world_width
        y_min = region[1][0] * world_height
        y_max = region[1][1] * world_height
        
        max_attempts = 1000
        
        for _ in range(max_attempts):
            x = self._rng.uniform(x_min, x_max)
            y = self._rng.uniform(y_min, y_max)
            candidate_pos = (x, y)
            
            if self._is_position_valid(
                candidate_pos,
                drone_positions,
                existing_target_positions,
                min_dist_drones,
                min_dist_targets
            ):
                # Pick a class
                class_types = list(class_distribution.keys())
                weights = list(class_distribution.values())
                class_type = self._rng.choices(class_types, weights=weights, k=1)[0]
                
                return {
                    "position": candidate_pos,
                    "class_type": class_type
                }
        
        return None
