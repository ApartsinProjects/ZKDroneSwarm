"""
LatentScenarioBuilder for Gaussian-mixture latent benchmark generation.

Builds deterministic drone and target configurations for the latent-world
benchmark using shared mode centers and isotropic Gaussian sampling.
"""

from typing import Any, Dict, List, Optional, Tuple, TypedDict

import numpy as np


class DroneParams(TypedDict):
    """Type definition for latent drone configuration parameters."""
    count: int
    region: Tuple[Tuple[float, float], Tuple[float, float]]
    min_distance_between_drones: float


class TargetParams(TypedDict):
    """Type definition for latent target configuration parameters."""
    count: int
    region: Tuple[Tuple[float, float], Tuple[float, float]]
    min_distance_from_drones: float
    min_distance_between_targets: float


class LatentScenarioBuilder:
    """
    Builder for latent Gaussian-mixture drone and target configurations.

    The builder owns shared latent mode centers. Drones and targets both sample
    from those same centers, but may use different per-side variances.
    """

    def __init__(
        self,
        world_size: Tuple[float, float],
        latent_dim: int,
        num_modes: int,
        drone_variance: float,
        target_variance: float,
        seed: Optional[int] = None,
    ):
        if latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        if num_modes < 1:
            raise ValueError("num_modes must be >= 1")
        if drone_variance < 0:
            raise ValueError("drone_variance must be >= 0")
        if target_variance < 0:
            raise ValueError("target_variance must be >= 0")

        self.world_size = world_size
        self.latent_dim = latent_dim
        self.num_modes = num_modes
        self.drone_variance = drone_variance
        self.target_variance = target_variance
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        self._drone_params: Optional[DroneParams] = None
        self._target_params: Optional[TargetParams] = None

        # Shared mixture centers define the latent world for both sides.
        self.mode_centers = self._sample_mode_centers()

    def _sample_mode_centers(self) -> np.ndarray:
        """Sample shared Gaussian-mixture centers."""
        return self._rng.normal(
            loc=0.0,
            scale=1.0,
            size=(self.num_modes, self.latent_dim),
        ).astype(np.float64)

    def with_drones(
        self,
        count: int,
        region: Tuple[Tuple[float, float], Tuple[float, float]],
        min_distance_between_drones: float,
    ) -> "LatentScenarioBuilder":
        if not isinstance(count, int) or count <= 0:
            raise ValueError("Drone count must be a positive integer")
        if min_distance_between_drones < 0:
            raise ValueError("min_distance_between_drones must be non-negative")

        self._drone_params = {
            "count": count,
            "region": region,
            "min_distance_between_drones": min_distance_between_drones,
        }
        return self

    def with_targets(
        self,
        count: int,
        region: Tuple[Tuple[float, float], Tuple[float, float]],
        min_distance_from_drones: float,
        min_distance_between_targets: float,
    ) -> "LatentScenarioBuilder":
        if not isinstance(count, int) or count <= 0:
            raise ValueError("Target count must be a positive integer")
        if min_distance_from_drones < 0:
            raise ValueError("min_distance_from_drones must be non-negative")
        if min_distance_between_targets < 0:
            raise ValueError("min_distance_between_targets must be non-negative")

        self._target_params = {
            "count": count,
            "region": region,
            "min_distance_from_drones": min_distance_from_drones,
            "min_distance_between_targets": min_distance_between_targets,
        }
        return self

    def _is_position_valid(
        self,
        pos: Tuple[float, float],
        drone_positions: List[Tuple[float, float]],
        existing_target_positions: List[Tuple[float, float]],
        min_dist_drones: float,
        min_dist_targets: float,
    ) -> bool:
        x, y = pos
        world_width, world_height = self.world_size

        if not (0 <= x <= world_width and 0 <= y <= world_height):
            return False

        for drone_pos in drone_positions:
            dx = x - drone_pos[0]
            dy = y - drone_pos[1]
            if (dx * dx + dy * dy) ** 0.5 < min_dist_drones:
                return False

        for target_pos in existing_target_positions:
            dx = x - target_pos[0]
            dy = y - target_pos[1]
            if (dx * dx + dy * dy) ** 0.5 < min_dist_targets:
                return False

        return True

    def _generate_positions(
        self,
        count: int,
        region: Tuple[Tuple[float, float], Tuple[float, float]],
        existing_positions: List[Tuple[float, float]],
        min_dist_existing: float,
        cross_positions: Optional[List[Tuple[float, float]]] = None,
        min_dist_cross: float = 0.0,
    ) -> List[Tuple[float, float]]:
        world_width, world_height = self.world_size
        x_min = region[0][0] * world_width
        x_max = region[0][1] * world_width
        y_min = region[1][0] * world_height
        y_max = region[1][1] * world_height
        positions = list(existing_positions)
        max_attempts_per_position = 1000
        cross_positions = cross_positions or []

        for idx in range(count - len(existing_positions)):
            position_found = False
            for _ in range(max_attempts_per_position):
                x = float(self._rng.uniform(x_min, x_max))
                y = float(self._rng.uniform(y_min, y_max))
                candidate_pos = (x, y)
                if self._is_position_valid(
                    candidate_pos,
                    cross_positions,
                    positions,
                    min_dist_cross,
                    min_dist_existing,
                ):
                    positions.append(candidate_pos)
                    position_found = True
                    break

            if not position_found:
                raise ValueError(
                    f"Unable to place entity {idx + 1} after {max_attempts_per_position} attempts. "
                    f"Configuration may be infeasible for world_size={self.world_size}."
                )

        return positions

    def _sample_latent_vector(self, variance: float) -> Tuple[int, Tuple[float, ...]]:
        """Sample one latent vector from the shared Gaussian mixture."""
        mode_id = int(self._rng.randint(0, self.num_modes))
        center = self.mode_centers[mode_id]
        if variance == 0:
            sample = center
        else:
            sample = self._rng.normal(
                loc=center,
                scale=np.sqrt(variance),
                size=self.latent_dim,
            )
        return mode_id, tuple(float(value) for value in sample)

    def build(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Build drone and target configs for the latent benchmark."""
        if self._drone_params is None:
            raise ValueError("Builder not configured for drones. Call with_drones() before build().")
        if self._target_params is None:
            raise ValueError("Builder not configured for targets. Call with_targets() before build().")

        drone_positions = self._generate_positions(
            count=self._drone_params["count"],
            region=self._drone_params["region"],
            existing_positions=[],
            min_dist_existing=self._drone_params["min_distance_between_drones"],
        )

        drones_config: List[Dict[str, Any]] = []
        for position in drone_positions:
            mode_id, latent_vector = self._sample_latent_vector(self.drone_variance)
            drones_config.append(
                {
                    "position": position,
                    "mode_id": mode_id,
                    "latent_vector": latent_vector,
                }
            )

        target_positions = self._generate_positions(
            count=self._target_params["count"],
            region=self._target_params["region"],
            existing_positions=[],
            min_dist_existing=self._target_params["min_distance_between_targets"],
            cross_positions=drone_positions,
            min_dist_cross=self._target_params["min_distance_from_drones"],
        )

        targets_config: List[Dict[str, Any]] = []
        for position in target_positions:
            mode_id, latent_vector = self._sample_latent_vector(self.target_variance)
            targets_config.append(
                {
                    "position": position,
                    "mode_id": mode_id,
                    "latent_vector": latent_vector,
                }
            )

        return drones_config, targets_config
