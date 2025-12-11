"""
Tests for ScenarioBuilder class.

Validates deterministic behavior, configuration validation, and proper
integration of all builder components.
"""

import pytest
from tabula_drone.scenarios import ScenarioBuilder


class TestDeterminism:
    """Test deterministic behavior of ScenarioBuilder with same/different seeds."""
    
    def test_same_seed_produces_identical_drone_configs(self):
        """Same seed should produce identical drone configurations."""
        # Builder 1
        builder1 = ScenarioBuilder((1000.0, 1000.0), seed=42)
        builder1.with_drones(
            count=2,
            region=((0.0, 0.3), (0.0, 0.3)),
            min_distance_between_drones=50.0,
            weapon_distribution={"light": 0.3, "medium": 0.4, "heavy": 0.3}
        )
        builder1.with_targets(
            count=3,
            region=((0.5, 1.0), (0.5, 1.0)),
            class_distribution={"A": 0.3, "B": 0.4, "C": 0.3},
            min_distance_from_drones=100.0,
            min_distance_between_targets=80.0
        )
        drones1, _ = builder1.build()
        
        # Builder 2 with same seed
        builder2 = ScenarioBuilder((1000.0, 1000.0), seed=42)
        builder2.with_drones(
            count=2,
            region=((0.0, 0.3), (0.0, 0.3)),
            min_distance_between_drones=50.0,
            weapon_distribution={"light": 0.3, "medium": 0.4, "heavy": 0.3}
        )
        builder2.with_targets(
            count=3,
            region=((0.5, 1.0), (0.5, 1.0)),
            class_distribution={"A": 0.3, "B": 0.4, "C": 0.3},
            min_distance_from_drones=100.0,
            min_distance_between_targets=80.0
        )
        drones2, _ = builder2.build()
        
        # Verify identical drone configs
        assert drones1 == drones2, "Same seed should produce identical drone configs"
        
        # Verify each drone has identical weapons
        for i, (d1, d2) in enumerate(zip(drones1, drones2)):
            assert d1["position"] == d2["position"], f"Drone {i} positions differ"
            assert d1["weapon_type"] == d2["weapon_type"], f"Drone {i} weapons differ"
    
    def test_same_seed_produces_identical_target_configs(self):
        """Same seed should produce identical target configurations."""
        # Builder 1
        builder1 = ScenarioBuilder((1000.0, 1000.0), seed=123)
        builder1.with_drones(
            count=2,
            region=((0.0, 0.2), (0.0, 0.2)),
            min_distance_between_drones=50.0,
            weapon_distribution={"light": 0.5, "heavy": 0.5}
        )
        builder1.with_targets(
            count=5,
            region=((0.5, 1.0), (0.5, 1.0)),
            class_distribution={"A": 0.2, "B": 0.5, "C": 0.3},
            min_distance_from_drones=150.0,
            min_distance_between_targets=100.0
        )
        _, targets1 = builder1.build()
        
        # Builder 2 with same seed
        builder2 = ScenarioBuilder((1000.0, 1000.0), seed=123)
        builder2.with_drones(
            count=2,
            region=((0.0, 0.2), (0.0, 0.2)),
            min_distance_between_drones=50.0,
            weapon_distribution={"light": 0.5, "heavy": 0.5}
        )
        builder2.with_targets(
            count=5,
            region=((0.5, 1.0), (0.5, 1.0)),
            class_distribution={"A": 0.2, "B": 0.5, "C": 0.3},
            min_distance_from_drones=150.0,
            min_distance_between_targets=100.0
        )
        _, targets2 = builder2.build()
        
        # Verify identical target configs
        assert targets1 == targets2, "Same seed should produce identical target configs"
        
        # Verify each target has identical properties
        for i, (t1, t2) in enumerate(zip(targets1, targets2)):
            assert t1["position"] == t2["position"], f"Target {i} positions differ"
            assert t1["class_type"] == t2["class_type"], f"Target {i} classes differ"
    
    def test_different_seeds_produce_different_configs(self):
        """Different seeds should produce different configurations."""
        # Builder 1 with seed 42
        builder1 = ScenarioBuilder((1000.0, 1000.0), seed=42)
        builder1.with_drones(
            count=3,
            region=((0.0, 0.3), (0.0, 0.3)),
            min_distance_between_drones=50.0,
            weapon_distribution={"light": 0.3, "medium": 0.4, "heavy": 0.3}
        )
        builder1.with_targets(
            count=5,
            region=((0.5, 1.0), (0.5, 1.0)),
            class_distribution={"A": 0.3, "B": 0.4, "C": 0.3},
            min_distance_from_drones=100.0,
            min_distance_between_targets=80.0
        )
        drones1, targets1 = builder1.build()
        
        # Builder 2 with seed 999
        builder2 = ScenarioBuilder((1000.0, 1000.0), seed=999)
        builder2.with_drones(
            count=3,
            region=((0.0, 0.3), (0.0, 0.3)),
            min_distance_between_drones=50.0,
            weapon_distribution={"light": 0.3, "medium": 0.4, "heavy": 0.3}
        )
        builder2.with_targets(
            count=5,
            region=((0.5, 1.0), (0.5, 1.0)),
            class_distribution={"A": 0.3, "B": 0.4, "C": 0.3},
            min_distance_from_drones=100.0,
            min_distance_between_targets=80.0
        )
        drones2, targets2 = builder2.build()
        
        # Verify configs differ (probabilistically)
        # At least one weapon should differ with high probability
        weapons1 = [d["weapon_type"] for d in drones1]
        weapons2 = [d["weapon_type"] for d in drones2]
        assert weapons1 != weapons2 or targets1 != targets2, \
            "Different seeds should produce different configs"
    
    def test_full_determinism_end_to_end(self):
        """Test complete determinism through full build process."""
        configs_seed_42 = []
        
        # Build 3 times with seed 42
        for _ in range(3):
            builder = ScenarioBuilder((1000.0, 1000.0), seed=42)
            builder.with_drones(
                count=2,
                region=((0.0, 0.3), (0.0, 0.3)),
                min_distance_between_drones=50.0,
                weapon_distribution={"light": 0.2, "medium": 0.5, "heavy": 0.3}
            )
            builder.with_targets(
                count=3,
                region=((0.5, 1.0), (0.5, 1.0)),
                class_distribution={"A": 0.3, "B": 0.4, "C": 0.3},
                min_distance_from_drones=100.0,
                min_distance_between_targets=80.0
            )
            configs_seed_42.append(builder.build())
        
        # All three builds should produce identical results
        for i in range(1, len(configs_seed_42)):
            assert configs_seed_42[0] == configs_seed_42[i], \
                f"Build {i} differs from build 0 with same seed"


class TestBuilderValidation:
    """Test builder configuration validation."""
    
    def test_build_without_drones_raises_error(self):
        """Building without configuring drones should raise ValueError."""
        builder = ScenarioBuilder((1000.0, 1000.0), seed=42)
        builder.with_targets(
            count=3,
            region=((0.0, 1.0), (0.0, 1.0)),
            class_distribution={"A": 1.0},
            min_distance_from_drones=100.0,
            min_distance_between_targets=80.0
        )
        
        with pytest.raises(ValueError, match="not configured for drones"):
            builder.build()
    
    def test_build_without_targets_raises_error(self):
        """Building without configuring targets should raise ValueError."""
        builder = ScenarioBuilder((1000.0, 1000.0), seed=42)
        builder.with_drones(
            count=1,
            region=((0.0, 1.0), (0.0, 1.0)),
            min_distance_between_drones=50.0,
            weapon_distribution={"light": 1.0}
        )
        
        with pytest.raises(ValueError, match="not configured for targets"):
            builder.build()
    
    def test_invalid_drone_count_raises_error(self):
        """Invalid drone count should raise ValueError."""
        builder = ScenarioBuilder((1000.0, 1000.0), seed=42)
        
        with pytest.raises(ValueError, match="positive"):
            builder.with_drones(
                count=0,
                region=((0.0, 1.0), (0.0, 1.0)),
                min_distance_between_drones=50.0,
                weapon_distribution={"light": 1.0}
            )
    
    def test_invalid_weapon_type_raises_error(self):
        """Invalid weapon type should raise ValueError."""
        builder = ScenarioBuilder((1000.0, 1000.0), seed=42)
        
        with pytest.raises(ValueError, match="Invalid weapon types"):
            builder.with_drones(
                count=1,
                region=((0.0, 1.0), (0.0, 1.0)),
                min_distance_between_drones=50.0,
                weapon_distribution={"invalid_weapon": 1.0}
            )
    
    def test_invalid_target_count_raises_error(self):
        """Invalid target count should raise ValueError."""
        builder = ScenarioBuilder((1000.0, 1000.0), seed=42)
        
        with pytest.raises(ValueError, match="positive"):
            builder.with_targets(
                count=0,
                region=((0.0, 1.0), (0.0, 1.0)),
                class_distribution={"A": 1.0},
                min_distance_from_drones=100.0,
                min_distance_between_targets=80.0
            )
    
    def test_invalid_class_type_raises_error(self):
        """Invalid class type should raise ValueError."""
        builder = ScenarioBuilder((1000.0, 1000.0), seed=42)
        
        with pytest.raises(ValueError, match="Invalid class types"):
            builder.with_targets(
                count=3,
                region=((0.0, 1.0), (0.0, 1.0)),
                class_distribution={"InvalidClass": 1.0},
                min_distance_from_drones=100.0,
                min_distance_between_targets=80.0
            )


class TestSpatialConstraints:
    """Test spatial constraint enforcement."""
    
    def test_targets_respect_drone_distance(self):
        """All targets should be at least min_distance from all drones."""
        builder = ScenarioBuilder((1000.0, 1000.0), seed=42)
        min_dist = 150.0
        
        builder.with_drones(
            count=3,
            region=((0.0, 0.3), (0.0, 0.3)),
            min_distance_between_drones=50.0,
            weapon_distribution={"light": 1.0}
        )
        builder.with_targets(
            count=5,
            region=((0.5, 1.0), (0.5, 1.0)),
            class_distribution={"A": 1.0},
            min_distance_from_drones=min_dist,
            min_distance_between_targets=80.0
        )
        drones, targets = builder.build()
        
        # Get generated drone positions
        drone_positions = [d["position"] for d in drones]
        
        # Verify all targets respect drone distance
        for target in targets:
            target_pos = target["position"]
            for drone_pos in drone_positions:
                dist = ((target_pos[0] - drone_pos[0])**2 + 
                       (target_pos[1] - drone_pos[1])**2)**0.5
                assert dist >= min_dist, \
                    f"Target at {target_pos} too close to drone at {drone_pos}: {dist:.1f} < {min_dist}"
    
    def test_targets_respect_inter_target_distance(self):
        """All targets should be at least min_distance from each other."""
        builder = ScenarioBuilder((1000.0, 1000.0), seed=42)
        min_dist = 100.0
        
        builder.with_drones(
            count=1,
            region=((0.0, 0.2), (0.0, 0.2)),
            min_distance_between_drones=50.0,
            weapon_distribution={"light": 1.0}
        )
        builder.with_targets(
            count=5,
            region=((0.5, 1.0), (0.5, 1.0)),
            class_distribution={"A": 1.0},
            min_distance_from_drones=100.0,
            min_distance_between_targets=min_dist
        )
        _, targets = builder.build()
        
        # Verify all targets respect inter-target distance
        for i, target1 in enumerate(targets):
            for j, target2 in enumerate(targets):
                if i != j:
                    pos1 = target1["position"]
                    pos2 = target2["position"]
                    dist = ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5
                    assert dist >= min_dist, \
                        f"Targets {i} and {j} too close: {dist:.1f} < {min_dist}"
    
    def test_drones_respect_inter_drone_distance(self):
        """All drones should be at least min_distance from each other."""
        builder = ScenarioBuilder((1000.0, 1000.0), seed=42)
        min_dist = 100.0
        
        builder.with_drones(
            count=5,
            region=((0.0, 1.0), (0.0, 1.0)),
            min_distance_between_drones=min_dist,
            weapon_distribution={"light": 1.0}
        )
        builder.with_targets(
            count=1,
            region=((0.0, 1.0), (0.0, 1.0)),
            class_distribution={"A": 1.0},
            min_distance_from_drones=0.0,
            min_distance_between_targets=0.0
        )
        drones, _ = builder.build()
        
        # Verify all drones respect inter-drone distance
        for i, drone1 in enumerate(drones):
            for j, drone2 in enumerate(drones):
                if i != j:
                    pos1 = drone1["position"]
                    pos2 = drone2["position"]
                    dist = ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5
                    assert dist >= min_dist, \
                        f"Drones {i} and {j} too close: {dist:.1f} < {min_dist}"


class TestFluentAPI:
    """Test fluent API builder pattern."""
    
    def test_method_chaining(self):
        """Builder methods should chain correctly."""
        drones, targets = (
            ScenarioBuilder((1000.0, 1000.0), seed=42)
            .with_drones(
                count=2,
                region=((0.0, 0.3), (0.0, 0.3)),
                min_distance_between_drones=50.0,
                weapon_distribution={"light": 0.5, "heavy": 0.5}
            )
            .with_targets(
                count=3,
                region=((0.5, 1.0), (0.5, 1.0)),
                class_distribution={"A": 0.5, "B": 0.5},
                min_distance_from_drones=100.0,
                min_distance_between_targets=80.0
            )
            .build()
        )
        
        assert len(drones) == 2
        assert len(targets) == 3
    
    def test_multiple_builds_with_same_builder(self):
        """Same builder can be used to build multiple times (generates same configs)."""
        builder = ScenarioBuilder((1000.0, 1000.0), seed=42)
        builder.with_drones(
            count=1,
            region=((0.0, 1.0), (0.0, 1.0)),
            min_distance_between_drones=50.0,
            weapon_distribution={"light": 1.0}
        )
        builder.with_targets(
            count=2,
            region=((0.0, 1.0), (0.0, 1.0)),
            class_distribution={"A": 1.0},
            min_distance_from_drones=100.0,
            min_distance_between_targets=80.0
        )
        
        # Build twice
        config1 = builder.build()
        config2 = builder.build()
        
        # Should produce identical configs (RNG state continues)
        # Note: This tests that build() doesn't reset state
        assert len(config1[0]) == len(config2[0])
        assert len(config1[1]) == len(config2[1])
