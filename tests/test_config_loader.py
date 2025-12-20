"""
Unit tests for configuration loader.

Tests cover:
- Valid config loading
- Missing file handling
- Invalid JSON handling
- Missing required fields
"""

import json
import os
import tempfile
import pytest

from tabula_drone.config import load_config, ScenarioConfig
from tabula_drone.config.config_loader import load_mappings, MappingsConfig


# Test mappings data - matches current attribute names
TEST_MAPPINGS_DATA = {
    "class_attribute_mapping": {
        "A": {"structural_integrity": 60.0, "envelope_integrity": 50.0, "utilities_lifesafety": 40.0},
        "B": {"structural_integrity": 80.0, "envelope_integrity": 70.0, "utilities_lifesafety": 60.0},
        "C": {"structural_integrity": 100.0, "envelope_integrity": 90.0, "utilities_lifesafety": 80.0},
    },
    "weapon_damage_profile_mapping": {
        "light": {"structural_integrity": 0.0, "envelope_integrity": 8.0, "utilities_lifesafety": 3.0},
        "medium": {"structural_integrity": 4.0, "envelope_integrity": 12.0, "utilities_lifesafety": 8.0},
        "heavy": {"structural_integrity": 18.0, "envelope_integrity": 15.0, "utilities_lifesafety": 14.0},
    }
}


def create_temp_config_with_mappings(config_data):
    """Create temp config file with accompanying mappings.json in same directory."""
    temp_dir = tempfile.mkdtemp()
    config_path = os.path.join(temp_dir, "scenario.json")
    mappings_path = os.path.join(temp_dir, "mappings.json")
    
    with open(config_path, "w") as f:
        json.dump(config_data, f)
    
    with open(mappings_path, "w") as f:
        json.dump(TEST_MAPPINGS_DATA, f)
    
    return config_path, temp_dir


def cleanup_temp_dir(temp_dir):
    """Clean up temp directory and all files."""
    import shutil
    shutil.rmtree(temp_dir)


class TestLoadConfigValid:
    """Tests for valid configuration loading."""
    
    def test_load_valid_config(self):
        """Valid config file loads successfully and returns ScenarioConfig."""
        config = load_config("config/scenario.json")
        
        assert isinstance(config, ScenarioConfig)
        assert config.seed == 42
        assert config.world.size == (1000.0, 1000.0)
        assert config.drones.count == 2
        assert config.drones.region == ((0.35, 0.65), (0.3, 0.4))
        assert config.drones.min_distance_between_drones == 50.0
        assert config.targets.count == 15
        assert config.environment.max_steps == 100
        assert config.environment.scenario_id == "random_policy_demo"
        assert config.policy.allow_noop is False
        assert config.execution.num_episodes == 1
        assert config.execution.verbose is True
        assert config.logging.output_dir == "logs/"
    
    def test_weapon_distribution_loaded(self):
        """Weapon distribution dictionary is loaded correctly."""
        config = load_config("config/scenario.json")
        
        assert "light" in config.drones.weapon_distribution
        assert "medium" in config.drones.weapon_distribution
        assert "heavy" in config.drones.weapon_distribution
        assert config.drones.weapon_distribution["light"] == 0.2
    
    def test_class_distribution_loaded(self):
        """Class distribution dictionary is loaded correctly."""
        config = load_config("config/scenario.json")
        
        assert "A" in config.targets.class_distribution
        assert "B" in config.targets.class_distribution
        assert "C" in config.targets.class_distribution


class TestLoadConfigMissingFile:
    """Tests for missing configuration file handling."""
    
    def test_missing_file_raises_file_not_found(self):
        """Missing config file raises FileNotFoundError with clear message."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_config("nonexistent/config.json")
        
        assert "Configuration file not found" in str(exc_info.value)
        assert "nonexistent/config.json" in str(exc_info.value)


class TestLoadConfigInvalidJson:
    """Tests for invalid JSON handling."""
    
    def test_invalid_json_raises_value_error(self):
        """Invalid JSON raises ValueError with clear message."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ invalid json }")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError) as exc_info:
                load_config(temp_path)
            
            assert "Invalid JSON" in str(exc_info.value)
        finally:
            os.unlink(temp_path)


class TestLoadConfigMissingFields:
    """Tests for missing required fields."""
    
    def test_missing_seed_raises_value_error(self):
        """Missing seed field raises ValueError."""
        config_data = {
            "world": {"size": [1000.0, 1000.0]},
            "drones": {"count": 1, "region": {"x_fraction": [0.0, 1.0], "y_fraction": [0.0, 1.0]}, "min_distance_between_drones": 50.0, "weapon_distribution": {"light": 1.0}},
            "targets": {"count": 1, "region": {"x_fraction": [0.0, 1.0], "y_fraction": [0.0, 1.0]}, "class_distribution": {"A": 1.0}, "min_distance_from_drones": 100.0, "min_distance_between_targets": 80.0},
            "environment": {"max_steps": 50, "scenario_id": "test"},
            "policy": {"allow_noop": False},
            "execution": {"num_episodes": 1, "verbose": True},
            "logging": {"output_dir": "logs/"}
        }
        
        temp_path, temp_dir = create_temp_config_with_mappings(config_data)
        
        try:
            with pytest.raises(ValueError) as exc_info:
                load_config(temp_path)
            
            assert "Missing required keys" in str(exc_info.value)
            assert "seed" in str(exc_info.value)
        finally:
            cleanup_temp_dir(temp_dir)
    
    def test_missing_world_section_raises_value_error(self):
        """Missing world section raises ValueError."""
        config_data = {
            "seed": 42,
            "drones": {"count": 1, "region": {"x_fraction": [0.0, 1.0], "y_fraction": [0.0, 1.0]}, "min_distance_between_drones": 50.0, "weapon_distribution": {"light": 1.0}},
            "targets": {"count": 1, "region": {"x_fraction": [0.0, 1.0], "y_fraction": [0.0, 1.0]}, "class_distribution": {"A": 1.0}, "min_distance_from_drones": 100.0, "min_distance_between_targets": 80.0},
            "environment": {"max_steps": 50, "scenario_id": "test"},
            "policy": {"allow_noop": False},
            "execution": {"num_episodes": 1, "verbose": True},
            "logging": {"output_dir": "logs/"}
        }
        
        temp_path, temp_dir = create_temp_config_with_mappings(config_data)
        
        try:
            with pytest.raises(ValueError) as exc_info:
                load_config(temp_path)
            
            assert "Missing required keys" in str(exc_info.value)
            assert "world" in str(exc_info.value)
        finally:
            cleanup_temp_dir(temp_dir)


class TestLoadConfigInvalidValues:
    """Tests for invalid configuration values."""
    
    def test_invalid_world_size_raises_value_error(self):
        """Invalid world size format raises ValueError."""
        config_data = {
            "seed": 42,
            "world": {"size": [1000.0]},  # Should be [width, height]
            "drones": {"count": 1, "region": {"x_fraction": [0.0, 1.0], "y_fraction": [0.0, 1.0]}, "min_distance_between_drones": 50.0, "weapon_distribution": {"light": 1.0}},
            "targets": {"count": 1, "region": {"x_fraction": [0.0, 1.0], "y_fraction": [0.0, 1.0]}, "class_distribution": {"A": 1.0}, "min_distance_from_drones": 100.0, "min_distance_between_targets": 80.0},
            "environment": {"max_steps": 50, "scenario_id": "test"},
            "policy": {"allow_noop": False},
            "execution": {"num_episodes": 1, "verbose": True},
            "logging": {"output_dir": "logs/"}
        }
        
        temp_path, temp_dir = create_temp_config_with_mappings(config_data)
        
        try:
            with pytest.raises(ValueError) as exc_info:
                load_config(temp_path)
            
            assert "world.size" in str(exc_info.value)
        finally:
            cleanup_temp_dir(temp_dir)
    
    def test_invalid_target_count_raises_value_error(self):
        """Non-positive target count raises ValueError."""
        config_data = {
            "seed": 42,
            "world": {"size": [1000.0, 1000.0]},
            "drones": {"count": 1, "region": {"x_fraction": [0.0, 1.0], "y_fraction": [0.0, 1.0]}, "min_distance_between_drones": 50.0, "weapon_distribution": {"light": 1.0}},
            "targets": {"count": 0, "region": {"x_fraction": [0.0, 1.0], "y_fraction": [0.0, 1.0]}, "class_distribution": {"A": 1.0}, "min_distance_from_drones": 100.0, "min_distance_between_targets": 80.0},
            "environment": {"max_steps": 50, "scenario_id": "test"},
            "policy": {"allow_noop": False},
            "execution": {"num_episodes": 1, "verbose": True},
            "logging": {"output_dir": "logs/"}
        }
        
        temp_path, temp_dir = create_temp_config_with_mappings(config_data)
        
        try:
            with pytest.raises(ValueError) as exc_info:
                load_config(temp_path)
            
            assert "targets.count" in str(exc_info.value)
            assert "positive integer" in str(exc_info.value)
        finally:
            cleanup_temp_dir(temp_dir)


class TestLoadMappings:
    """Tests for mappings configuration loading."""
    
    def test_load_valid_mappings(self):
        """Valid mappings file loads successfully."""
        mappings = load_mappings("config/mappings.json")
        
        assert isinstance(mappings, MappingsConfig)
        assert "A" in mappings.class_attribute_mapping
        assert "B" in mappings.class_attribute_mapping
        assert "C" in mappings.class_attribute_mapping
        assert "light" in mappings.weapon_damage_profile_mapping
        assert "medium" in mappings.weapon_damage_profile_mapping
        assert "heavy" in mappings.weapon_damage_profile_mapping
    
    def test_mappings_values_correct(self):
        """Mappings contain correct values."""
        mappings = load_mappings("config/mappings.json")
        
        assert mappings.class_attribute_mapping["A"]["structural_integrity"] == 60.0
        assert mappings.class_attribute_mapping["A"]["envelope_integrity"] == 50.0
        assert mappings.weapon_damage_profile_mapping["heavy"]["structural_integrity"] == 18.0
    
    def test_missing_mappings_file_raises_error(self):
        """Missing mappings file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_mappings("nonexistent/mappings.json")
        
        assert "Mappings file not found" in str(exc_info.value)
    
    def test_missing_class_attribute_mapping_raises_error(self):
        """Missing class_attribute_mapping raises ValueError."""
        temp_dir = tempfile.mkdtemp()
        mappings_path = os.path.join(temp_dir, "mappings.json")
        
        with open(mappings_path, "w") as f:
            json.dump({"weapon_damage_profile_mapping": {"light": {"hp": 10.0}}}, f)
        
        try:
            with pytest.raises(ValueError) as exc_info:
                load_mappings(mappings_path)
            
            assert "Missing required keys" in str(exc_info.value)
            assert "class_attribute_mapping" in str(exc_info.value)
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_missing_weapon_damage_profile_mapping_raises_error(self):
        """Missing weapon_damage_profile_mapping raises ValueError."""
        temp_dir = tempfile.mkdtemp()
        mappings_path = os.path.join(temp_dir, "mappings.json")
        
        with open(mappings_path, "w") as f:
            json.dump({"class_attribute_mapping": {"A": {"hp": 100.0}}}, f)
        
        try:
            with pytest.raises(ValueError) as exc_info:
                load_mappings(mappings_path)
            
            assert "Missing required keys" in str(exc_info.value)
            assert "weapon_damage_profile_mapping" in str(exc_info.value)
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_empty_class_mapping_raises_error(self):
        """Empty class_attribute_mapping raises ValueError."""
        temp_dir = tempfile.mkdtemp()
        mappings_path = os.path.join(temp_dir, "mappings.json")
        
        with open(mappings_path, "w") as f:
            json.dump({
                "class_attribute_mapping": {},
                "weapon_damage_profile_mapping": {"light": {"hp": 10.0}}
            }, f)
        
        try:
            with pytest.raises(ValueError) as exc_info:
                load_mappings(mappings_path)
            
            assert "must not be empty" in str(exc_info.value)
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_cross_validation_invalid_weapon_attribute(self):
        """Weapon attributes not in target attributes raises ValueError."""
        temp_dir = tempfile.mkdtemp()
        mappings_path = os.path.join(temp_dir, "mappings.json")
        
        with open(mappings_path, "w") as f:
            json.dump({
                "class_attribute_mapping": {"A": {"structural_integrity": 100.0}},
                "weapon_damage_profile_mapping": {"light": {"envelope_integrity": 10.0}}
            }, f)
        
        try:
            with pytest.raises(ValueError) as exc_info:
                load_mappings(mappings_path)
            
            assert "not defined in any target class" in str(exc_info.value)
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_config_includes_mappings(self):
        """load_config includes mappings in returned ScenarioConfig."""
        config = load_config("config/scenario.json")
        
        assert config.mappings is not None
        assert isinstance(config.mappings, MappingsConfig)
        assert "A" in config.mappings.class_attribute_mapping
        assert "light" in config.mappings.weapon_damage_profile_mapping
