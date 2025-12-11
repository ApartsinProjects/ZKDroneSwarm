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


class TestLoadConfigValid:
    """Tests for valid configuration loading."""
    
    def test_load_valid_config(self):
        """Valid config file loads successfully and returns ScenarioConfig."""
        config = load_config("config/scenario.json")
        
        assert isinstance(config, ScenarioConfig)
        assert config.seed == 42
        assert config.world.size == (1000.0, 1000.0)
        assert config.drones.count == 2
        assert config.drones.region == ((0.05, 0.25), (0.03, 0.5))
        assert config.drones.min_distance_between_drones == 50.0
        assert config.targets.count == 15
        assert config.environment.max_steps == 50
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
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError) as exc_info:
                load_config(temp_path)
            
            assert "Missing required keys" in str(exc_info.value)
            assert "seed" in str(exc_info.value)
        finally:
            os.unlink(temp_path)
    
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
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError) as exc_info:
                load_config(temp_path)
            
            assert "Missing required keys" in str(exc_info.value)
            assert "world" in str(exc_info.value)
        finally:
            os.unlink(temp_path)


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
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError) as exc_info:
                load_config(temp_path)
            
            assert "world.size" in str(exc_info.value)
        finally:
            os.unlink(temp_path)
    
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
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError) as exc_info:
                load_config(temp_path)
            
            assert "targets.count" in str(exc_info.value)
            assert "positive integer" in str(exc_info.value)
        finally:
            os.unlink(temp_path)
