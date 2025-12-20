"""
Unit tests for EpisodeLogger.

Tests logger initialization, episode capture, step logging,
and JSON serialization/round-trip validation.
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

from tabula_drone.logging import EpisodeLogger


class TestEpisodeLoggerInit(unittest.TestCase):
    """Tests for EpisodeLogger initialization."""
    
    def test_init_default_output_dir(self):
        """Logger initializes with default output directory."""
        logger = EpisodeLogger()
        self.assertEqual(logger.output_dir, "logs/")
    
    def test_init_custom_output_dir(self):
        """Logger initializes with custom output directory."""
        logger = EpisodeLogger(output_dir="custom_logs/")
        self.assertEqual(logger.output_dir, "custom_logs/")
    
    def test_init_empty_state(self):
        """Logger initializes with empty state."""
        logger = EpisodeLogger()
        self.assertIsNone(logger._episode_data)
        self.assertEqual(logger._steps, [])


class TestEpisodeLoggerStartEpisode(unittest.TestCase):
    """Tests for start_episode method."""
    
    def setUp(self):
        """Create mock environment for testing."""
        self.mock_env = MagicMock()
        self.mock_env.num_drones = 2
        self.mock_env.num_targets = 3
        
        # Mock drone objects
        mock_drone_0 = MagicMock()
        mock_drone_0.id = "drone_0"
        mock_drone_0.position = (100.0, 100.0)
        mock_drone_0.weapon_type = "medium"
        
        mock_drone_1 = MagicMock()
        mock_drone_1.id = "drone_1"
        mock_drone_1.position = (200.0, 200.0)
        mock_drone_1.weapon_type = "heavy"
        
        self.mock_env.drones = [mock_drone_0, mock_drone_1]
        
        # Mock target objects
        mock_target_0 = MagicMock()
        mock_target_0.position = (500.0, 500.0)
        mock_target_0.class_type = "A"
        
        mock_target_1 = MagicMock()
        mock_target_1.position = (600.0, 600.0)
        mock_target_1.class_type = "B"
        
        mock_target_2 = MagicMock()
        mock_target_2.position = (700.0, 700.0)
        mock_target_2.class_type = "C"
        
        self.mock_env.targets = [mock_target_0, mock_target_1, mock_target_2]
        
        # Config attributes for _build_config_snapshot
        self.mock_env.world_size = (1000.0, 1000.0)
        self.mock_env.max_steps = 100
        self.mock_env.scenario_id = "test_scenario"
        self.mock_env.class_attribute_mapping = {"A": {"structural_integrity": 60.0, "envelope_integrity": 50.0, "utilities_lifesafety": 40.0}, "B": {"structural_integrity": 80.0, "envelope_integrity": 70.0, "utilities_lifesafety": 60.0}, "C": {"structural_integrity": 100.0, "envelope_integrity": 90.0, "utilities_lifesafety": 80.0}}
        
        self.reset_info = {
            "step_index": 0,
            "scenario_id": "test_scenario",
        }
    
    def test_start_episode_initializes_data(self):
        """start_episode initializes episode data structure."""
        logger = EpisodeLogger()
        logger.start_episode(self.mock_env, self.reset_info, seed=42)
        
        self.assertIsNotNone(logger._episode_data)
        self.assertIsNotNone(logger._episode_id)
        self.assertIsNotNone(logger._timestamp)
    
    def test_start_episode_captures_version(self):
        """start_episode captures schema version."""
        logger = EpisodeLogger()
        logger.start_episode(self.mock_env, self.reset_info, seed=42)
        
        self.assertEqual(logger._episode_data["version"], "1.1")
    
    def test_start_episode_captures_seed(self):
        """start_episode captures random seed."""
        logger = EpisodeLogger()
        logger.start_episode(self.mock_env, self.reset_info, seed=42)
        
        self.assertEqual(logger._episode_data["rng_seed"], 42)
    
    def test_start_episode_captures_scenario(self):
        """start_episode captures scenario snapshot."""
        logger = EpisodeLogger()
        logger.start_episode(self.mock_env, self.reset_info, seed=42)
        
        scenario = logger._episode_data["scenario"]
        self.assertEqual(scenario["num_drones"], 2)
        self.assertEqual(scenario["num_targets"], 3)
        self.assertEqual(scenario["drone_positions"], [[100.0, 100.0], [200.0, 200.0]])
        self.assertEqual(scenario["target_positions"], [[500.0, 500.0], [600.0, 600.0], [700.0, 700.0]])
        self.assertEqual(scenario["weapon_assignments"], {"drone_0": "medium", "drone_1": "heavy"})
        self.assertEqual(scenario["target_classes"], ["A", "B", "C"])
    
    def test_start_episode_resets_steps(self):
        """start_episode resets steps list."""
        logger = EpisodeLogger()
        logger._steps = [{"dummy": "step"}]
        
        logger.start_episode(self.mock_env, self.reset_info, seed=42)
        
        self.assertEqual(logger._steps, [])
    
    def test_start_episode_captures_config(self):
        """start_episode captures config snapshot."""
        logger = EpisodeLogger()
        logger.start_episode(self.mock_env, self.reset_info, seed=42)
        
        config = logger._episode_data["config"]
        self.assertEqual(config["world_size"], [1000.0, 1000.0])
        self.assertEqual(config["max_steps"], 100)
        self.assertEqual(config["scenario_id"], "test_scenario")
        self.assertEqual(config["class_attribute_mapping"], {"A": {"structural_integrity": 60.0, "envelope_integrity": 50.0, "utilities_lifesafety": 40.0}, "B": {"structural_integrity": 80.0, "envelope_integrity": 70.0, "utilities_lifesafety": 60.0}, "C": {"structural_integrity": 100.0, "envelope_integrity": 90.0, "utilities_lifesafety": 80.0}})


class TestEpisodeLoggerLogStep(unittest.TestCase):
    """Tests for log_step method."""
    
    def setUp(self):
        """Create logger with started episode."""
        self.logger = EpisodeLogger()
        
        # Create minimal mock env
        mock_env = MagicMock()
        mock_env.num_drones = 1
        mock_env.num_targets = 1
        mock_drone = MagicMock()
        mock_drone.id = "drone_0"
        mock_drone.position = (0.0, 0.0)
        mock_drone.weapon_type = "light"
        mock_env.drones = [mock_drone]
        mock_target = MagicMock()
        mock_target.position = (100.0, 100.0)
        mock_target.class_type = "A"
        mock_env.targets = [mock_target]
        
        # Config attributes
        mock_env.world_size = (500.0, 500.0)
        mock_env.max_steps = 50
        mock_env.scenario_id = "test"
        mock_env.class_hp_mapping = {"A": 100.0}
        
        self.logger.start_episode(mock_env, {}, seed=1)
    
    def test_log_step_appends_record(self):
        """log_step appends step record to list."""
        self.logger.log_step(
            step_num=1,
            actions={"drone_0": 1},
            rewards={"drone_0": 0.0},
            terminated=False,
            truncated=False,
            info={"target_hps": [100.0], "target_active": [True], "ammo_used": {"drone_0": 1}}
        )
        
        self.assertEqual(len(self.logger._steps), 1)
    
    def test_log_step_captures_action(self):
        """log_step captures action data."""
        self.logger.log_step(
            step_num=1,
            actions={"drone_0": 1},
            rewards={"drone_0": 1.0},
            terminated=False,
            truncated=False,
            info={"target_hps": [0.0], "target_active": [False], "ammo_used": {"drone_0": 1}}
        )
        
        step = self.logger._steps[0]
        self.assertEqual(step["step_num"], 1)
        self.assertEqual(step["action"], {"drone_0": 1})
        self.assertEqual(step["reward"], {"drone_0": 1.0})
    
    def test_log_step_captures_termination_flags(self):
        """log_step captures terminated and truncated flags."""
        self.logger.log_step(
            step_num=1,
            actions={"drone_0": 1},
            rewards={"drone_0": 1.0},
            terminated=True,
            truncated=False,
            info={"target_hps": [0.0], "target_active": [False], "ammo_used": {"drone_0": 1}}
        )
        
        step = self.logger._steps[0]
        self.assertTrue(step["terminated"])
        self.assertFalse(step["truncated"])
    
    def test_log_step_captures_info_subset(self):
        """log_step captures relevant info fields."""
        self.logger.log_step(
            step_num=1,
            actions={"drone_0": 1},
            rewards={"drone_0": 1.0},
            terminated=True,
            truncated=False,
            info={
                "target_hps": [0.0],
                "target_active": [False],
                "ammo_used": {"drone_0": 1},
                "overkill": {0: 10.0}
            }
        )
        
        step_info = self.logger._steps[0]["info"]
        self.assertEqual(step_info["target_hps"], [0.0])
        self.assertEqual(step_info["target_active"], [False])
        self.assertEqual(step_info["ammo_used"], {"drone_0": 1})
        self.assertEqual(step_info["overkill"], {0: 10.0})
    
    def test_log_step_multiple_steps(self):
        """log_step accumulates multiple steps."""
        for i in range(5):
            self.logger.log_step(
                step_num=i + 1,
                actions={"drone_0": 1},
                rewards={"drone_0": 0.0},
                terminated=False,
                truncated=False,
                info={"target_hps": [100.0], "target_active": [True], "ammo_used": {"drone_0": i + 1}}
            )
        
        self.assertEqual(len(self.logger._steps), 5)
        self.assertEqual(self.logger._steps[0]["step_num"], 1)
        self.assertEqual(self.logger._steps[4]["step_num"], 5)


class TestEpisodeLoggerEndEpisode(unittest.TestCase):
    """Tests for end_episode method."""
    
    def setUp(self):
        """Create logger with started episode and logged steps."""
        self.logger = EpisodeLogger()
        
        mock_env = MagicMock()
        mock_env.num_drones = 1
        mock_env.num_targets = 2
        mock_drone = MagicMock()
        mock_drone.id = "drone_0"
        mock_drone.position = (0.0, 0.0)
        mock_drone.weapon_type = "heavy"
        mock_env.drones = [mock_drone]
        mock_target_0 = MagicMock()
        mock_target_0.position = (100.0, 100.0)
        mock_target_0.class_type = "A"
        mock_target_1 = MagicMock()
        mock_target_1.position = (200.0, 200.0)
        mock_target_1.class_type = "B"
        mock_env.targets = [mock_target_0, mock_target_1]
        
        # Config attributes
        mock_env.world_size = (500.0, 500.0)
        mock_env.max_steps = 50
        mock_env.scenario_id = "test"
        mock_env.class_hp_mapping = {"A": 100.0, "B": 150.0}
        
        self.logger.start_episode(mock_env, {}, seed=1)
        
        # Log some steps
        self.logger.log_step(1, {"drone_0": 1}, {"drone_0": 1.0}, False, False,
                            {"target_hps": [0.0, 150.0], "target_active": [False, True], "ammo_used": {"drone_0": 1}})
        self.logger.log_step(2, {"drone_0": 2}, {"drone_0": 1.0}, True, False,
                            {"target_hps": [0.0, 0.0], "target_active": [False, False], "ammo_used": {"drone_0": 2}})
    
    def test_end_episode_builds_summary(self):
        """end_episode builds summary in episode data."""
        self.logger.end_episode({"drone_0": 2.0}, "all_targets_neutralized")
        
        self.assertIsNotNone(self.logger._episode_data["summary"])
    
    def test_end_episode_captures_total_steps(self):
        """end_episode captures total step count."""
        self.logger.end_episode({"drone_0": 2.0}, "all_targets_neutralized")
        
        summary = self.logger._episode_data["summary"]
        self.assertEqual(summary["total_steps"], 2)
    
    def test_end_episode_captures_total_reward(self):
        """end_episode captures total rewards."""
        self.logger.end_episode({"drone_0": 2.0}, "all_targets_neutralized")
        
        summary = self.logger._episode_data["summary"]
        self.assertEqual(summary["total_reward"], {"drone_0": 2.0})
    
    def test_end_episode_captures_termination_reason(self):
        """end_episode captures termination reason."""
        self.logger.end_episode({"drone_0": 2.0}, "all_targets_neutralized")
        
        summary = self.logger._episode_data["summary"]
        self.assertEqual(summary["termination_reason"], "all_targets_neutralized")
    
    def test_end_episode_computes_success(self):
        """end_episode computes success flag."""
        self.logger.end_episode({"drone_0": 2.0}, "all_targets_neutralized")
        
        summary = self.logger._episode_data["summary"]
        self.assertTrue(summary["success"])
    
    def test_end_episode_computes_metrics(self):
        """end_episode computes metrics from final state."""
        self.logger.end_episode({"drone_0": 2.0}, "all_targets_neutralized")
        
        metrics = self.logger._episode_data["summary"]["metrics"]
        self.assertEqual(metrics["targets_destroyed"], 2)
        self.assertEqual(metrics["total_ammo_used"], 2)


class TestEpisodeLoggerSave(unittest.TestCase):
    """Tests for save method and JSON output."""
    
    def setUp(self):
        """Create temp directory and logger with complete episode."""
        self.temp_dir = tempfile.mkdtemp()
        self.logger = EpisodeLogger(output_dir=self.temp_dir)
        
        mock_env = MagicMock()
        mock_env.num_drones = 1
        mock_env.num_targets = 1
        mock_drone = MagicMock()
        mock_drone.id = "drone_0"
        mock_drone.position = (0.0, 0.0)
        mock_drone.weapon_type = "light"
        mock_env.drones = [mock_drone]
        mock_target = MagicMock()
        mock_target.position = (100.0, 100.0)
        mock_target.class_type = "A"
        mock_env.targets = [mock_target]
        
        # Config attributes
        mock_env.world_size = (1000.0, 1000.0)
        mock_env.max_steps = 100
        mock_env.scenario_id = "save_test"
        mock_env.class_hp_mapping = {"A": 100.0}
        
        self.logger.start_episode(mock_env, {}, seed=42)
        self.logger.log_step(1, {"drone_0": 1}, {"drone_0": 1.0}, True, False,
                            {"target_hps": [0.0], "target_active": [False], "ammo_used": {"drone_0": 1}})
        self.logger.end_episode({"drone_0": 1.0}, "all_targets_neutralized")
    
    def tearDown(self):
        """Clean up temp directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_save_returns_filepath(self):
        """save returns filepath of created file."""
        filepath = self.logger.save()
        
        self.assertTrue(filepath.startswith(self.temp_dir))
        self.assertTrue(filepath.endswith(".json"))
    
    def test_save_creates_file(self):
        """save creates JSON file."""
        filepath = self.logger.save()
        
        self.assertTrue(os.path.exists(filepath))
    
    def test_save_creates_directory(self):
        """save creates output directory if not exists."""
        new_dir = os.path.join(self.temp_dir, "nested", "logs")
        logger = EpisodeLogger(output_dir=new_dir)
        
        # Reuse episode data
        logger._episode_data = self.logger._episode_data
        logger._episode_id = self.logger._episode_id
        logger._steps = self.logger._steps
        
        filepath = logger.save()
        
        self.assertTrue(os.path.exists(new_dir))
        self.assertTrue(os.path.exists(filepath))
    
    def test_save_raises_without_start(self):
        """save raises ValueError if start_episode not called."""
        logger = EpisodeLogger()
        
        with self.assertRaises(ValueError) as ctx:
            logger.save()
        
        self.assertIn("start_episode", str(ctx.exception))
    
    def test_save_produces_valid_json(self):
        """save produces valid JSON file."""
        filepath = self.logger.save()
        
        with open(filepath) as f:
            data = json.load(f)
        
        self.assertIsInstance(data, dict)
    
    def test_save_json_has_required_keys(self):
        """Saved JSON has all required top-level keys."""
        filepath = self.logger.save()
        
        with open(filepath) as f:
            data = json.load(f)
        
        required_keys = ["version", "episode_id", "timestamp", "rng_seed", "config", "scenario", "steps", "summary"]
        for key in required_keys:
            self.assertIn(key, data, f"Missing key: {key}")
    
    def test_save_json_roundtrip(self):
        """Saved JSON can be loaded and contains correct data."""
        filepath = self.logger.save()
        
        with open(filepath) as f:
            data = json.load(f)
        
        # Verify structure
        self.assertEqual(data["version"], "1.1")
        self.assertEqual(data["rng_seed"], 42)
        self.assertEqual(len(data["steps"]), 1)
        self.assertEqual(data["summary"]["total_steps"], 1)
        self.assertTrue(data["summary"]["success"])
        
        # Verify scenario
        self.assertEqual(data["scenario"]["num_drones"], 1)
        self.assertEqual(data["scenario"]["num_targets"], 1)
        self.assertEqual(data["scenario"]["drone_positions"], [[0.0, 0.0]])
        self.assertEqual(data["scenario"]["target_classes"], ["A"])


if __name__ == "__main__":
    unittest.main()
