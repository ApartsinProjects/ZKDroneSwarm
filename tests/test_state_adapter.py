"""
Unit tests for viewer/state_adapter.py
"""

import unittest
from viewer.state_adapter import extract_hp_history, extract_initial_state


class TestExtractHpHistory(unittest.TestCase):
    """Tests for extract_hp_history function."""

    def test_valid_episode_data(self):
        """extract_hp_history returns correct total HP per step."""
        episode_data = {
            "steps": [
                {"info": {"target_hps": [100.0, 150.0, 200.0]}},
                {"info": {"target_hps": [80.0, 150.0, 200.0]}},
                {"info": {"target_hps": [80.0, 100.0, 150.0]}},
            ]
        }
        result = extract_hp_history(episode_data)
        self.assertEqual(result, [450.0, 430.0, 330.0])

    def test_empty_steps(self):
        """extract_hp_history returns empty list when steps is empty."""
        episode_data = {"steps": []}
        result = extract_hp_history(episode_data)
        self.assertEqual(result, [])

    def test_missing_steps_key(self):
        """extract_hp_history returns empty list when steps key is missing."""
        episode_data = {}
        result = extract_hp_history(episode_data)
        self.assertEqual(result, [])

    def test_missing_target_hps(self):
        """extract_hp_history stops at step with missing target_hps."""
        episode_data = {
            "steps": [
                {"info": {"target_hps": [100.0, 150.0]}},
                {"info": {}},
                {"info": {"target_hps": [50.0, 100.0]}},
            ]
        }
        result = extract_hp_history(episode_data)
        self.assertEqual(result, [250.0])

    def test_missing_info_key(self):
        """extract_hp_history stops at step with missing info key."""
        episode_data = {
            "steps": [
                {"info": {"target_hps": [100.0]}},
                {},
            ]
        }
        result = extract_hp_history(episode_data)
        self.assertEqual(result, [100.0])

    def test_single_target(self):
        """extract_hp_history works with single target."""
        episode_data = {
            "steps": [
                {"info": {"target_hps": [100.0]}},
                {"info": {"target_hps": [50.0]}},
                {"info": {"target_hps": [0.0]}},
            ]
        }
        result = extract_hp_history(episode_data)
        self.assertEqual(result, [100.0, 50.0, 0.0])


class TestExtractInitialStateHpHistory(unittest.TestCase):
    """Tests for hp_history in extract_initial_state."""

    def test_hp_history_included(self):
        """extract_initial_state includes hp_history key."""
        episode_data = {
            "scenario": {
                "drone_positions": [[0, 0]],
                "target_positions": [[100, 100]],
            },
            "steps": [
                {"info": {"target_hps": [100.0]}},
            ]
        }
        result = extract_initial_state(episode_data)
        self.assertIn("hp_history", result)
        self.assertEqual(result["hp_history"], [100.0])

    def test_hp_history_empty_when_no_steps(self):
        """extract_initial_state returns empty hp_history when no steps."""
        episode_data = {
            "scenario": {
                "drone_positions": [[0, 0]],
                "target_positions": [[100, 100]],
            }
        }
        result = extract_initial_state(episode_data)
        self.assertIn("hp_history", result)
        self.assertEqual(result["hp_history"], [])


if __name__ == "__main__":
    unittest.main()
