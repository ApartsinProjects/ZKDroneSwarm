"""
Unit tests for OracleTimeToKillPolicy.

Tests cover:
- Hits-to-kill calculation
- Target selection (minimum hits)
- Tie-breaking behavior
- Edge cases (no active targets, unkillable targets)
"""

import numpy as np
import pytest

from tabula_drone.policies.oracle_policy import OracleTimeToKillPolicy


class TestEstimatedHitsToKill:
    """Tests for _estimated_hits_to_kill method."""
    
    def test_single_attribute_exact_division(self):
        """Single attribute with exact division returns correct hits."""
        policy = OracleTimeToKillPolicy(
            weapon_damage_profile={"armor": 10.0},
            seed=42,
        )
        # 30 armor / 10 damage = 3 hits
        hits = policy._estimated_hits_to_kill({"armor": 30.0})
        assert hits == 3.0
    
    def test_single_attribute_with_remainder(self):
        """Single attribute with remainder uses ceiling."""
        policy = OracleTimeToKillPolicy(
            weapon_damage_profile={"armor": 10.0},
            seed=42,
        )
        # 25 armor / 10 damage = 2.5 -> ceil = 3 hits
        hits = policy._estimated_hits_to_kill({"armor": 25.0})
        assert hits == 3.0
    
    def test_multiple_attributes_takes_max(self):
        """Multiple attributes returns max across all."""
        policy = OracleTimeToKillPolicy(
            weapon_damage_profile={"armor": 10.0, "shields": 5.0},
            seed=42,
        )
        # armor: 20/10 = 2 hits, shields: 25/5 = 5 hits -> max = 5
        hits = policy._estimated_hits_to_kill({"armor": 20.0, "shields": 25.0})
        assert hits == 5.0
    
    def test_zero_remaining_attribute_ignored(self):
        """Attribute with zero remaining is ignored."""
        policy = OracleTimeToKillPolicy(
            weapon_damage_profile={"armor": 10.0, "shields": 5.0},
            seed=42,
        )
        # armor: 0 (ignored), shields: 15/5 = 3 hits
        hits = policy._estimated_hits_to_kill({"armor": 0.0, "shields": 15.0})
        assert hits == 3.0
    
    def test_zero_damage_returns_infinity(self):
        """Zero damage for non-zero attribute returns infinity."""
        policy = OracleTimeToKillPolicy(
            weapon_damage_profile={"armor": 0.0},
            seed=42,
        )
        hits = policy._estimated_hits_to_kill({"armor": 30.0})
        assert hits == float("inf")
    
    def test_missing_damage_profile_returns_infinity(self):
        """Attribute not in damage profile returns infinity."""
        policy = OracleTimeToKillPolicy(
            weapon_damage_profile={"armor": 10.0},
            seed=42,
        )
        # shields not in weapon profile
        hits = policy._estimated_hits_to_kill({"shields": 30.0})
        assert hits == float("inf")
    
    def test_all_depleted_returns_zero(self):
        """All attributes depleted returns zero hits."""
        policy = OracleTimeToKillPolicy(
            weapon_damage_profile={"armor": 10.0},
            seed=42,
        )
        hits = policy._estimated_hits_to_kill({"armor": 0.0})
        assert hits == 0.0


class TestSelectAction:
    """Tests for select_action method."""
    
    def create_observation(self, active_states):
        """Create observation array from active states list."""
        obs = []
        for i, active in enumerate(active_states):
            obs.extend([100.0 + i * 10, 200.0 + i * 10, 1.0 if active else 0.0])
        return np.array(obs, dtype=np.float32)
    
    def test_selects_minimum_hits_target(self):
        """Selects target with minimum hits-to-kill."""
        policy = OracleTimeToKillPolicy(
            weapon_damage_profile={"hp": 10.0},
            seed=42,
        )
        obs = self.create_observation([True, True, True])
        targets_state = [
            {"hp": 50.0},  # 5 hits
            {"hp": 20.0},  # 2 hits <- minimum
            {"hp": 30.0},  # 3 hits
        ]
        action = policy.select_action(obs, num_targets=3, targets_state=targets_state)
        assert action == 2  # Target index 1, 1-indexed action
    
    def test_ignores_inactive_targets(self):
        """Does not select inactive targets even if lower hits."""
        policy = OracleTimeToKillPolicy(
            weapon_damage_profile={"hp": 10.0},
            seed=42,
        )
        obs = self.create_observation([True, False, True])  # Target 1 inactive
        targets_state = [
            {"hp": 50.0},  # 5 hits
            {"hp": 10.0},  # 1 hit but INACTIVE
            {"hp": 30.0},  # 3 hits <- minimum among active
        ]
        action = policy.select_action(obs, num_targets=3, targets_state=targets_state)
        assert action == 3  # Target index 2, 1-indexed action
    
    def test_no_active_targets_returns_noop(self):
        """Returns NoOp when no active targets."""
        policy = OracleTimeToKillPolicy(
            weapon_damage_profile={"hp": 10.0},
            seed=42,
            allow_noop=True,
        )
        obs = self.create_observation([False, False, False])
        targets_state = [{"hp": 10.0}, {"hp": 20.0}, {"hp": 30.0}]
        action = policy.select_action(obs, num_targets=3, targets_state=targets_state)
        assert action == 0
    
    def test_all_unkillable_with_noop_returns_noop(self):
        """Returns NoOp when all targets unkillable and allow_noop=True."""
        policy = OracleTimeToKillPolicy(
            weapon_damage_profile={"armor": 10.0},  # No "shields" damage
            seed=42,
            allow_noop=True,
        )
        obs = self.create_observation([True, True])
        targets_state = [
            {"shields": 50.0},  # unkillable
            {"shields": 30.0},  # unkillable
        ]
        action = policy.select_action(obs, num_targets=2, targets_state=targets_state)
        assert action == 0
    
    def test_all_unkillable_without_noop_selects_random(self):
        """Selects random active target when all unkillable and allow_noop=False."""
        policy = OracleTimeToKillPolicy(
            weapon_damage_profile={"armor": 10.0},
            seed=42,
            allow_noop=False,
        )
        obs = self.create_observation([True, True])
        targets_state = [
            {"shields": 50.0},
            {"shields": 30.0},
        ]
        action = policy.select_action(obs, num_targets=2, targets_state=targets_state)
        assert action in [1, 2]  # Must fire at one of them
    
    def test_tie_break_is_deterministic_with_seed(self):
        """Tie-breaking is deterministic with same seed."""
        targets_state = [
            {"hp": 20.0},  # 2 hits
            {"hp": 20.0},  # 2 hits (tie)
        ]
        
        actions = []
        for _ in range(5):
            policy = OracleTimeToKillPolicy(
                weapon_damage_profile={"hp": 10.0},
                seed=42,
            )
            obs = self.create_observation([True, True])
            action = policy.select_action(obs, num_targets=2, targets_state=targets_state)
            actions.append(action)
        
        # All actions should be the same with same seed
        assert all(a == actions[0] for a in actions)


class TestSelectActions:
    """Tests for select_actions method."""
    
    def create_observation(self, active_states):
        """Create observation array from active states list."""
        obs = []
        for i, active in enumerate(active_states):
            obs.extend([100.0 + i * 10, 200.0 + i * 10, 1.0 if active else 0.0])
        return np.array(obs, dtype=np.float32)
    
    def test_returns_dict_of_actions(self):
        """Returns dictionary with action per agent."""
        policy = OracleTimeToKillPolicy(
            weapon_damage_profile={"hp": 10.0},
            seed=42,
        )
        obs = self.create_observation([True, True])
        observations = {
            "drone_0": obs,
            "drone_1": obs,
        }
        targets_state = [{"hp": 20.0}, {"hp": 30.0}]
        
        actions = policy.select_actions(observations, num_targets=2, targets_state=targets_state)
        
        assert isinstance(actions, dict)
        assert "drone_0" in actions
        assert "drone_1" in actions
        assert all(isinstance(a, int) for a in actions.values())
    
    def test_all_agents_select_same_best_target(self):
        """All agents select the same minimum-hits target."""
        policy = OracleTimeToKillPolicy(
            weapon_damage_profile={"hp": 10.0},
            seed=42,
        )
        obs = self.create_observation([True, True, True])
        observations = {
            "drone_0": obs,
            "drone_1": obs,
            "drone_2": obs,
        }
        targets_state = [{"hp": 50.0}, {"hp": 10.0}, {"hp": 30.0}]  # Target 1 is best
        
        actions = policy.select_actions(observations, num_targets=3, targets_state=targets_state)
        
        # All agents should select target 1 (action 2)
        assert all(a == 2 for a in actions.values())
