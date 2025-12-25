"""
Tests for Collaborative Filtering Policy.
"""

import pytest
import numpy as np

from tabula_drone.policies.ep_greedy_cf_policy import EpGreedyCFPolicy, normalize


class TestNormalize:
    """Test normalize helper function."""

    def test_normalize_unit_vector(self):
        """Test normalizing already unit vector."""
        v = np.array([1.0, 0.0])
        result = normalize(v)
        assert np.allclose(np.linalg.norm(result), 1.0)

    def test_normalize_arbitrary_vector(self):
        """Test normalizing arbitrary vector."""
        v = np.array([3.0, 4.0])
        result = normalize(v)
        assert np.allclose(np.linalg.norm(result), 1.0)
        assert np.allclose(result, [0.6, 0.8])

    def test_normalize_zero_vector(self):
        """Test normalizing zero vector doesn't crash."""
        v = np.array([0.0, 0.0])
        result = normalize(v)
        # Should return something without NaN
        assert not np.any(np.isnan(result))


class TestEpGreedyCFPolicyInit:
    """Test CF policy initialization."""

    def test_basic_init(self):
        """Test basic initialization."""
        policy = EpGreedyCFPolicy(num_agents=2, num_targets=3)
        assert policy.num_agents == 2
        assert policy.num_targets == 3
        assert policy.latent_dim == 2
        assert policy.learning_rate == 0.1
        assert policy.epsilon == 0.3

    def test_custom_params(self):
        """Test initialization with custom parameters."""
        policy = EpGreedyCFPolicy(
            num_agents=4,
            num_targets=8,
            latent_dim=4,
            learning_rate=0.05,
            epsilon=0.5,
            seed=42,
        )
        assert policy.num_agents == 4
        assert policy.num_targets == 8
        assert policy.latent_dim == 4
        assert policy.learning_rate == 0.05
        assert policy.epsilon == 0.5

    def test_latent_vectors_shape(self):
        """Test latent vectors have correct shape."""
        policy = EpGreedyCFPolicy(num_agents=3, num_targets=5, latent_dim=4)
        assert policy.agent_lv.shape == (3, 4)
        assert policy.target_lv.shape == (5, 4)

    def test_latent_vectors_normalized(self):
        """Test latent vectors are normalized."""
        policy = EpGreedyCFPolicy(num_agents=3, num_targets=5, seed=42)
        for i in range(policy.num_agents):
            assert np.allclose(np.linalg.norm(policy.agent_lv[i]), 1.0)
        for i in range(policy.num_targets):
            assert np.allclose(np.linalg.norm(policy.target_lv[i]), 1.0)

    def test_determinism_with_seed(self):
        """Test initialization is deterministic with seed."""
        policy1 = EpGreedyCFPolicy(num_agents=2, num_targets=3, seed=42)
        policy2 = EpGreedyCFPolicy(num_agents=2, num_targets=3, seed=42)
        policy3 = EpGreedyCFPolicy(num_agents=2, num_targets=3, seed=99)
        
        assert np.allclose(policy1.agent_lv, policy2.agent_lv)
        assert np.allclose(policy1.target_lv, policy2.target_lv)
        assert not np.allclose(policy1.agent_lv, policy3.agent_lv)


class TestPredictReward:
    """Test reward prediction."""

    def test_predict_reward_range(self):
        """Test predicted reward is in [0, 1] range."""
        policy = EpGreedyCFPolicy(num_agents=2, num_targets=3, seed=42)
        for a in range(policy.num_agents):
            for t in range(policy.num_targets):
                pred = policy.predict_reward(a, t)
                assert 0 <= pred <= 1, f"Prediction {pred} out of range"

    def test_predict_reward_identical_vectors(self):
        """Test prediction when agent and target vectors are identical."""
        policy = EpGreedyCFPolicy(num_agents=1, num_targets=1, seed=42)
        # Set identical vectors
        policy.agent_lv[0] = np.array([1.0, 0.0])
        policy.target_lv[0] = np.array([1.0, 0.0])
        pred = policy.predict_reward(0, 0)
        assert np.isclose(pred, 1.0)  # dot product = 1, scaled to 1

    def test_predict_reward_opposite_vectors(self):
        """Test prediction when vectors are opposite."""
        policy = EpGreedyCFPolicy(num_agents=1, num_targets=1, seed=42)
        policy.agent_lv[0] = np.array([1.0, 0.0])
        policy.target_lv[0] = np.array([-1.0, 0.0])
        pred = policy.predict_reward(0, 0)
        assert np.isclose(pred, 0.0)  # dot product = -1, scaled to 0


class TestUpdate:
    """Test SGD update."""

    def test_update_modifies_vectors(self):
        """Test that update modifies latent vectors."""
        policy = EpGreedyCFPolicy(num_agents=1, num_targets=1, seed=42)
        agent_before = policy.agent_lv[0].copy()
        target_before = policy.target_lv[0].copy()
        
        policy.update(0, 0, 1.0)
        
        # At least one should change (unless prediction was exactly 1.0)
        changed = (
            not np.allclose(policy.agent_lv[0], agent_before) or
            not np.allclose(policy.target_lv[0], target_before)
        )
        assert changed

    def test_update_vectors_stay_normalized(self):
        """Test that vectors stay normalized after update."""
        policy = EpGreedyCFPolicy(num_agents=2, num_targets=3, seed=42)
        
        for _ in range(10):
            policy.update(0, 0, 1.0)
            policy.update(1, 2, 0.5)
        
        for i in range(policy.num_agents):
            assert np.allclose(np.linalg.norm(policy.agent_lv[i]), 1.0)
        for i in range(policy.num_targets):
            assert np.allclose(np.linalg.norm(policy.target_lv[i]), 1.0)

    def test_update_noop_does_nothing(self):
        """Test that update with NoOp target does nothing."""
        policy = EpGreedyCFPolicy(num_agents=1, num_targets=1, seed=42)
        agent_before = policy.agent_lv.copy()
        target_before = policy.target_lv.copy()
        
        policy.update(0, -1, 1.0)  # NoOp
        
        assert np.allclose(policy.agent_lv, agent_before)
        assert np.allclose(policy.target_lv, target_before)

    def test_update_improves_prediction(self):
        """Test that repeated updates improve prediction toward observed reward."""
        policy = EpGreedyCFPolicy(num_agents=1, num_targets=1, learning_rate=0.5, seed=42)
        
        target_reward = 0.9
        initial_pred = policy.predict_reward(0, 0)
        
        for _ in range(20):
            policy.update(0, 0, target_reward)
        
        final_pred = policy.predict_reward(0, 0)
        
        # Final prediction should be closer to target than initial
        assert abs(final_pred - target_reward) < abs(initial_pred - target_reward)


class TestUpdateFromObservation:
    """Test update from collaborative observation."""

    def test_update_from_observation(self):
        """Test updating from collaborative mode observation."""
        policy = EpGreedyCFPolicy(num_agents=2, num_targets=3, seed=42)
        
        observation = {
            'targets': np.array([100, 100, 1.0, 200, 200, 1.0, 300, 300, 1.0]),
            'selected_targets': np.array([1, 2]),  # drone_0 -> target 0, drone_1 -> target 1
            'observed_rewards': np.array([0.8, 0.3]),
        }
        
        agent_before = policy.agent_lv.copy()
        policy.update_from_observation(observation, 'drone_0')
        
        # Vectors should have changed
        assert not np.allclose(policy.agent_lv, agent_before)


class TestSelectAction:
    """Test action selection."""

    def test_select_action_returns_valid_action(self):
        """Test that select_action returns valid action."""
        policy = EpGreedyCFPolicy(num_agents=2, num_targets=3, seed=42)
        
        observation = {
            'targets': np.array([100, 100, 1.0, 200, 200, 1.0, 300, 300, 0.0]),  # 2 active
            'selected_targets': np.array([0, 0]),
            'observed_rewards': np.array([0.0, 0.0]),
        }
        
        action = policy.select_action(0, observation, allow_noop=False)
        assert action in [1, 2]  # Only active targets (1-indexed)

    def test_select_action_with_noop(self):
        """Test that select_action can return NoOp when allowed."""
        policy = EpGreedyCFPolicy(num_agents=1, num_targets=1, epsilon=1.0, seed=42)
        
        observation = {
            'targets': np.array([100, 100, 1.0]),
            'selected_targets': np.array([0]),
            'observed_rewards': np.array([0.0]),
        }
        
        # With high epsilon, should eventually select NoOp
        actions = [policy.select_action(0, observation, allow_noop=True) for _ in range(100)]
        assert 0 in actions  # NoOp should appear

    def test_select_action_no_active_targets(self):
        """Test select_action returns NoOp when no active targets."""
        policy = EpGreedyCFPolicy(num_agents=1, num_targets=2, seed=42)
        
        observation = {
            'targets': np.array([100, 100, 0.0, 200, 200, 0.0]),  # All inactive
            'selected_targets': np.array([0]),
            'observed_rewards': np.array([0.0]),
        }
        
        action = policy.select_action(0, observation, allow_noop=False)
        assert action == 0  # Must be NoOp

    def test_epsilon_decay(self):
        """Test that epsilon decays after action selection."""
        policy = EpGreedyCFPolicy(num_agents=1, num_targets=1, epsilon=0.5, epsilon_decay=0.9, seed=42)
        
        observation = {
            'targets': np.array([100, 100, 1.0]),
            'selected_targets': np.array([0]),
            'observed_rewards': np.array([0.0]),
        }
        
        initial_epsilon = policy.epsilon
        policy.select_action(0, observation)
        assert policy.epsilon < initial_epsilon
        assert np.isclose(policy.epsilon, initial_epsilon * 0.9)

    def test_epsilon_min_respected(self):
        """Test that epsilon doesn't go below minimum."""
        policy = EpGreedyCFPolicy(
            num_agents=1, num_targets=1,
            epsilon=0.1, epsilon_decay=0.5, epsilon_min=0.05,
            seed=42
        )
        
        observation = {
            'targets': np.array([100, 100, 1.0]),
            'selected_targets': np.array([0]),
            'observed_rewards': np.array([0.0]),
        }
        
        for _ in range(100):
            policy.select_action(0, observation)
        
        assert policy.epsilon >= policy.epsilon_min


class TestSelectActions:
    """Test batch action selection."""

    def test_select_actions_all_agents(self):
        """Test selecting actions for all agents."""
        policy = EpGreedyCFPolicy(num_agents=2, num_targets=3, seed=42)
        
        observations = {
            'drone_0': {
                'targets': np.array([100, 100, 1.0, 200, 200, 1.0, 300, 300, 1.0]),
                'selected_targets': np.array([0, 0]),
                'observed_rewards': np.array([0.0, 0.0]),
            },
            'drone_1': {
                'targets': np.array([100, 100, 1.0, 200, 200, 1.0, 300, 300, 1.0]),
                'selected_targets': np.array([0, 0]),
                'observed_rewards': np.array([0.0, 0.0]),
            },
        }
        
        actions = policy.select_actions(observations)
        
        assert 'drone_0' in actions
        assert 'drone_1' in actions
        assert actions['drone_0'] in [1, 2, 3]
        assert actions['drone_1'] in [1, 2, 3]


class TestReset:
    """Test policy reset."""

    def test_reset_reinitializes_vectors(self):
        """Test that reset reinitializes latent vectors."""
        policy = EpGreedyCFPolicy(num_agents=2, num_targets=3, seed=42)
        
        # Modify vectors
        policy.agent_lv[0] = np.array([0.5, 0.5])
        
        policy.reset()
        
        # Vectors should be different (reinitialized)
        assert not np.allclose(policy.agent_lv[0], [0.5, 0.5])


class TestIntegration:
    """Integration tests with DroneEngageZKMRTA environment."""

    def test_cf_policy_with_env_no_errors(self):
        """Test CF policy runs with environment without errors."""
        from tabula_drone.envs.drone_engage_zk_mrta_v0 import DroneEngageZKMRTA
        
        env = DroneEngageZKMRTA(
            drones_config=[
                {'position': (100, 100), 'weapon_type': 'heavy'},
                {'position': (200, 200), 'weapon_type': 'heavy'},
            ],
            targets_config=[
                {'position': (500, 500), 'class_type': 'A'},
                {'position': (600, 600), 'class_type': 'B'},
            ],
            class_attribute_mapping={
                'A': {'hp': 100},
                'B': {'hp': 150},
            },
            weapon_damage_profile_mapping={
                'heavy': {'hp': 50},
            },
            observation_mode='collaborative',
            reward_noise=0.0,
            observation_noise=0.0,
        )
        
        policy = EpGreedyCFPolicy(
            num_agents=env.num_drones,
            num_targets=env.num_targets,
            seed=42,
        )
        
        obs, _ = env.reset(seed=42)
        
        for step in range(20):
            actions = policy.select_actions(obs)
            
            # Update policy from observations
            for agent_id, agent_obs in obs.items():
                policy.update_from_observation(agent_obs, agent_id)
            
            obs, rewards, terminations, truncations, _ = env.step(actions)
            
            if terminations['drone_0'] or truncations['drone_0']:
                break
        
        # Should complete without errors
        assert True

    def test_cf_policy_produces_valid_actions(self):
        """Test CF policy produces valid actions for environment."""
        from tabula_drone.envs.drone_engage_zk_mrta_v0 import DroneEngageZKMRTA
        
        env = DroneEngageZKMRTA(
            drones_config=[
                {'position': (100, 100), 'weapon_type': 'medium'},
            ],
            targets_config=[
                {'position': (500, 500), 'class_type': 'A'},
                {'position': (600, 600), 'class_type': 'A'},
            ],
            class_attribute_mapping={'A': {'hp': 100}},
            weapon_damage_profile_mapping={'medium': {'hp': 25}},
            observation_mode='collaborative',
        )
        
        policy = EpGreedyCFPolicy(num_agents=1, num_targets=2, seed=42)
        
        obs, _ = env.reset(seed=42)
        
        for _ in range(10):
            actions = policy.select_actions(obs)
            
            # Validate action is in valid range
            for agent_id, action in actions.items():
                assert 0 <= action <= env.num_targets, f"Invalid action {action}"
            
            obs, _, term, trunc, _ = env.step(actions)
            if term['drone_0'] or trunc['drone_0']:
                break

    def test_cf_policy_learns_over_episode(self):
        """Test that CF policy's predictions change as it learns."""
        from tabula_drone.envs.drone_engage_zk_mrta_v0 import DroneEngageZKMRTA
        
        env = DroneEngageZKMRTA(
            drones_config=[
                {'position': (100, 100), 'weapon_type': 'heavy'},
            ],
            targets_config=[
                {'position': (500, 500), 'class_type': 'A'},
            ],
            class_attribute_mapping={'A': {'hp': 100}},
            weapon_damage_profile_mapping={'heavy': {'hp': 50}},
            observation_mode='collaborative',
            reward_noise=0.0,
            observation_noise=0.0,
        )
        
        policy = EpGreedyCFPolicy(num_agents=1, num_targets=1, learning_rate=0.5, seed=42)
        
        initial_prediction = policy.predict_reward(0, 0)
        
        obs, _ = env.reset(seed=42)
        
        for _ in range(10):
            actions = policy.select_actions(obs)
            obs, rewards, term, trunc, _ = env.step(actions)
            
            # Update from observation
            for agent_id, agent_obs in obs.items():
                policy.update_from_observation(agent_obs, agent_id)
            
            if term['drone_0'] or trunc['drone_0']:
                break
        
        final_prediction = policy.predict_reward(0, 0)
        
        # Prediction should have changed due to learning
        assert initial_prediction != final_prediction
