"""
Collaborative Filtering Policy for ZK-MRTA Environment.

Implements SGD-based matrix factorization with ε-greedy exploration
to learn agent-target compatibility from observed rewards.
"""

from typing import Dict, Optional, Any

import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length."""
    norm = np.linalg.norm(v)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v / norm


class EpGreedyCFPolicy:
    """
    Collaborative Filtering policy using SGD matrix factorization.
    
    Learns latent vectors for agents and targets from observed rewards.
    Uses ε-greedy exploration for action selection.
    
    Designed for use with DroneEngageZKMRTA in collaborative observation mode.
    """
    
    def __init__(
        self,
        num_agents: int,
        num_targets: int,
        latent_dim: int = 2,
        learning_rate: float = 0.1,
        epsilon: float = 0.3,
        epsilon_decay: float = 0.99,
        epsilon_min: float = 0.05,
        seed: Optional[int] = None,
    ):
        """
        Initialize CF policy.
        
        Args:
            num_agents: Number of agents in the environment
            num_targets: Number of targets in the environment
            latent_dim: Dimension of latent vectors (default 2)
            learning_rate: SGD learning rate (default 0.1)
            epsilon: Initial exploration rate for ε-greedy (default 0.3)
            epsilon_decay: Decay factor for epsilon per step (default 0.99)
            epsilon_min: Minimum epsilon value (default 0.05)
            seed: Random seed for reproducibility
        """
        self.num_agents = num_agents
        self.num_targets = num_targets
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.rng = np.random.RandomState(seed)
        
        # Initialize latent vectors randomly (normalized)
        self.agent_lv = self._init_latent_vectors(num_agents)
        self.target_lv = self._init_latent_vectors(num_targets)
    
    def _init_latent_vectors(self, count: int) -> np.ndarray:
        """Initialize normalized random latent vectors."""
        vectors = self.rng.uniform(-1, 1, (count, self.latent_dim))
        # Normalize each vector
        for i in range(count):
            vectors[i] = normalize(vectors[i])
        return vectors.astype(np.float32)
    
    def reset(self) -> None:
        """Reset all state including agent latent vectors (full reset)."""
        self.agent_lv = self._init_latent_vectors(self.num_agents)
        self.target_lv = self._init_latent_vectors(self.num_targets)
        # Reset epsilon to initial value
        self.epsilon = max(self.epsilon, 0.3)
    
    def soft_reset(self) -> None:
        """Reset for new episode, preserving agent latent vectors (weapon knowledge)."""
        self.target_lv = self._init_latent_vectors(self.num_targets)
        # Reset epsilon to initial value for fresh exploration
        self.epsilon = max(self.epsilon, 0.3)
    
    def predict_reward(self, agent_idx: int, target_idx: int) -> float:
        """
        Predict reward for agent-target pair using dot product.
        
        Args:
            agent_idx: Agent index (0-based)
            target_idx: Target index (0-based)
        
        Returns:
            Predicted reward in [0, 1] range
        """
        dot = np.dot(self.agent_lv[agent_idx], self.target_lv[target_idx])
        # Scale from [-1, 1] to [0, 1]
        return (1 + dot) / 2
    
    def update(self, agent_idx: int, target_idx: int, observed_reward: float) -> None:
        """
        Update latent vectors based on observed reward using SGD.
        
        Args:
            agent_idx: Agent index that took action
            target_idx: Target index that was selected
            observed_reward: Observed reward value
        """
        if target_idx < 0:  # NoOp action
            return
        
        predicted = self.predict_reward(agent_idx, target_idx)
        error = observed_reward - predicted
        
        # SGD update
        agent_vec = self.agent_lv[agent_idx].copy()
        target_vec = self.target_lv[target_idx].copy()
        
        self.agent_lv[agent_idx] += self.learning_rate * error * target_vec
        self.target_lv[target_idx] += self.learning_rate * error * agent_vec
        
        # Re-normalize to keep vectors on unit sphere
        self.agent_lv[agent_idx] = normalize(self.agent_lv[agent_idx])
        self.target_lv[target_idx] = normalize(self.target_lv[target_idx])
    
    def update_from_observation(self, observation: Dict[str, Any], agent_id: str) -> None:
        """
        Update latent vectors from collaborative mode observation.
        
        Learns from all agents' observed rewards (full CF).
        
        Args:
            observation: Dict observation from collaborative mode
            agent_id: ID of the agent receiving this observation
        """
        selected_targets = observation['selected_targets']
        observed_rewards = observation['observed_rewards']
        
        for other_agent_idx in range(self.num_agents):
            target_action = selected_targets[other_agent_idx]
            reward = observed_rewards[other_agent_idx]
            
            if target_action > 0:  # Not NoOp
                target_idx = target_action - 1  # Convert from 1-indexed action
                self.update(other_agent_idx, target_idx, reward)
    
    def select_action(
        self,
        agent_idx: int,
        observation: Dict[str, Any],
        allow_noop: bool = False,
    ) -> int:
        """
        Select action using ε-greedy over predicted rewards.
        
        Args:
            agent_idx: Index of the agent selecting action
            observation: Dict observation from collaborative mode
            allow_noop: If True, include NoOp (0) as valid action
        
        Returns:
            Action: 0 for NoOp, 1-N for fire at target
        """
        # Parse active targets from observation
        targets_obs = observation['targets']
        num_targets = len(targets_obs) // 3
        
        active_targets = []
        for t_idx in range(num_targets):
            is_active = targets_obs[t_idx * 3 + 2] > 0.5
            if is_active:
                active_targets.append(t_idx)
        
        if not active_targets:
            return 0  # NoOp if no active targets
        
        # Build valid actions
        valid_actions = [0] if allow_noop else []
        valid_actions.extend([t + 1 for t in active_targets])  # 1-indexed
        
        # ε-greedy selection
        if self.rng.random() < self.epsilon:
            # Explore: random action
            action = self.rng.choice(valid_actions)
        else:
            # Exploit: best predicted reward
            best_action = valid_actions[0]
            best_reward = -np.inf
            
            for action in valid_actions:
                if action == 0:
                    continue  # Skip NoOp for exploitation
                target_idx = action - 1
                predicted = self.predict_reward(agent_idx, target_idx)
                if predicted > best_reward:
                    best_reward = predicted
                    best_action = action
            
            action = best_action
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return int(action)
    
    def select_actions(
        self,
        observations: Dict[str, Dict[str, Any]],
        allow_noop: bool = False,
    ) -> Dict[str, int]:
        """
        Select actions for all agents.
        
        Args:
            observations: Dict of {agent_id: observation}
            allow_noop: If True, include NoOp as valid action
        
        Returns:
            Dict of {agent_id: action}
        """
        actions = {}
        for agent_id, obs in observations.items():
            agent_idx = int(agent_id.split('_')[1])
            actions[agent_id] = self.select_action(agent_idx, obs, allow_noop)
        return actions
