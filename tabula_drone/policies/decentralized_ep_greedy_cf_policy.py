"""
Decentralized Collaborative Filtering Policy for ZK-MRTA Environment.

Implements a true ZK-MRTA compliant single-agent policy where each agent
maintains its own private latent vectors and learns independently from
collaborative observations.

This is the decentralized version of EpGreedyCFPolicy - each agent instance
owns its own state with no shared matrices between agents.
"""

from typing import Dict, Optional, Any

import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length."""
    norm = np.linalg.norm(v)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v / norm


class DecentralizedEpGreedyCFPolicy:
    """
    Decentralized Collaborative Filtering policy using SGD matrix factorization.
    
    Each agent maintains its own private latent vectors:
    - agent_lv: This agent's latent vector (1D)
    - target_lv: This agent's estimates of all targets (2D matrix)
    
    Learns from collaborative observations (all agents' actions/rewards)
    but uses only its own private vectors for predictions and updates.
    
    ZK-MRTA Compliant: No shared state between agent instances.
    
    Designed for use with DroneEngageZKMRTA in collaborative observation mode.
    One instance per agent in the swarm.
    """
    
    def __init__(
        self,
        num_targets: int,
        agent_idx: int,
        num_agents: int,
        latent_dim: int = 2,
        learning_rate: float = 0.01,
        epsilon: float = 0.3,
        epsilon_decay: float = 0.99,
        epsilon_min: float = 0.05,
        seed: Optional[int] = None,
    ):
        """
        Initialize decentralized CF policy for a single agent.
        
        Args:
            num_targets: Number of targets in the environment
            agent_idx: This agent's index (0-based) - used to identify self in observations
            num_agents: Total number of agents (needed to parse observations)
            latent_dim: Dimension of latent vectors (default 2)
            learning_rate: SGD learning rate (default 0.01)
            epsilon: Initial exploration rate for ε-greedy (default 0.3)
            epsilon_decay: Decay factor for epsilon per step (default 0.99)
            epsilon_min: Minimum epsilon value (default 0.05)
            seed: Random seed for reproducibility
        """
        self.num_targets = num_targets
        self.agent_idx = agent_idx
        self.num_agents = num_agents
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.epsilon_initial = epsilon
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.rng = np.random.RandomState(seed)
        
        # Initialize private latent vectors
        # agent_lv: This agent's latent vector (1D)
        self.agent_lv = self._init_latent_vector()
        # target_lv: This agent's private estimates of all targets (2D)
        self.target_lv = self._init_latent_vectors(num_targets)
        # other_agents_lv: This agent's estimates of other agents' latent vectors
        # Needed to learn from others' observed rewards
        self.other_agents_lv = self._init_latent_vectors(num_agents)
    
    def _init_latent_vector(self) -> np.ndarray:
        """Initialize a single normalized random latent vector."""
        vector = self.rng.uniform(-1, 1, self.latent_dim)
        return normalize(vector).astype(np.float32)
    
    def _init_latent_vectors(self, count: int) -> np.ndarray:
        """Initialize normalized random latent vectors."""
        vectors = self.rng.uniform(-1, 1, (count, self.latent_dim))
        for i in range(count):
            vectors[i] = normalize(vectors[i])
        return vectors.astype(np.float32)
    
    def reset(self) -> None:
        """Reset all state (full reset for new experiment)."""
        self.agent_lv = self._init_latent_vector()
        self.target_lv = self._init_latent_vectors(self.num_targets)
        self.other_agents_lv = self._init_latent_vectors(self.num_agents)
        self.epsilon = self.epsilon_initial
    
    def soft_reset(self) -> None:
        """Reset for new episode, preserving learned knowledge."""
        pass
    
    def predict_reward(self, target_idx: int) -> float:
        """
        Predict reward for this agent attacking a target.
        
        Uses this agent's private latent vectors.
        
        Args:
            target_idx: Target index (0-based)
        
        Returns:
            Predicted reward in [0, 1] range
        """
        dot = np.dot(self.agent_lv, self.target_lv[target_idx])
        return (1 + dot) / 2
    
    def _predict_reward_for_other(self, other_agent_idx: int, target_idx: int) -> float:
        """
        Predict reward for another agent attacking a target.
        
        Uses this agent's estimates of the other agent's latent vector.
        
        Args:
            other_agent_idx: Other agent's index (0-based)
            target_idx: Target index (0-based)
        
        Returns:
            Predicted reward in [0, 1] range
        """
        dot = np.dot(self.other_agents_lv[other_agent_idx], self.target_lv[target_idx])
        return (1 + dot) / 2
    
    def update(self, target_idx: int, observed_reward: float) -> None:
        """
        Update this agent's latent vectors based on its own observed reward.
        
        Args:
            target_idx: Target index that was selected
            observed_reward: Observed reward value
        """
        if target_idx < 0 or observed_reward < 0:
            return
        
        predicted = self.predict_reward(target_idx)
        error = observed_reward - predicted
        
        agent_vec = self.agent_lv.copy()
        target_vec = self.target_lv[target_idx].copy()
        
        self.agent_lv += self.learning_rate * error * target_vec
        self.target_lv[target_idx] += self.learning_rate * error * agent_vec
        
        self.agent_lv = normalize(self.agent_lv)
        self.target_lv[target_idx] = normalize(self.target_lv[target_idx])
    
    def _update_from_other(
        self, other_agent_idx: int, target_idx: int, observed_reward: float
    ) -> None:
        """
        Update this agent's estimates based on another agent's observed reward.
        
        Updates this agent's estimate of the other agent's latent vector
        and this agent's estimate of the target's latent vector.
        
        Args:
            other_agent_idx: Other agent's index (0-based)
            target_idx: Target index that was selected
            observed_reward: Observed reward value
        """
        if target_idx < 0 or observed_reward < 0:
            return
        
        predicted = self._predict_reward_for_other(other_agent_idx, target_idx)
        error = observed_reward - predicted
        
        other_agent_vec = self.other_agents_lv[other_agent_idx].copy()
        target_vec = self.target_lv[target_idx].copy()
        
        self.other_agents_lv[other_agent_idx] += self.learning_rate * error * target_vec
        self.target_lv[target_idx] += self.learning_rate * error * other_agent_vec
        
        self.other_agents_lv[other_agent_idx] = normalize(self.other_agents_lv[other_agent_idx])
        self.target_lv[target_idx] = normalize(self.target_lv[target_idx])
    
    def update_from_observation(self, observation: Dict[str, Any]) -> None:
        """
        Update latent vectors from collaborative mode observation.
        
        Learns from all agents' observed rewards:
        - For this agent's own action: updates agent_lv and target_lv
        - For other agents' actions: updates other_agents_lv and target_lv
        
        Args:
            observation: Dict observation from collaborative mode containing
                        'selected_targets' and 'observed_rewards' arrays
        """
        selected_targets = observation['selected_targets']
        observed_rewards = observation['observed_rewards']
        
        for other_agent_idx in range(self.num_agents):
            target_action = selected_targets[other_agent_idx]
            reward = observed_rewards[other_agent_idx]
            
            if target_action > 0:
                target_idx = target_action - 1
                
                if other_agent_idx == self.agent_idx:
                    self.update(target_idx, reward)
                else:
                    self._update_from_other(other_agent_idx, target_idx, reward)
    
    def select_action(
        self,
        observation: Dict[str, Any],
        allow_noop: bool = False,
    ) -> int:
        """
        Select action using ε-greedy over predicted rewards.
        
        Args:
            observation: Dict observation from collaborative mode
            allow_noop: If True, include NoOp (0) as valid action
        
        Returns:
            Action: 0 for NoOp, 1-N for fire at target
        """
        targets_obs = observation['targets']
        num_targets = len(targets_obs) // 3
        
        active_targets = []
        for t_idx in range(num_targets):
            is_active = targets_obs[t_idx * 3 + 2] > 0.5
            if is_active:
                active_targets.append(t_idx)
        
        if not active_targets:
            return 0
        
        valid_actions = [0] if allow_noop else []
        valid_actions.extend([t + 1 for t in active_targets])
        
        if self.rng.random() < self.epsilon:
            action = self.rng.choice(valid_actions)
        else:
            best_action = valid_actions[0]
            best_reward = -np.inf
            
            for action in valid_actions:
                if action == 0:
                    continue
                target_idx = action - 1
                predicted = self.predict_reward(target_idx)
                if predicted > best_reward:
                    best_reward = predicted
                    best_action = action
            
            action = best_action
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return int(action)
