"""
UCB Collaborative Filtering Policy for ZK-MRTA Environment.

Implements SGD-based matrix factorization with UCB1 exploration
to learn agent-target compatibility from observed rewards.

UCB1 replaces ε-greedy exploration, providing uncertainty-aware
action selection that automatically prioritizes unexplored targets.
"""

from typing import Dict, Optional, Any

import numpy as np

from tabula_drone.policies.ep_greedy_cf_policy import normalize


class UCBCFPolicy:
    """
    UCB Collaborative Filtering policy using SGD matrix factorization.
    
    Learns latent vectors for agents and targets from observed rewards.
    Uses UCB1 exploration for action selection instead of ε-greedy.
    
    Key difference from CFPolicy:
    - Tracks visit counts per target (shared across all agents)
    - Computes UCB score = predicted_reward + exploration_bonus
    - Automatically prioritizes unexplored/new targets (infinite UCB score)
    
    Designed for use with DroneEngageZKMRTA in collaborative observation mode.
    """
    
    def __init__(
        self,
        num_agents: int,
        num_targets: int,
        latent_dim: int = 2,
        learning_rate: float = 0.01,
        ucb_c: float = 2.0,
        seed: Optional[int] = None,
    ):
        """
        Initialize UCB CF policy.
        
        Args:
            num_agents: Number of agents in the environment
            num_targets: Number of targets in the environment
            latent_dim: Dimension of latent vectors (default 2)
            learning_rate: SGD learning rate (default 0.1)
            ucb_c: UCB exploration coefficient (default 2.0)
            seed: Random seed for reproducibility
        """
        self.num_agents = num_agents
        self.num_targets = num_targets
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.ucb_c = ucb_c
        
        self.rng = np.random.RandomState(seed)
        
        # Initialize latent vectors randomly (normalized)
        self.agent_lv = self._init_latent_vectors(num_agents)
        self.target_lv = self._init_latent_vectors(num_targets)
        
        # UCB tracking: shared visit counts across all agents
        self.visit_counts = np.zeros(num_targets, dtype=np.int32)
        self.total_steps = 0
    
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
        # Reset UCB tracking
        self.visit_counts = np.zeros(self.num_targets, dtype=np.int32)
        self.total_steps = 0
    
    def soft_reset(self) -> None:
        """Reset for new episode, preserving agent/target latent vectors and visit counts."""
        # Preserve target_lv, visit_counts, and total_steps across episodes
        # This allows UCB to exploit learned knowledge instead of re-exploring
        pass
    
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
    
    def ucb_score(self, agent_idx: int, target_idx: int) -> float:
        """
        Compute UCB1 score for agent-target pair.
        
        UCB score = predicted_reward + c * sqrt(log(total_steps + 1) / visits)
        
        Args:
            agent_idx: Agent index (0-based)
            target_idx: Target index (0-based)
        
        Returns:
            UCB score (float('inf') for unvisited targets)
        """
        visits = self.visit_counts[target_idx]
        
        if visits == 0:
            return float('inf')  # Unexplored = highest priority
        
        predicted = self.predict_reward(agent_idx, target_idx)
        
        # UCB1 exploration bonus
        bonus = self.ucb_c * np.sqrt(np.log(self.total_steps + 1) / visits)
        
        return predicted + bonus
    
    def update(self, agent_idx: int, target_idx: int, observed_reward: float) -> None:
        """
        Update latent vectors based on observed reward using SGD.
        
        Args:
            agent_idx: Agent index that took action
            target_idx: Target index that was selected
            observed_reward: Observed reward value
        """
        if target_idx < 0 or observed_reward < 0:  # Skip NoOp and wasted shots
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
        Select action using UCB1 over predicted rewards.
        
        Args:
            agent_idx: Index of the agent selecting action
            observation: Dict observation from collaborative mode
            allow_noop: If True, include NoOp (0) as valid action (not used in UCB selection)
        
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
        
        # UCB selection: find target with highest UCB score
        best_action = 0
        best_score = -np.inf
        
        for t_idx in active_targets:
            score = self.ucb_score(agent_idx, t_idx)
            if score > best_score:
                best_score = score
                best_action = t_idx + 1  # Convert to 1-indexed action
        
        # Update visit count for selected target
        if best_action > 0:
            self.visit_counts[best_action - 1] += 1
        
        return int(best_action)
    
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
        self.total_steps += 1  # Increment once per tick, not per agent
        return actions
