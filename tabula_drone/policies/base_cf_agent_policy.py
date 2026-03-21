"""
Base Collaborative Filtering Agent Policy for ZK-MRTA Environment.

Abstract base class containing shared CF logic for SGD-based matrix factorization
policies. Subclasses implement their own action selection strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any

import numpy as np

from .base import IPolicy


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length."""
    norm = np.linalg.norm(v)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v / norm


"""
Dual-Track Mechanism
"""
class BaseCFAgentPolicy(ABC, IPolicy):
    """
    Abstract base class for Collaborative Filtering agent policies using SGD matrix factorization.
    
    Each agent maintains its own private latent vectors:
    - agent_lv: This agent's latent vector (1D)
    - target_lv: This agent's estimates of all targets (2D matrix)
    - other_agents_lv: This agent's estimates of other agents' latent vectors
    
    Learns from collaborative observations (all agents' actions/rewards)
    but uses only its own private vectors for predictions and updates.
    
    ZK-MRTA Compliant: No shared state between agent instances.
    
    Designed for use with DroneEngageZKMRTA in collaborative observation mode.
    One instance per agent in the swarm.
    
    NOTE: This is a per-agent CF policy. To use with multi-agent environments,
    wrap instances in MultiAgentPolicy.
    
    All CF policies are non-deterministic due to learning and exploration.
    
    Subclasses must implement:
    - select_action(): Action selection strategy (e.g., ε-greedy, Hungarian)
    """
    
    is_deterministic: bool = False
    
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
        social_trust_factor: float = 0.3,
        divergence_threshold: float = 0.5,
        confidence_threshold: float = 0.8,
        social_reward_clip_min: float = -0.5,
        max_episodes: int = 100,
        seed: Optional[int] = None,
    ):
        """
        Initialize base CF policy for a single agent.
        
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
        self.social_trust_factor = social_trust_factor
        self.divergence_threshold = divergence_threshold
        self.confidence_threshold = confidence_threshold
        self.social_reward_clip_min = social_reward_clip_min
        self.max_episodes = max_episodes

        self.rng = np.random.RandomState(seed)

        # Initialize private latent vectors
        # agent_lv: This agent's latent vector (1D)
        self.agent_lv = self._init_latent_vector()
        # target_lv_private: This agent's private estimates of all targets (2D)
        self.target_lv_private = self._init_latent_vectors(num_targets)
        # target_lv_social: This agent's estimates of targets learned from others (2D)
        self.target_lv_social = self._init_latent_vectors(num_targets)
        # Backward-compatible alias (until later steps refactor update/predict logic)
        self.target_lv = self.target_lv_private
        # other_agents_lv: This agent's estimates of other agents' latent vectors
        # Needed to learn from others' observed rewards
        self.other_agents_lv = self._init_latent_vectors(num_agents)
        self.episode_count: Optional[int] = None

        self.private_mse_ema = np.ones(self.num_targets, dtype=np.float32)
        self.private_confidence = np.zeros(self.num_targets, dtype=np.float32)
        self._confidence_ema_alpha = np.float32(0.1)
    
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
        self.target_lv_private = self._init_latent_vectors(self.num_targets)
        self.target_lv_social = self._init_latent_vectors(self.num_targets)
        self.target_lv = self.target_lv_private
        self.other_agents_lv = self._init_latent_vectors(self.num_agents)
        self.epsilon = self.epsilon_initial
        self.episode_count = None
        self.private_mse_ema = np.ones(self.num_targets, dtype=np.float32)
        self.private_confidence = np.zeros(self.num_targets, dtype=np.float32)
    
    def soft_reset(self, episode_count: Optional[int] = None) -> None:
        """Reset for new episode, preserving learned knowledge."""
        self.episode_count = episode_count
        self.private_mse_ema.fill(np.float32(1.0))
        self.private_confidence.fill(np.float32(0.0))

        if episode_count is None:
            self.current_beta = float(self.social_trust_factor)
            return

        progress = min(max(float(episode_count) / float(self.max_episodes), 0.0), 1.0)
        min_beta = 0.1
        self.current_beta = float(self.social_trust_factor) * (1.0 - progress) + min_beta * progress

    def _update_private_confidence(self, target_idx: int, error: float) -> None:
        mse = np.float32(error * error)
        prev = self.private_mse_ema[target_idx]
        alpha = self._confidence_ema_alpha
        self.private_mse_ema[target_idx] = (np.float32(1.0) - alpha) * prev + alpha * mse
        self.private_confidence[target_idx] = np.float32(1.0) / (
            np.float32(1.0) + self.private_mse_ema[target_idx]
        )

    def _compute_effective_beta(
        self,
        target_idx: int,
        private_pred: float,
        social_pred: float,
    ) -> float:
        base_beta = float(getattr(self, "current_beta", self.social_trust_factor))
        divergence = abs(float(private_pred) - float(social_pred))
        if divergence > float(self.divergence_threshold) and float(self.private_confidence[target_idx]) >= float(
            self.confidence_threshold
        ):
            return base_beta * 0.1
        return base_beta
    
    def predict_reward(self, target_idx: int) -> float:
        """
        Predict reward for this agent attacking a target.
        
        Uses this agent's private latent vectors.
        
        Args:
            target_idx: Target index (0-based)
        
        Returns:
            Predicted reward in [0, 1] range
        """
        private_dot = np.dot(self.agent_lv, self.target_lv_private[target_idx])
        private_pred = (1 + private_dot) / 2

        social_dot = np.dot(self.agent_lv, self.target_lv_social[target_idx])
        social_pred = (1 + social_dot) / 2

        beta = self._compute_effective_beta(target_idx, float(private_pred), float(social_pred))
        return (1 - beta) * private_pred + beta * social_pred
    
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
        dot = np.dot(self.other_agents_lv[other_agent_idx], self.target_lv_social[target_idx])
        return (1 + dot) / 2
    
    def _update_from_own_reward(self, target_idx: int, observed_reward: float) -> None:
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

        self._update_private_confidence(target_idx, float(error))
        
        agent_vec = self.agent_lv.copy()
        target_vec = self.target_lv_private[target_idx].copy()
        
        self.agent_lv += self.learning_rate * error * target_vec
        self.target_lv_private[target_idx] += self.learning_rate * error * agent_vec
        
        self.agent_lv = normalize(self.agent_lv)
        self.target_lv_private[target_idx] = normalize(self.target_lv_private[target_idx])

    def update(self, obs: Dict[str, Any]) -> None:
        """
        Update policy state from observations.
        
        Args:
            obs: Observations dict keyed by agent_id
        """
        agent_obs = obs.get(f"drone_{self.agent_idx}")
        if agent_obs is None:
            return
        self.update_from_observation(agent_obs)
    
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
        if target_idx < 0:
            return

        observed_reward = max(observed_reward, self.social_reward_clip_min)
        
        predicted = self._predict_reward_for_other(other_agent_idx, target_idx)
        error = observed_reward - predicted
        
        other_agent_vec = self.other_agents_lv[other_agent_idx].copy()
        target_vec = self.target_lv_social[target_idx].copy()
        
        self.other_agents_lv[other_agent_idx] += self.learning_rate * error * target_vec
        self.target_lv_social[target_idx] += self.learning_rate * error * other_agent_vec
        
        self.other_agents_lv[other_agent_idx] = normalize(self.other_agents_lv[other_agent_idx])
        self.target_lv_social[target_idx] = normalize(self.target_lv_social[target_idx])
    
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
        
        for agent_idx in range(self.num_agents):
            target_action = selected_targets[agent_idx]
            reward = observed_rewards[agent_idx]
            
            if target_action > 0:
                target_idx = target_action - 1
                
                if agent_idx == self.agent_idx:
                    self._update_from_own_reward(target_idx, reward)
                else:
                    self._update_from_other(agent_idx, target_idx, reward)
    
    def get_learning_state(self) -> Optional[Dict[str, Any]]:
        """
        Return learning state for logging/visualization.
        
        Returns:
            Dict with this agent's latent vectors
        """
        return {
            "agent_idx": self.agent_idx,
            "agent_lv": self.agent_lv.tolist(),
            "target_lv": self.target_lv.tolist(),
            "target_lv_private": self.target_lv_private.tolist(),
            "target_lv_social": self.target_lv_social.tolist(),
            "other_agents_lv": self.other_agents_lv.tolist(),
            "private_confidence": self.private_confidence.tolist(),
            "current_beta": float(getattr(self, "current_beta", self.social_trust_factor)),
            "episode_count": self.episode_count,
            "epsilon": self.epsilon,
        }

    def select_actions(
        self, obs: Dict[str, Any], infos: Dict[str, Dict[str, Any]]
    ) -> Dict[str, int]:
        """
        IPolicy protocol interface - NOT directly usable for per-agent CF policies.
        
        BaseCFAgentPolicy is designed for single-agent use. To use with multi-agent
        environments, wrap instances in MultiAgentPolicy.
        
        Args:
            obs: Observations dict keyed by agent_id
            infos: Environment infos dict keyed by agent_id
        
        Raises:
            NotImplementedError: Always raised - use MultiAgentPolicy instead
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} is a per-agent CF policy and does not implement "
            "select_actions() directly. Use MultiAgentPolicy to aggregate "
            "per-agent policies into a multi-agent IPolicy interface."
        )

    @abstractmethod
    def select_action(
        self,
        observation: Dict[str, Any],
        allow_noop: bool = False,
    ) -> int:
        """
        Select action based on observation.
        
        Subclasses implement their own action selection strategy.
        
        Args:
            observation: Dict observation from collaborative mode
            allow_noop: If True, include NoOp (0) as valid action
        
        Returns:
            Action: 0 for NoOp, 1-N for fire at target
        """
        pass
