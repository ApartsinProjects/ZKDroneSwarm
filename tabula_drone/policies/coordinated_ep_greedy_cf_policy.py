"""
Coordinated Collaborative Filtering Policy for ZK-MRTA Environment.

Implements the "Decentralized Internal Oracle" concept: each agent uses its
learned belief matrix to compute a globally optimal assignment via the
Hungarian algorithm, then executes its assigned target.

This is a ZK-compliant policy where each agent maintains private latent vectors
and uses implicit coordination through shared beliefs about agent-target
compatibility.
"""

from typing import Dict, Optional, Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from .base_cf_policy import BaseCFPolicy


class CoordinatedEpGreedyCFPolicy(BaseCFPolicy):
    """
    Coordinated Collaborative Filtering policy using SGD matrix factorization
    with Hungarian algorithm-based action selection.
    
    Each agent maintains its own private latent vectors:
    - agent_lv: This agent's latent vector (1D)
    - target_lv: This agent's estimates of all targets (2D matrix)
    - other_agents_lv: This agent's estimates of other agents' latent vectors
    
    Action selection uses the Hungarian algorithm on the belief matrix to
    compute a globally optimal assignment, enabling implicit coordination
    without explicit communication.
    
    ZK-MRTA Compliant: No shared state between agent instances.
    
    Designed for use with DroneEngageZKMRTA in collaborative observation mode.
    One instance per agent in the swarm.
    
    Inherits from BaseCFPolicy. Implements Hungarian algorithm action selection.
    """
    
    is_deterministic: bool = False
    is_cf: bool = True
    is_ep_greedy_cf: bool = True
    
    def _select_action_greedy(
        self,
        observation: Dict[str, Any],
        allow_noop: bool = False,
    ) -> int:
        """
        Select action using selfish greedy over predicted rewards.
        
        Used as fallback when coordinated assignment doesn't assign this agent.
        
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
        
        # Exploit: best predicted reward
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
        
        return int(best_action)
    
    def select_action(
        self,
        observation: Dict[str, Any],
        allow_noop: bool = False,
    ) -> int:
        """
        Select action using coordinated assignment with ε-greedy exploration.
        
        Uses the Hungarian algorithm on the belief matrix to compute a globally
        optimal assignment, then returns this agent's assigned target.
        
        Args:
            observation: Dict observation from collaborative mode
            allow_noop: If True, include NoOp (0) as valid action in fallback
        
        Returns:
            Action: 0 for NoOp, 1-N for fire at target
        """
        # Parse active targets
        targets_obs = observation['targets']
        total_targets = len(targets_obs) // 3
        active_indices = [
            i for i in range(total_targets)
            if targets_obs[i * 3 + 2] > 0.5
        ]
        
        if not active_indices:
            return 0
        
        # ε-greedy: Explore
        if self.rng.random() < self.epsilon:
            valid_actions = [t + 1 for t in active_indices]
            action = self.rng.choice(valid_actions)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            return int(action)
        
        # ε-greedy: Exploit (Coordinated Assignment)
        # Build sliced belief matrix (only active targets)
        # Shape: (num_agents, num_active_targets)
        # Inject accurate self.agent_lv into the matrix (other_agents_lv[self.agent_idx] is stale)
        full_agent_matrix = self.other_agents_lv.copy()
        full_agent_matrix[self.agent_idx] = self.agent_lv
        relevant_target_lvs = self.target_lv[active_indices]
        belief_matrix = full_agent_matrix @ relevant_target_lvs.T
        
        # Solve assignment (maximize → minimize negative)
        row_ind, col_ind = linear_sum_assignment(-belief_matrix)
        
        # Find my assignment
        my_target_global_idx = -1
        if self.agent_idx in row_ind:
            assignment_pos = list(row_ind).index(self.agent_idx)
            col_idx = col_ind[assignment_pos]
            my_target_global_idx = active_indices[col_idx]
        
        # Fallback to selfish greedy if unassigned
        if my_target_global_idx == -1:
            action = self._select_action_greedy(observation, allow_noop)
        else:
            action = my_target_global_idx + 1
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return int(action)
