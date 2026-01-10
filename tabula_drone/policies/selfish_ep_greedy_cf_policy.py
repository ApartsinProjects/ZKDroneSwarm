"""
Selfish Collaborative Filtering Policy for ZK-MRTA Environment.

Implements a true ZK-MRTA compliant single-agent policy where each agent
maintains its own private latent vectors and learns independently from
collaborative observations.

This is the selfish version of EpGreedyCFPolicy - each agent instance
owns its own state with no shared matrices between agents.
"""

from typing import Dict, Optional, Any

import numpy as np

from .base_cf_agent_policy import BaseCFAgentPolicy


class SelfishEpGreedyCFPolicy(BaseCFAgentPolicy):
    """
    Selfish Collaborative Filtering policy using SGD matrix factorization.
    
    Each agent maintains its own private latent vectors:
    - agent_lv: This agent's latent vector (1D)
    - target_lv: This agent's estimates of all targets (2D matrix)
    
    Learns from collaborative observations (all agents' actions/rewards)
    but uses only its own private vectors for predictions and updates.
    
    ZK-MRTA Compliant: No shared state between agent instances.
    
    Designed for use with DroneEngageZKMRTA in collaborative observation mode.
    One instance per agent in the swarm.
    
    Inherits from BaseCFAgentPolicy. Implements ε-greedy action selection.
    """
    
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

    def get_learning_state(self) -> Optional[Dict[str, Any]]:
        state = super().get_learning_state()
        if state is None:
            return None

        predicted_rewards = [float(self.predict_reward(t)) for t in range(self.num_targets)]
        ranked_targets = [
            int(x)
            for x in np.argsort(-np.asarray(predicted_rewards, dtype=np.float32)).tolist()
        ]
        best_target = int(ranked_targets[0]) if ranked_targets else None

        state["match"] = {
            "predicted_rewards": predicted_rewards,
            "ranked_targets": ranked_targets,
            "best_target": best_target,
        }

        return state
