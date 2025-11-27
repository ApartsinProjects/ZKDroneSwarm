"""
Random Policy for ZK-MRTA Environment.

Implements uniform random selection over active targets, compliant with
Zero-Knowledge Multi-Robot Task Allocation constraints.
"""

from typing import Dict, Optional

import numpy as np


class RandomPolicy:
    """
    Random policy baseline for ZK-MRTA environment.
    
    Selects uniformly at random from:
    - All active targets (1 to num_targets)
    - NoOp (0) - optional, controlled by allow_noop parameter
    
    ZK-Compliant:
    - Does not use target HP, class types, or damage values
    - Only uses binary active/inactive status from observations
    - No memory or learning across steps
    - Treats all active targets identically
    """
    
    def __init__(self, seed: Optional[int] = None, allow_noop: bool = True):
        """
        Initialize random policy with optional seed.
        
        Args:
            seed: Random seed for reproducibility
            allow_noop: If True, action 0 (NoOp) is included in valid actions.
                       If False, agents must always fire at an active target.
        """
        self.rng = np.random.RandomState(seed)
        self.allow_noop = allow_noop
    
    def select_action(
        self,
        observation: np.ndarray,
        num_targets: int,
    ) -> int:
        """
        Select a random action for a single agent.
        
        Parses observation to identify active targets, then selects
        uniformly from valid actions. When allow_noop=True, includes
        [NoOp (0)] + [active target indices (1-N)]. When allow_noop=False,
        only includes [active target indices (1-N)].
        
        Args:
            observation: ZK observation array with shape (3 * num_targets,)
                        Format: [target_0_x, target_0_y, target_0_active, ...]
            num_targets: Total number of targets in environment
        
        Returns:
            action: Integer in [0, num_targets] if allow_noop=True,
                   or [1, num_targets] if allow_noop=False
                   0 = NoOp (only when allow_noop=True)
                   1 to num_targets = Fire at target (1-indexed)
        """
        # Parse observation to extract active status
        # Observation structure: [x, y, active] * num_targets
        # Active status is every 3rd element starting at index 2
        active_states = []
        for target_idx in range(num_targets):
            obs_idx = target_idx * 3 + 2  # Index of active field
            is_active = observation[obs_idx] > 0.5  # Binary: 1.0 or 0.0
            active_states.append(is_active)
        
        # Build list of valid actions: conditionally include NoOp
        valid_actions = [0] if self.allow_noop else []
        for target_idx in range(num_targets):
            if active_states[target_idx]:
                action = target_idx + 1  # Convert to 1-indexed action
                valid_actions.append(action)
        
        # Uniform random selection
        action = self.rng.choice(valid_actions)
        
        return int(action)
    
    def select_actions(
        self,
        observations: Dict[str, np.ndarray],
        num_targets: int,
    ) -> Dict[str, int]:
        """
        Select random actions for all agents.
        
        Each agent independently samples from their valid actions
        (NoOp + active targets). Actions are independent - no coordination.
        
        Args:
            observations: Dict of {agent_id: observation_array}
            num_targets: Total number of targets in environment
        
        Returns:
            actions: Dict of {agent_id: action}
        """
        actions = {}
        
        for agent_id, observation in observations.items():
            action = self.select_action(observation, num_targets)
            actions[agent_id] = action
        
        return actions
