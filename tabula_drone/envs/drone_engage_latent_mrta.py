"""
DroneEngageLatentMRTA: latent Gaussian-mixture benchmark environment.

This environment preserves the shared PettingZoo-facing interaction contract
used by the current collaborative policies, while replacing the custom
class/weapon semantics with hidden latent vectors and MF dot-product reward.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from .diagnostics import EnvDiagnosticsSnapshot
from ..core.states import WorldState


@dataclass
class LatentDroneState:
    """Internal drone state for the latent benchmark."""
    id: str
    position: Tuple[float, float]
    mode_id: int
    latent_vector: Tuple[float, ...]
    ammo_used: int = 0


@dataclass
class LatentTargetState:
    """Internal target state for the latent benchmark."""
    id: str
    position: Tuple[float, float]
    mode_id: int
    latent_vector: Tuple[float, ...]
    hp: float = 1.0
    hp_initial: float = 1.0
    is_active: bool = True


class DroneEngageLatentMRTA(ParallelEnv):
    """
    PettingZoo ParallelEnv for the latent Gaussian-mixture MRTA benchmark.

    Public contract:
    - observations expose target positions, active state, last actions, last rewards
    - actions are 0=NoOp, 1..N=select target
    - reward is latent dot product for the first successful engagement on a target
    """

    metadata = {"name": "drone_engage_latent_mrta"}

    def __init__(
        self,
        world_size: Tuple[float, float] = (1000.0, 1000.0),
        max_steps: int = 100,
        drones_config: List[Dict[str, Any]] = None,
        targets_config: List[Dict[str, Any]] = None,
        scenario_id: str = "latent_mrta_benchmark",
        reward_noise: float = 0.0,
        observation_noise: float = 0.0,
        builder: Optional[Any] = None,
        latent_world: Optional[Dict[str, Any]] = None,
        target_hp: float = 1.0,
    ):
        super().__init__()

        if not drones_config:
            raise ValueError("drones_config must contain at least one drone")
        if not targets_config:
            raise ValueError("targets_config must contain at least one target")

        self.world_model = "latent"
        self.world_size = world_size
        self.max_steps = max_steps
        self.drones_config = drones_config
        self.targets_config = targets_config
        self.scenario_id = scenario_id
        self.reward_noise = reward_noise
        self.observation_noise = observation_noise
        self.builder = builder
        # Convert latent_world to dict if it's a dataclass, otherwise use as-is
        if latent_world is not None:
            if isinstance(latent_world, dict):
                self.latent_world = dict(latent_world)
            else:
                # Assume it's a dataclass instance
                self.latent_world = asdict(latent_world)
            # Ensure target_hp is included in the config for frontend display
            self.latent_world["target_hp"] = float(target_hp)
        else:
            self.latent_world = None
        self.target_hp = float(target_hp)
        self.num_modes = self._infer_num_modes()
        self.class_attribute_mapping = {
            f"mode_{mode_id}": {"latent_reward": self.target_hp}
            for mode_id in range(self.num_modes)
        }
        # Build per-mode weapon profiles using ||drone|| × mean(||target||) as a
        # Cauchy-Schwarz upper bound on achievable dot-product per shot.
        avg_target_norm = float(np.mean([
            np.linalg.norm(tcfg["latent_vector"]) for tcfg in self.targets_config
        ])) if self.targets_config else 1.0
        mode_norms: Dict[int, List[float]] = {}
        for dcfg in self.drones_config:
            mid = int(dcfg["mode_id"])
            norm = float(np.linalg.norm(dcfg["latent_vector"]))
            mode_norms.setdefault(mid, []).append(norm)
        self.weapon_damage_profile_mapping = {
            f"mode_{mid}": {"latent_reward": float(np.mean(norms)) * avg_target_norm}
            for mid, norms in mode_norms.items()
        }

        self.possible_agents = [f"drone_{i}" for i in range(len(self.drones_config))]
        self._agents = self.possible_agents[:]
        self.num_drones = len(self.drones_config)
        self.num_targets = len(self.targets_config)

        self.action_spaces = {
            agent_id: spaces.Discrete(self.num_targets + 1)
            for agent_id in self.possible_agents
        }

        obs_dim = 3 * self.num_targets
        self.observation_spaces = {
            agent_id: spaces.Dict(
                {
                    "targets": spaces.Box(
                        low=0.0,
                        high=np.inf,
                        shape=(obs_dim,),
                        dtype=np.float32,
                    ),
                    "selected_targets": spaces.Box(
                        low=0,
                        high=self.num_targets,
                        shape=(self.num_drones,),
                        dtype=np.int32,
                    ),
                    "observed_rewards": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(self.num_drones,),
                        dtype=np.float32,
                    ),
                    "target_was_active_at_engagement": spaces.Box(
                        low=0,
                        high=1,
                        shape=(self.num_drones,),
                        dtype=np.int8,
                    ),
                }
            )
            for agent_id in self.possible_agents
        }

        self.drones: Optional[List[LatentDroneState]] = None
        self.targets: Optional[List[LatentTargetState]] = None
        self.world: Optional[WorldState] = None
        self.rng: Optional[np.random.RandomState] = None
        self.last_actions: Dict[str, int] = {}
        self.last_rewards: Dict[str, float] = {}
        self.last_target_was_active_at_engagement: Dict[str, int] = {}
        self._latest_diagnostics: Optional[EnvDiagnosticsSnapshot] = None
        self.cumulative_neutralizations = 0

    def _infer_num_modes(self) -> int:
        """Infer the total number of latent modes for compatibility metadata."""
        if self.latent_world is not None and "num_modes" in self.latent_world:
            return int(self.latent_world["num_modes"])
        max_mode_id = max(
            [int(cfg["mode_id"]) for cfg in self.drones_config]
            + [int(cfg["mode_id"]) for cfg in self.targets_config]
        )
        return max_mode_id + 1

    @property
    def agents(self) -> List[str]:
        return self._agents

    @property
    def num_agents(self) -> int:
        return len(self.possible_agents)

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    @property
    def diagnostics(self) -> Optional[EnvDiagnosticsSnapshot]:
        return self._latest_diagnostics

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        self.rng = np.random.RandomState(seed)
        self.world = WorldState(
            world_size=self.world_size,
            time_step=0,
            max_steps=self.max_steps,
            scenario_id=self.scenario_id,
            seed=seed,
        )

        self.drones = []
        for idx, drone_cfg in enumerate(self.drones_config):
            self.drones.append(
                LatentDroneState(
                    id=f"drone_{idx}",
                    position=tuple(drone_cfg["position"]),
                    mode_id=int(drone_cfg["mode_id"]),
                    latent_vector=tuple(float(v) for v in drone_cfg["latent_vector"]),
                    ammo_used=0,
                )
            )

        self.targets = []
        for idx, target_cfg in enumerate(self.targets_config):
            self.targets.append(
                LatentTargetState(
                    id=f"target_{idx}",
                    position=tuple(target_cfg["position"]),
                    mode_id=int(target_cfg["mode_id"]),
                    latent_vector=tuple(float(v) for v in target_cfg["latent_vector"]),
                    hp=self.target_hp,
                    hp_initial=self.target_hp,
                    is_active=True,
                )
            )

        self._agents = self.possible_agents[:]
        self.last_actions = {agent_id: 0 for agent_id in self.possible_agents}
        self.last_rewards = {agent_id: 0.0 for agent_id in self.possible_agents}
        self.last_target_was_active_at_engagement = {
            agent_id: 0 for agent_id in self.possible_agents
        }
        self.cumulative_neutralizations = 0

        observations = self._compute_observations()
        shared_info = self._build_info_dict(actions={})
        infos = self._wrap_shared_info(shared_info)
        return observations, infos

    def _compute_observations(self) -> Dict[str, Any]:
        target_obs: List[float] = []
        for target in self.targets or []:
            x, y = target.position
            target_obs.extend([x, y, 1.0 if target.is_active else 0.0])
        target_array = np.array(target_obs, dtype=np.float32)

        # Build selected_targets array once (shared across all agents)
        selected_targets = np.array(
            [self.last_actions.get(aid, 0) for aid in self.possible_agents],
            dtype=np.int32,
        )

        # Apply observation noise (corrupt other agents' observed actions)
        if self.observation_noise > 0:
            for i in range(len(selected_targets)):
                # Only corrupt non-noop actions (preserve 0 = noop)
                if selected_targets[i] > 0 and self.rng.random() < self.observation_noise:
                    # Replace with random valid target ID [1, num_targets]
                    selected_targets[i] = self.rng.randint(1, self.num_targets + 1)

        observations = {}
        for agent_id in self.agents:
            observed_rewards = np.array(
                [self._compute_observed_reward(aid) for aid in self.possible_agents],
                dtype=np.float32,
            )
            target_was_active_at_engagement = np.array(
                [
                    self.last_target_was_active_at_engagement.get(aid, 0)
                    for aid in self.possible_agents
                ],
                dtype=np.int8,
            )
            observations[agent_id] = {
                "targets": target_array.copy(),
                "selected_targets": selected_targets,
                "observed_rewards": observed_rewards,
                "target_was_active_at_engagement": target_was_active_at_engagement,
            }
        return observations

    def _compute_observed_reward(self, source_agent_id: str) -> float:
        base_reward = self.last_rewards.get(source_agent_id, 0.0)
        noise = self.rng.normal(0, self.reward_noise) if self.reward_noise > 0 else 0.0
        return float(base_reward + noise)

    def _validate_actions(self, actions: Dict[str, int]) -> None:
        missing_agents = set(self.agents) - set(actions.keys())
        if missing_agents:
            raise ValueError(f"Missing actions for agents: {sorted(missing_agents)}")
        for agent_id, action in actions.items():
            if not self.action_spaces[agent_id].contains(action):
                raise ValueError(f"Invalid action {action} for {agent_id}")

    def _dot_product_reward(self, drone: LatentDroneState, target: LatentTargetState) -> float:
        drone_vec = np.array(drone.latent_vector, dtype=np.float64)
        target_vec = np.array(target.latent_vector, dtype=np.float64)
        return float(np.dot(drone_vec, target_vec))

    def _build_info_dict(
        self,
        actions: Dict[str, int],
        processing_order: Optional[List[str]] = None,
        net_damage: Optional[float] = None,
        neutralizations_this_step: Optional[int] = None,
        cumulative_neutralizations: Optional[int] = None,
        collisions: Optional[int] = None,
        target_selections: Optional[Dict[int, List[str]]] = None,
        overkill: Optional[Dict[int, float]] = None,
        done_reason: Optional[str] = None,
        total_gross_damage: Optional[float] = None,
    ) -> Dict[str, Any]:
        snapshot = EnvDiagnosticsSnapshot(
            step_index=self.world.time_step if self.world is not None else 0,
            scenario_id=self.scenario_id,
            actions=actions.copy(),
            ammo_used={drone.id: drone.ammo_used for drone in self.drones or []},
            weapon_types=[f"mode_{drone.mode_id}" for drone in self.drones or []],
            target_hps=[target.hp for target in self.targets or []],
            target_classes=[f"mode_{target.mode_id}" for target in self.targets or []],
            target_active=[target.is_active for target in self.targets or []],
            processing_order=processing_order,
            net_damage=net_damage,
            neutralizations_this_step=neutralizations_this_step,
            cumulative_neutralizations=cumulative_neutralizations,
            collisions=collisions,
            target_selections=target_selections,
            overkill=overkill,
            done_reason=done_reason,
            total_gross_damage=total_gross_damage,
        )
        self._latest_diagnostics = snapshot
        return snapshot.to_dict()

    def _wrap_shared_info(self, shared_info: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        return {agent_id: {} for agent_id in self.agents}

    def state(self) -> Dict[str, Any]:
        return {
            "world": {
                "world_size": list(self.world.world_size) if self.world is not None else list(self.world_size),
                "time_step": self.world.time_step if self.world is not None else 0,
                "max_steps": self.world.max_steps if self.world is not None else self.max_steps,
                "scenario_id": self.world.scenario_id if self.world is not None else self.scenario_id,
                "seed": self.world.seed if self.world is not None else None,
            },
            "drones": [
                {
                    "id": drone.id,
                    "position": list(drone.position),
                    "ammo_used": drone.ammo_used,
                    "mode_id": drone.mode_id,
                    "latent_vector": list(drone.latent_vector),
                }
                for drone in self.drones or []
            ],
            "targets": [
                {
                    "id": target.id,
                    "position": list(target.position),
                    "mode_id": target.mode_id,
                    "latent_vector": list(target.latent_vector),
                    "hp": target.hp,
                    "hp_initial": target.hp_initial,
                    "is_active": target.is_active,
                }
                for target in self.targets or []
            ],
        }

    def step(
        self,
        actions: Dict[str, int],
    ) -> Tuple[
        Dict[str, Union[np.ndarray, Dict[str, Any]]],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        self._validate_actions(actions)

        rewards = {agent_id: 0.0 for agent_id in self.agents}
        processing_order = list(self.agents)
        self.rng.shuffle(processing_order)

        target_selections: Dict[int, List[str]] = {}
        step_net_damage = 0.0
        step_overkill: Dict[int, float] = {}
        collisions = 0
        neutralizations_this_step = 0
        step_gross_damage = 0.0

        reward_mode = "cosine" # "cosine" or "damage"
        for agent_id in processing_order:
            action = actions[agent_id]
            self.last_actions[agent_id] = action
            if action == 0:
                self.last_rewards[agent_id] = 0.0
                self.last_target_was_active_at_engagement[agent_id] = 0
                continue

            drone_idx = int(agent_id.split("_")[1])
            drone = self.drones[drone_idx]
            target_idx = action - 1
            drone.ammo_used += 1

            target_selections.setdefault(target_idx, []).append(agent_id)
            if len(target_selections[target_idx]) > 1:
                collisions += 1

            target = self.targets[target_idx]
            if not target.is_active:
                # Target already dead — wasted shot
                raw_dot_wasted = self._dot_product_reward(drone, target)
                step_gross_damage += max(0.0, raw_dot_wasted)
                rewards[agent_id] = -1.0
                self.last_rewards[agent_id] = -1.0
                self.last_target_was_active_at_engagement[agent_id] = 0
                continue

            raw_dot = self._dot_product_reward(drone, target)
            # Damage is the non-negative part of the dot product
            damage = max(0.0, raw_dot)
            step_gross_damage += damage

            hp_before = target.hp
            target.hp -= damage

            if target.hp <= 0:
                # Target neutralized — track overkill
                overkill_amount = abs(target.hp)  # hp went below 0
                target.hp = 0.0
                target.is_active = False
                neutralizations_this_step += 1
                if overkill_amount > 0:
                    step_overkill[target_idx] = step_overkill.get(target_idx, 0.0) + overkill_amount

            # Reward = cosine similarity (direction-only alignment, independent of damage)
            drone_vec = np.array(drone.latent_vector, dtype=np.float64)
            target_vec = np.array(target.latent_vector, dtype=np.float64)
            drone_norm = np.linalg.norm(drone_vec)
            target_norm = np.linalg.norm(target_vec)
            
            if drone_norm > 0 and target_norm > 0:
                cosine_similarity = raw_dot / (drone_norm * target_norm)
                reward = float(cosine_similarity)
            else:
                reward = 0.0
            
            rewards[agent_id] = reward
            self.last_rewards[agent_id] = reward
            self.last_target_was_active_at_engagement[agent_id] = 1
            
            # Track effective damage for metrics (independent of reward)
            effective_damage = min(damage, hp_before)
            step_net_damage += effective_damage

            if(reward_mode == "damage"):
                rewards[agent_id] = effective_damage
                self.last_rewards[agent_id] = effective_damage

        self.cumulative_neutralizations += neutralizations_this_step
        if self.world is not None:
            self.world.time_step += 1

        all_targets_done = all(not target.is_active for target in self.targets or [])
        max_steps_reached = self.world.time_step >= self.max_steps if self.world is not None else False
        done_reason = None
        if all_targets_done:
            done_reason = "all_targets_neutralized"
        elif max_steps_reached:
            done_reason = "max_steps_reached"

        observations = self._compute_observations()
        shared_info = self._build_info_dict(
            actions=actions,
            processing_order=processing_order,
            net_damage=step_net_damage,
            neutralizations_this_step=neutralizations_this_step,
            cumulative_neutralizations=self.cumulative_neutralizations,
            collisions=collisions,
            target_selections=target_selections,
            overkill=step_overkill,
            done_reason=done_reason,
            total_gross_damage=step_gross_damage,
        )
        infos = self._wrap_shared_info(shared_info)

        terminations = {
            agent_id: all_targets_done
            for agent_id in self.agents
        }
        truncations = {
            agent_id: max_steps_reached and not all_targets_done
            for agent_id in self.agents
        }
        return observations, rewards, terminations, truncations, infos
