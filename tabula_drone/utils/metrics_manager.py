"""
Typed metrics models and manager for ZK-MRTA performance reporting.

The manager is scoped to a single policy run and owns the mode-specific
calculation logic for per-episode and cross-episode metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Union


MetricSourceInput = "EpisodeMetrics"


@dataclass(frozen=True)
class EpisodeMetrics:
    """Fully calculated metrics for a single episode."""

    # Raw episode data
    episode: Optional[int]
    steps: int
    mode: str
    done_reason: Optional[str]
    targets_neutralized: int
    total_ammo_used: int
    total_overkill: float
    total_net_damage: float
    total_gross_damage: float
    total_collisions: int
    agent_rewards: Dict[str, float]
    weapon_damage_profile_mapping: Mapping[str, Mapping[str, float]]
    
    # Calculated fields (set in __post_init__)
    ammo_eff: float = field(init=False)
    dmg_eff: float = field(init=False)
    shots_per_target: Optional[float] = field(default=None, init=False)
    throughput: Optional[float] = field(default=None, init=False)
    coordination_score: Optional[float] = field(default=None, init=False)
    coordination_str: Optional[str] = field(default=None, init=False)
    
    def __post_init__(self):
        # Calculate efficiency metrics
        object.__setattr__(self, 'ammo_eff', 
            self.targets_neutralized / self.total_ammo_used if self.total_ammo_used > 0 else 0.0)
        object.__setattr__(self, 'dmg_eff',
            self.total_net_damage / self.total_gross_damage if self.total_gross_damage > 0 else 0.0)
        
        # Mode-specific calculations
        if self.mode == "episodic":
            object.__setattr__(self, 'shots_per_target',
                self.total_ammo_used / self.targets_neutralized if self.targets_neutralized > 0 else 0.0)
        elif self.mode == "continuous":
            object.__setattr__(self, 'throughput',
                self.targets_neutralized / self.steps * 100 if self.steps > 0 else 0.0)
            if self.total_collisions == 0:
                object.__setattr__(self, 'coordination_score', float('inf'))
                object.__setattr__(self, 'coordination_str', 'N/A')
            else:
                coord_score = self.targets_neutralized / self.total_collisions
                object.__setattr__(self, 'coordination_score', coord_score)
                object.__setattr__(self, 'coordination_str', f"{coord_score:.2f}")

    @property
    def total_reward(self) -> float:
        return sum(self.agent_rewards.values())

    @property
    def best_steps(self) -> int:
        """Alias for steps (backward compatibility)."""
        return self.steps


@dataclass(frozen=True)
class PolicyRunSummary:
    """Summary of one policy across all of its episodes."""

    mode: str
    episode_count: int
    avg_steps: float
    avg_targets: float
    avg_ammo: float
    avg_overkill: float
    avg_reward: float
    avg_net_damage: float
    avg_gross_damage: float
    success_count: int
    success_rate: float
    ammo_eff: float
    dmg_eff: float
    representative_episode: Optional[EpisodeMetrics]


class MetricsManager:
    """Calculate metrics for a single policy run."""

    def __init__(self, mode: str) -> None:
        if mode not in {"episodic", "continuous"}:
            raise ValueError(f"Unsupported mode: {mode!r}")
        self.mode = mode

    def calc_episode_metrics(self, metrics: EpisodeMetrics) -> EpisodeMetrics:
        """
        Accept EpisodeMetrics and return it as-is.
        Calculations are now done in EpisodeMetrics.__post_init__().
        This method exists for backward compatibility with calc_total_episodes_metrics().
        """
        return metrics

    def calc_total_episodes_metrics(
        self,
        episode_metrics: Sequence[MetricSourceInput],
    ) -> PolicyRunSummary:
        normalized = [
            item if isinstance(item, EpisodeMetrics) else self.calc_episode_metrics(item)
            for item in episode_metrics
        ]
        if not normalized:
            return PolicyRunSummary(
                mode=self.mode,
                episode_count=0,
                avg_steps=0.0,
                avg_targets=0.0,
                avg_ammo=0.0,
                avg_overkill=0.0,
                avg_reward=0.0,
                avg_net_damage=0.0,
                avg_gross_damage=0.0,
                success_count=0,
                success_rate=0.0,
                ammo_eff=0.0,
                dmg_eff=0.0,
                representative_episode=None,
            )

        episode_count = len(normalized)
        avg_steps = sum(item.steps for item in normalized) / episode_count
        avg_targets = sum(item.targets_neutralized for item in normalized) / episode_count
        avg_ammo = sum(item.total_ammo_used for item in normalized) / episode_count
        avg_overkill = sum(item.total_overkill for item in normalized) / episode_count
        avg_reward = sum(item.total_reward for item in normalized) / episode_count
        avg_net_damage = sum(item.total_net_damage for item in normalized) / episode_count
        avg_gross_damage = sum(item.total_gross_damage for item in normalized) / episode_count
        success_count = sum(
            1 for item in normalized if item.done_reason == "all_targets_neutralized"
        )
        success_rate = (success_count / episode_count) * 100 if episode_count > 0 else 0.0
        ammo_eff = avg_targets / avg_ammo if avg_ammo > 0 else 0.0
        dmg_eff = (
            avg_net_damage / avg_gross_damage
            if avg_gross_damage > 0
            else 0.0
        )
        representative_episode = (
            normalized[-1]
            if self.mode == "continuous"
            else min(normalized, key=lambda item: item.steps)
        )

        return PolicyRunSummary(
            mode=self.mode,
            episode_count=episode_count,
            avg_steps=avg_steps,
            avg_targets=avg_targets,
            avg_ammo=avg_ammo,
            avg_overkill=avg_overkill,
            avg_reward=avg_reward,
            avg_net_damage=avg_net_damage,
            avg_gross_damage=avg_gross_damage,
            success_count=success_count,
            success_rate=success_rate,
            ammo_eff=ammo_eff,
            dmg_eff=dmg_eff,
            representative_episode=representative_episode,
        )



def format_metric_display(val: Union[float, str], fmt: str = "{}") -> str:
    """Helper to format numeric values safely."""
    if isinstance(val, str):
        return val
    if val == float("inf"):
        return "N/A"
    return fmt.format(val)
