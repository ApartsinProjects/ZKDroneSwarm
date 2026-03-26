"""
Typed metrics models and manager for ZK-MRTA performance reporting.

The manager is scoped to a single policy run and owns the mode-specific
calculation logic for per-episode and cross-episode metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Union


MetricSourceInput = Union["EpisodeMetricsSource", "EpisodeMetrics"]


@dataclass(frozen=True)
class EpisodeMetricsSource:
    """Raw episode facts gathered at the runner/environment boundary."""

    episode: Optional[int]
    steps: int
    final_diagnostics: Mapping[str, Any]
    overkill_events: Sequence[Mapping[int, float]]
    agent_rewards: Dict[str, float]
    total_net_damage: float
    total_collisions: int
    weapon_damage_profile_mapping: Mapping[str, Mapping[str, float]]


@dataclass(frozen=True)
class EpisodeMetrics:
    """Fully calculated metrics for a single episode."""

    source: EpisodeMetricsSource
    mode: str
    done_reason: Optional[str]
    targets_neutralized: int
    total_ammo_used: int
    total_overkill: float
    total_net_damage: float
    total_gross_damage: float
    total_collisions: int
    ammo_eff: float
    dmg_eff: float
    shots_per_target: Optional[float] = None
    throughput: Optional[float] = None
    coordination_score: Optional[float] = None
    coordination_str: Optional[str] = None

    @property
    def episode(self) -> Optional[int]:
        return self.source.episode

    @property
    def steps(self) -> int:
        return self.source.steps

    @property
    def agent_rewards(self) -> Dict[str, float]:
        return self.source.agent_rewards

    @property
    def total_reward(self) -> float:
        return sum(self.agent_rewards.values())

    @property
    def best_steps(self) -> int:
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

    def calc_episode_metrics(self, source: MetricSourceInput) -> EpisodeMetrics:
        raw = self._coerce_source(source)
        done_reason = raw.final_diagnostics.get("done_reason")
        targets_neutralized = self._calc_targets_neutralized(raw.final_diagnostics)
        total_ammo_used = self._calc_total_ammo_used(raw.final_diagnostics)
        total_overkill = self._calc_total_overkill(raw.overkill_events)
        total_gross_damage = self._calc_total_gross_damage(
            raw.final_diagnostics,
            raw.weapon_damage_profile_mapping,
        )
        ammo_eff = targets_neutralized / total_ammo_used if total_ammo_used > 0 else 0.0
        dmg_eff = (
            raw.total_net_damage / total_gross_damage
            if total_gross_damage > 0
            else 0.0
        )

        if self.mode == "continuous":
            throughput = (
                targets_neutralized / raw.steps * 100
                if raw.steps > 0
                else 0.0
            )
            if raw.total_collisions == 0:
                coordination_score = float("inf")
                coordination_str = "N/A"
            else:
                coordination_score = targets_neutralized / raw.total_collisions
                coordination_str = f"{coordination_score:.2f}"
            return EpisodeMetrics(
                source=raw,
                mode=self.mode,
                done_reason=done_reason,
                targets_neutralized=targets_neutralized,
                total_ammo_used=total_ammo_used,
                total_overkill=total_overkill,
                total_net_damage=raw.total_net_damage,
                total_gross_damage=total_gross_damage,
                total_collisions=raw.total_collisions,
                ammo_eff=ammo_eff,
                dmg_eff=dmg_eff,
                throughput=throughput,
                coordination_score=coordination_score,
                coordination_str=coordination_str,
            )

        shots_per_target = (
            total_ammo_used / targets_neutralized
            if targets_neutralized > 0
            else 0.0
        )
        return EpisodeMetrics(
            source=raw,
            mode=self.mode,
            done_reason=done_reason,
            targets_neutralized=targets_neutralized,
            total_ammo_used=total_ammo_used,
            total_overkill=total_overkill,
            total_net_damage=raw.total_net_damage,
            total_gross_damage=total_gross_damage,
            total_collisions=raw.total_collisions,
            ammo_eff=ammo_eff,
            dmg_eff=dmg_eff,
            shots_per_target=shots_per_target,
        )

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

    @staticmethod
    def _coerce_source(source: MetricSourceInput) -> EpisodeMetricsSource:
        if isinstance(source, EpisodeMetrics):
            return source.source
        if isinstance(source, EpisodeMetricsSource):
            return source
        raise TypeError("MetricsManager expects EpisodeMetricsSource or EpisodeMetrics inputs")

    @staticmethod
    def _calc_targets_neutralized(final_diagnostics: Mapping[str, Any]) -> int:
        if "cumulative_neutralizations" in final_diagnostics:
            return int(final_diagnostics["cumulative_neutralizations"])
        return sum(1 for active in final_diagnostics.get("target_active", []) if not active)

    @staticmethod
    def _calc_total_ammo_used(final_diagnostics: Mapping[str, Any]) -> int:
        ammo_used = final_diagnostics.get("ammo_used", {})
        return sum(ammo_used.values())

    @staticmethod
    def _calc_total_overkill(overkill_events: Sequence[Mapping[int, float]]) -> float:
        return sum(sum(event.values()) for event in overkill_events)

    @staticmethod
    def _calc_total_gross_damage(
        final_diagnostics: Mapping[str, Any],
        weapon_damage_profile_mapping: Mapping[str, Mapping[str, float]],
    ) -> float:
        ammo_used = final_diagnostics.get("ammo_used", {})
        weapon_types = final_diagnostics.get("weapon_types", [])
        total_gross_damage = 0.0
        for agent_id, ammo in ammo_used.items():
            parts = agent_id.split("_")
            if len(parts) != 2 or not parts[1].isdigit():
                raise ValueError(f"Invalid agent id format: {agent_id!r}")
            agent_idx = int(parts[1])
            if agent_idx < 0 or agent_idx >= len(weapon_types):
                raise ValueError(
                    f"Agent index {agent_idx} out of range for weapon_types length {len(weapon_types)}"
                )
            weapon_type = weapon_types[agent_idx]
            if weapon_type not in weapon_damage_profile_mapping:
                raise KeyError(f"Missing weapon profile for weapon type: {weapon_type!r}")
            damage_per_shot = sum(weapon_damage_profile_mapping[weapon_type].values())
            total_gross_damage += ammo * damage_per_shot
        return total_gross_damage


def format_metric_display(val: Union[float, str], fmt: str = "{}") -> str:
    """Helper to format numeric values safely."""
    if isinstance(val, str):
        return val
    if val == float("inf"):
        return "N/A"
    return fmt.format(val)
