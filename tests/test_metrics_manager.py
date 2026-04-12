import math

import pytest

from tabula_drone.utils.metrics_manager import EpisodeMetrics, MetricsManager


def build_metrics(
    *,
    episode: int = 1,
    steps: int = 5,
    done_reason: str = "all_targets_neutralized",
    targets_neutralized: int = 2,
    total_ammo_used: int = 3,
    total_overkill: float = 1.5,
    total_net_damage: float = 9.0,
    total_gross_damage: float = 10.0,
    total_collisions: int = 1,
) -> EpisodeMetrics:
    return EpisodeMetrics(
        episode=episode,
        steps=steps,
        done_reason=done_reason,
        targets_neutralized=targets_neutralized,
        total_ammo_used=total_ammo_used,
        total_overkill=total_overkill,
        total_net_damage=total_net_damage,
        total_gross_damage=total_gross_damage,
        total_collisions=total_collisions,
        agent_rewards={"drone_0": 1.0, "drone_1": 2.0},
        weapon_damage_profile_mapping={
            "light": {"hp": 3.0},
            "medium": {"hp": 4.0},
        },
    )


def test_calc_episode_metrics_episodic_formulas() -> None:
    manager = MetricsManager()
    metrics = build_metrics()
    assert metrics.targets_neutralized == 2
    assert metrics.total_ammo_used == 3
    assert metrics.total_gross_damage == 10.0
    assert metrics.total_net_damage == 9.0
    assert metrics.total_overkill == 1.5
    assert metrics.shots_per_target == pytest.approx(1.5)


def test_calc_total_episodes_metrics_episodic_summary_and_representative() -> None:
    manager = MetricsManager()
    episode_1 = build_metrics(episode=1, steps=10, targets_neutralized=2)
    episode_2 = build_metrics(episode=2, steps=6, targets_neutralized=1, done_reason="max_steps_reached")

    summary = manager.calc_total_episodes_metrics([episode_1, episode_2])

    assert summary.episode_count == 2
    assert summary.avg_steps == pytest.approx(8.0)
    assert summary.avg_targets == pytest.approx(1.5)
    assert summary.success_count == 1
    assert summary.success_rate == pytest.approx(50.0)
    assert summary.representative_episode is not None
    assert summary.representative_episode.episode == 2


# Removed obsolete validation tests - gross damage is now provided directly by runner
