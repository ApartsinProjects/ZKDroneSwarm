import math

import pytest

from tabula_drone.utils.metrics_manager import EpisodeMetricsSource, MetricsManager


def build_source(
    *,
    episode: int = 1,
    steps: int = 5,
    done_reason: str = "all_targets_neutralized",
    cumulative_neutralizations: int = 2,
    ammo_used: dict[str, int] | None = None,
    weapon_types: list[str] | None = None,
    overkill_events: tuple[dict[int, float], ...] = ({0: 1.5},),
    total_effective_damage: float = 9.0,
    total_collisions: int = 1,
) -> EpisodeMetricsSource:
    return EpisodeMetricsSource(
        episode=episode,
        steps=steps,
        final_diagnostics={
            "done_reason": done_reason,
            "cumulative_neutralizations": cumulative_neutralizations,
            "ammo_used": ammo_used if ammo_used is not None else {"drone_0": 2, "drone_1": 1},
            "weapon_types": weapon_types if weapon_types is not None else ["light", "medium"],
            "target_active": [False, False],
        },
        overkill_events=overkill_events,
        agent_rewards={"drone_0": 1.0, "drone_1": 2.0},
        total_effective_damage=total_effective_damage,
        total_collisions=total_collisions,
        weapon_damage_profile_mapping={
            "light": {"hp": 3.0},
            "medium": {"hp": 4.0},
        },
    )


def test_calc_episode_metrics_episodic_formulas() -> None:
    manager = MetricsManager("episodic")
    source = build_source()

    metrics = manager.calc_episode_metrics(source)

    assert metrics.mode == "episodic"
    assert metrics.targets_neutralized == 2
    assert metrics.total_ammo_used == 3
    assert metrics.total_potential_damage == 10.0
    assert metrics.total_overkill == 1.5
    assert metrics.shots_per_target == pytest.approx(1.5)
    assert metrics.ammo_eff == pytest.approx(2 / 3)
    assert metrics.dmg_eff == pytest.approx(0.9)
    assert metrics.throughput is None
    assert metrics.coordination_score is None


def test_calc_episode_metrics_continuous_formulas() -> None:
    manager = MetricsManager("continuous")
    source = build_source(steps=4, cumulative_neutralizations=2, total_collisions=0)

    metrics = manager.calc_episode_metrics(source)

    assert metrics.mode == "continuous"
    assert metrics.shots_per_target is None
    assert metrics.throughput == pytest.approx(50.0)
    assert math.isinf(metrics.coordination_score)
    assert metrics.coordination_str == "N/A"


def test_calc_total_episodes_metrics_episodic_summary_and_representative() -> None:
    manager = MetricsManager("episodic")
    episode_1 = build_source(episode=1, steps=10, cumulative_neutralizations=2)
    episode_2 = build_source(episode=2, steps=6, cumulative_neutralizations=1, done_reason="max_steps_reached")

    summary = manager.calc_total_episodes_metrics([episode_1, episode_2])

    assert summary.mode == "episodic"
    assert summary.episode_count == 2
    assert summary.avg_steps == pytest.approx(8.0)
    assert summary.avg_targets == pytest.approx(1.5)
    assert summary.success_count == 1
    assert summary.success_rate == pytest.approx(50.0)
    assert summary.representative_episode is not None
    assert summary.representative_episode.episode == 2


def test_calc_total_episodes_metrics_continuous_representative_is_last() -> None:
    manager = MetricsManager("continuous")
    episode_1 = build_source(episode=1, steps=10)
    episode_2 = build_source(episode=2, steps=6)

    summary = manager.calc_total_episodes_metrics([episode_1, episode_2])

    assert summary.representative_episode is not None
    assert summary.representative_episode.episode == 2


def test_calc_total_potential_damage_fails_for_invalid_agent_id() -> None:
    manager = MetricsManager("episodic")
    source = build_source(ammo_used={"invalid_agent": 1})

    with pytest.raises(ValueError, match="Invalid agent id format"):
        manager.calc_episode_metrics(source)


def test_calc_total_potential_damage_fails_for_out_of_range_agent_index() -> None:
    manager = MetricsManager("episodic")
    source = build_source(
        ammo_used={"drone_2": 1},
        weapon_types=["light", "medium"],
    )

    with pytest.raises(ValueError, match="out of range"):
        manager.calc_episode_metrics(source)


def test_calc_total_potential_damage_fails_for_missing_weapon_profile() -> None:
    manager = MetricsManager("episodic")
    source = EpisodeMetricsSource(
        episode=1,
        steps=1,
        final_diagnostics={
            "done_reason": "max_steps_reached",
            "cumulative_neutralizations": 0,
            "ammo_used": {"drone_0": 1},
            "weapon_types": ["heavy"],
            "target_active": [True],
        },
        overkill_events=(),
        agent_rewards={"drone_0": 0.0},
        total_effective_damage=0.0,
        total_collisions=0,
        weapon_damage_profile_mapping={"light": {"hp": 3.0}},
    )

    with pytest.raises(KeyError, match="Missing weapon profile"):
        manager.calc_episode_metrics(source)
