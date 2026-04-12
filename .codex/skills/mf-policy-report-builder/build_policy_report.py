#!/usr/bin/env python3
"""Build deterministic MF policy reports from completed run artifacts."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


PRIMARY_POLICY = "matrix_factorization_cf"

METRIC_CATEGORIES: dict[str, str] = {
    "steps": "efficiency",
    "total_ammo_used": "efficiency",
    "shots_per_target": "efficiency",
    "total_overkill": "precision",
    "total_net_damage": "precision",
    "total_gross_damage": "precision",
    "total_collisions": "coordination",
    "targets_neutralized": "task_completion",
    "success": "task_completion",
    "mean_reward_per_agent": "reward",
    "total_latent_mismatch": "precision",
    "latent_mismatch_ratio": "precision",
}


class ReportBuilderError(RuntimeError):
    """Raised when report generation cannot proceed safely."""


@dataclass(frozen=True)
class BuilderPaths:
    """Filesystem paths required to analyze a completed MF run."""

    repo_root: Path
    run_dir: Path
    environment_path: Path
    primary_policy_dir: Path
    primary_summary_path: Path


@dataclass
class ValidationState:
    """Collect warnings and errors while keeping report generation deterministic."""

    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    unavailable_optional_fields: list[str] = field(default_factory=list)

    def add_warning(self, message: str) -> None:
        """Record a validation warning."""
        self.warnings.append(message)

    def add_error(self, message: str) -> None:
        """Record a validation error."""
        self.errors.append(message)

    def mark_optional_unavailable(self, field_name: str) -> None:
        """Record an optional field that is not persisted in the artifacts."""
        if field_name not in self.unavailable_optional_fields:
            self.unavailable_optional_fields.append(field_name)

    def status(self) -> str:
        """Return the aggregate validation status."""
        if self.errors:
            return "error"
        if self.warnings:
            return "warning"
        return "ok"


@dataclass(frozen=True)
class LoadedEpisode:
    """Loaded episode artifact plus its source path."""

    path: Path
    data: dict[str, Any]


@dataclass(frozen=True)
class LoadedLearningState:
    """Loaded learning-state artifact plus its source path."""

    path: Path
    data: dict[str, Any]


@dataclass(frozen=True)
class BaselineArtifacts:
    """Representative baseline artifact selection for a same-run policy."""

    policy_type: str
    episode: LoadedEpisode


@dataclass(frozen=True)
class RunArtifacts:
    """Loaded artifacts required for later report construction."""

    paths: BuilderPaths
    environment: dict[str, Any]
    primary_summary: dict[str, Any]
    primary_episodes: list[LoadedEpisode]
    best_episode: LoadedEpisode
    learning_states: list[LoadedLearningState]
    baselines: dict[str, BaselineArtifacts]
    validation: ValidationState


REPORT_VERSION = "1.0"
BUILDER_VERSION = "0.5.0"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for run selection and output selection."""
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic policy_report.json for a completed "
            "matrix_factorization_cf run."
        )
    )
    parser.add_argument(
        "--run",
        help=(
            "Run directory name like run_20260411_162952 or a path to a run directory. "
            "Defaults to the latest logs/run_* directory."
        ),
    )
    parser.add_argument(
        "--output",
        help=(
            "Optional output path. Defaults to <run_dir>/policy_report.json once report "
            "generation is fully implemented."
        ),
    )
    return parser.parse_args(argv)


def repo_root_from_script(script_path: Path) -> Path:
    """Resolve the repository root from this skill-local script path."""
    return script_path.resolve().parents[3]


def find_latest_run_dir(logs_dir: Path) -> Path:
    """Return the most recent run directory under logs/."""
    run_dirs = sorted(
        (path for path in logs_dir.iterdir() if path.is_dir() and path.name.startswith("run_")),
        key=lambda path: path.name,
    )
    if not run_dirs:
        raise ReportBuilderError(f"No run directories found under {logs_dir}")
    return run_dirs[-1]


def resolve_run_dir(repo_root: Path, run_arg: str | None) -> Path:
    """Resolve an explicit run path/name or default to the latest run directory."""
    logs_dir = repo_root / "logs"
    if not logs_dir.is_dir():
        raise ReportBuilderError(f"Logs directory not found: {logs_dir}")

    if not run_arg:
        return find_latest_run_dir(logs_dir)

    candidate = Path(run_arg)
    if not candidate.is_absolute():
        if candidate.parts and candidate.parts[0] == "logs":
            candidate = repo_root / candidate
        elif candidate.name.startswith("run_") and len(candidate.parts) == 1:
            candidate = logs_dir / candidate
        else:
            candidate = repo_root / candidate

    candidate = candidate.resolve()
    if not candidate.is_dir():
        raise ReportBuilderError(f"Run directory not found: {candidate}")
    return candidate


def resolve_builder_paths(script_path: Path, run_arg: str | None) -> BuilderPaths:
    """Resolve the target run and the required MF entry points inside it."""
    repo_root = repo_root_from_script(script_path)
    run_dir = resolve_run_dir(repo_root, run_arg)
    environment_path = run_dir / "environment.json"
    primary_policy_dir = run_dir / PRIMARY_POLICY
    primary_summary_path = primary_policy_dir / "episodes_summary.json"

    missing_paths = [
        path
        for path in (environment_path, primary_policy_dir, primary_summary_path)
        if not path.exists()
    ]
    if missing_paths:
        formatted = ", ".join(str(path) for path in missing_paths)
        raise ReportBuilderError(
            "Required matrix_factorization_cf artifacts are missing: " + formatted
        )

    return BuilderPaths(
        repo_root=repo_root,
        run_dir=run_dir,
        environment_path=environment_path,
        primary_policy_dir=primary_policy_dir,
        primary_summary_path=primary_summary_path,
    )


def load_json_file(path: Path, validation: ValidationState, label: str) -> dict[str, Any]:
    """Load a JSON artifact and convert parse failures into validation errors."""
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        validation.add_error(f"Missing {label}: {path}")
    except json.JSONDecodeError as exc:
        validation.add_error(f"Invalid JSON in {label} {path}: {exc}")
    return {}


def extract_sequence_number(path: Path, pattern: str) -> int:
    """Extract a deterministic numeric sort key from an artifact filename."""
    match = re.search(pattern, path.name)
    if not match:
        return -1
    return int(match.group(1))


def sorted_episode_paths(policy_dir: Path) -> list[Path]:
    """Return ordered episode artifact paths for a policy directory."""
    episodes_dir = policy_dir / "episodes"
    if not episodes_dir.is_dir():
        return []
    return sorted(
        episodes_dir.glob("episode_*.json"),
        key=lambda path: (extract_sequence_number(path, r"ep(\d+)"), path.name),
    )


def sorted_learning_state_paths(policy_dir: Path) -> list[Path]:
    """Return ordered learning-state artifact paths for the MF policy directory."""
    learning_state_dir = policy_dir / "learning_state"
    if not learning_state_dir.is_dir():
        return []
    return sorted(
        learning_state_dir.glob("learning_state_*.json"),
        key=lambda path: (extract_sequence_number(path, r"ep(\d+)"), path.name),
    )


def validate_scenario_id(
    data: dict[str, Any],
    expected_scenario_id: str,
    validation: ValidationState,
    label: str,
) -> None:
    """Validate that an artifact belongs to the same scenario/run."""
    artifact_scenario_id = data.get("scenario_id")
    if artifact_scenario_id != expected_scenario_id:
        validation.add_error(
            f"{label} scenario_id mismatch: expected {expected_scenario_id}, got "
            f"{artifact_scenario_id!r}"
        )


def load_primary_episodes(
    paths: BuilderPaths,
    expected_scenario_id: str,
    validation: ValidationState,
) -> list[LoadedEpisode]:
    """Load and validate all MF episode artifacts."""
    episode_paths = sorted_episode_paths(paths.primary_policy_dir)
    if not episode_paths:
        validation.add_error(
            f"No episode artifacts found under {paths.primary_policy_dir / 'episodes'}"
        )
        return []

    loaded: list[LoadedEpisode] = []
    expected_episode_num = 1
    for episode_path in episode_paths:
        episode_data = load_json_file(episode_path, validation, "episode artifact")
        if not episode_data:
            continue

        validate_scenario_id(episode_data, expected_scenario_id, validation, episode_path.name)

        policy_type = episode_data.get("config", {}).get("policy_type")
        if policy_type != PRIMARY_POLICY:
            validation.add_error(
                f"{episode_path.name} policy_type mismatch: expected {PRIMARY_POLICY}, "
                f"got {policy_type!r}"
            )

        episode_num = episode_data.get("episode_num")
        if episode_num != expected_episode_num:
            validation.add_warning(
                f"Episode numbering gap or reordering detected: expected "
                f"{expected_episode_num}, got {episode_num!r} in {episode_path.name}"
            )
            expected_episode_num = episode_num if isinstance(episode_num, int) else expected_episode_num

        loaded.append(LoadedEpisode(path=episode_path, data=episode_data))
        if isinstance(episode_num, int):
            expected_episode_num = episode_num + 1

    return loaded


def resolve_best_episode(
    paths: BuilderPaths,
    primary_summary: dict[str, Any],
    primary_episodes: list[LoadedEpisode],
    validation: ValidationState,
) -> LoadedEpisode:
    """Resolve and validate the canonical MF best episode from episodes_summary.json."""
    best_episode_relpath = primary_summary.get("best_episode_path")
    if not isinstance(best_episode_relpath, str) or not best_episode_relpath:
        validation.add_error("episodes_summary.json is missing best_episode_path")
        if primary_episodes:
            return primary_episodes[-1]
        raise ReportBuilderError("Unable to resolve MF best episode")

    best_episode_path = paths.primary_policy_dir / best_episode_relpath
    for episode in primary_episodes:
        if episode.path == best_episode_path:
            return episode

    validation.add_error(f"best_episode_path does not resolve to a loaded episode: {best_episode_path}")
    if primary_episodes:
        return primary_episodes[-1]
    raise ReportBuilderError("Unable to resolve MF best episode")


def load_learning_states(
    paths: BuilderPaths,
    expected_scenario_id: str,
    validation: ValidationState,
) -> list[LoadedLearningState]:
    """Load and lightly validate MF learning-state artifacts."""
    learning_state_paths = sorted_learning_state_paths(paths.primary_policy_dir)
    if not learning_state_paths:
        validation.add_warning(
            f"No learning-state artifacts found under {paths.primary_policy_dir / 'learning_state'}"
        )
        return []

    loaded: list[LoadedLearningState] = []
    for state_path in learning_state_paths:
        state_data = load_json_file(state_path, validation, "learning-state artifact")
        if not state_data:
            continue

        validate_scenario_id(state_data, expected_scenario_id, validation, state_path.name)
        if state_data.get("policy_type") != PRIMARY_POLICY:
            validation.add_error(
                f"{state_path.name} policy_type mismatch: expected {PRIMARY_POLICY}, "
                f"got {state_data.get('policy_type')!r}"
            )

        loaded.append(LoadedLearningState(path=state_path, data=state_data))

    return loaded


def discover_baseline_artifacts(
    paths: BuilderPaths,
    expected_scenario_id: str,
    validation: ValidationState,
) -> dict[str, BaselineArtifacts]:
    """Load representative baseline episodes from same-run policy folders."""
    baselines: dict[str, BaselineArtifacts] = {}
    for child in sorted(paths.run_dir.iterdir(), key=lambda path: path.name):
        if not child.is_dir() or child.name.startswith(".") or child.name == PRIMARY_POLICY:
            continue

        episode_paths = sorted_episode_paths(child)
        if not episode_paths:
            continue

        episode_path = episode_paths[0]
        episode_data = load_json_file(
            episode_path,
            validation,
            f"baseline episode artifact for {child.name}",
        )
        if not episode_data:
            continue

        validate_scenario_id(episode_data, expected_scenario_id, validation, episode_path.name)
        policy_type = episode_data.get("config", {}).get("policy_type")
        if policy_type and policy_type != child.name:
            validation.add_warning(
                f"Baseline policy folder/name mismatch: folder {child.name}, policy_type {policy_type!r}"
            )

        baselines[child.name] = BaselineArtifacts(
            policy_type=policy_type or child.name,
            episode=LoadedEpisode(path=episode_path, data=episode_data),
        )

    if not baselines:
        validation.add_warning("No same-run baseline episodes were found")

    return baselines


def collect_run_artifacts(paths: BuilderPaths) -> RunArtifacts:
    """Load the artifact set needed for deterministic report assembly."""
    validation = ValidationState()
    environment = load_json_file(paths.environment_path, validation, "environment artifact")
    primary_summary = load_json_file(paths.primary_summary_path, validation, "primary summary")

    expected_scenario_id = str(environment.get("scenario_id") or paths.run_dir.name)
    primary_episodes = load_primary_episodes(paths, expected_scenario_id, validation)
    best_episode = resolve_best_episode(paths, primary_summary, primary_episodes, validation)
    learning_states = load_learning_states(paths, expected_scenario_id, validation)
    baselines = discover_baseline_artifacts(paths, expected_scenario_id, validation)

    return RunArtifacts(
        paths=paths,
        environment=environment,
        primary_summary=primary_summary,
        primary_episodes=primary_episodes,
        best_episode=best_episode,
        learning_states=learning_states,
        baselines=baselines,
        validation=validation,
    )


def mean_or_none(values: list[float]) -> float | None:
    """Return the arithmetic mean for non-empty numeric lists."""
    if not values:
        return None
    return sum(values) / len(values)


def relative_to_run(paths: BuilderPaths, path: Path) -> str:
    """Return a run-relative path for portability."""
    return str(path.relative_to(paths.run_dir))


def compute_relative_change(mf_value: Any, baseline_value: Any) -> float | None:
    """Compute raw relative change percentage using the baseline as reference."""
    if isinstance(mf_value, bool) or isinstance(baseline_value, bool):
        return None
    if not isinstance(mf_value, (int, float)) or not isinstance(baseline_value, (int, float)):
        return None
    if baseline_value == 0:
        return 0.0 if mf_value == 0 else None
    return ((float(mf_value) - float(baseline_value)) / abs(float(baseline_value))) * 100.0


def last_step_record(episode_data: dict[str, Any]) -> dict[str, Any]:
    """Return the final step record for an episode, if present."""
    steps = episode_data.get("steps")
    if isinstance(steps, list) and steps:
        return steps[-1]
    return {}


def metric_source_name(episode_data: dict[str, Any], field_name: str) -> str | None:
    """Describe where an episode field came from."""
    metrics = episode_data.get("metrics", {})
    summary = episode_data.get("summary", {})
    if field_name in metrics:
        return "summary.metrics"
    if field_name in summary:
        return "summary"
    return None


def extract_episode_entry(
    episode: LoadedEpisode,
    validation: ValidationState,
) -> dict[str, Any]:
    """Convert a raw episode artifact into a compact report entry."""
    data = episode.data
    metrics = data.get("metrics", {})
    summary = data.get("summary", {})
    step_tail = last_step_record(data)
    step_info = step_tail.get("info", {}) if isinstance(step_tail, dict) else {}
    policy_type = data.get("config", {}).get("policy_type")
    if not policy_type:
        validation.add_warning(f"{episode.path.name} is missing config.policy_type")

    agent_rewards = metrics.get("agent_rewards")
    if not isinstance(agent_rewards, dict):
        agent_rewards = summary.get("total_reward") if isinstance(summary.get("total_reward"), dict) else {}

    total_reward = None
    mean_reward = None
    if isinstance(agent_rewards, dict) and agent_rewards:
        reward_values = [
            float(value)
            for value in agent_rewards.values()
            if isinstance(value, (int, float))
        ]
        if reward_values:
            total_reward = sum(reward_values)
            mean_reward = total_reward / len(reward_values)

    target_active = step_info.get("target_active") if isinstance(step_info, dict) else None
    end_active_target_count = None
    if isinstance(target_active, list):
        end_active_target_count = sum(1 for value in target_active if bool(value))

    entry = {
        "episode": metrics.get("episode", data.get("episode_num")),
        "steps": metrics.get("steps", summary.get("total_steps", len(data.get("steps", [])))),
        "success": summary.get("success"),
        "targets_neutralized": metrics.get("targets_neutralized"),
        "total_ammo_used": metrics.get("total_ammo_used"),
        "total_collisions": metrics.get("total_collisions"),
        "total_overkill": metrics.get("total_overkill"),
        "total_gross_damage": metrics.get("total_gross_damage"),
        "total_net_damage": metrics.get("total_net_damage"),
        "shots_per_target": metrics.get("shots_per_target"),
        "total_latent_mismatch": metrics.get("total_latent_mismatch"),
        "latent_mismatch_ratio": metrics.get("latent_mismatch_ratio"),
        "mean_reward_per_agent": mean_reward,
        "total_reward": total_reward,
        "end_of_episode_active_target_count": end_active_target_count,
        "termination_flag": step_tail.get("terminated"),
        "truncation_flag": step_tail.get("truncated"),
        "done_reason": metrics.get("done_reason", summary.get("termination_reason")),
    }
    return entry


def build_config_snapshot(artifacts: RunArtifacts) -> dict[str, Any]:
    """Build the compact run configuration snapshot from environment artifacts."""
    config_snapshot: dict[str, Any] = {}
    environment = artifacts.environment
    config = environment.get("config", {})
    scenario = environment.get("scenario", {})
    latent_world_config = scenario.get("latent_world", {}).get("config", {})

    if "world_size" in config:
        config_snapshot["world_size"] = config["world_size"]
    if "max_steps" in config:
        config_snapshot["max_steps"] = config["max_steps"]
    if "num_drones" in scenario:
        config_snapshot["num_agents"] = scenario["num_drones"]
    if "num_targets" in scenario:
        config_snapshot["num_targets"] = scenario["num_targets"]

    for field_name in (
        "target_hp",
        "latent_dim",
        "center_mode",
        "drone_variance",
        "target_variance",
        "mode",
        "num_modes",
    ):
        if field_name in latent_world_config:
            config_snapshot[field_name] = latent_world_config[field_name]

    for missing_field in ("reward_noise", "observation_noise"):
        if missing_field not in config and missing_field not in latent_world_config:
            artifacts.validation.mark_optional_unavailable(missing_field)

    return config_snapshot


def build_artifacts_section(artifacts: RunArtifacts) -> dict[str, Any]:
    """Record the compact artifact set used to build the report."""
    checkpoint_episodes: list[int] = []
    if artifacts.learning_states:
        learning_episode_numbers = [
            int(item.data.get("episode_num"))
            for item in artifacts.learning_states
            if isinstance(item.data.get("episode_num"), int)
        ]
        if learning_episode_numbers:
            first = learning_episode_numbers[0]
            middle = learning_episode_numbers[len(learning_episode_numbers) // 2]
            final = learning_episode_numbers[-1]
            best = artifacts.best_episode.data.get("episode_num")
            for episode_num in (first, middle, final, best):
                if isinstance(episode_num, int) and episode_num not in checkpoint_episodes:
                    checkpoint_episodes.append(episode_num)

    checkpoint_paths = {}
    for item in artifacts.learning_states:
        episode_num = item.data.get("episode_num")
        if isinstance(episode_num, int) and episode_num in checkpoint_episodes:
            checkpoint_paths[f"episode_{episode_num:02d}"] = relative_to_run(artifacts.paths, item.path)

    return {
        "run_dir": artifacts.paths.run_dir.name,
        "environment_path": relative_to_run(artifacts.paths, artifacts.paths.environment_path),
        "primary_summary_path": relative_to_run(artifacts.paths, artifacts.paths.primary_summary_path),
        "best_episode_path": relative_to_run(artifacts.paths, artifacts.best_episode.path),
        "primary_episode_count": len(artifacts.primary_episodes),
        "learning_state_count": len(artifacts.learning_states),
        "selected_learning_checkpoint_paths": checkpoint_paths,
        "baseline_episode_paths": {
            name: relative_to_run(artifacts.paths, item.episode.path)
            for name, item in artifacts.baselines.items()
        },
    }


TREND_METRICS = [
    "steps", "ammo_eff", "dmg_eff", "total_collisions", "total_overkill",
    "mean_reward_per_agent",
]


def compute_metric_trend(
    episodes: list[dict[str, Any]],
    metric_name: str,
) -> dict[str, Any] | None:
    """Compute a lightweight trend descriptor for a single metric across episodes."""
    values = [
        float(ep[metric_name])
        for ep in episodes
        if isinstance(ep.get(metric_name), (int, float))
    ]
    if len(values) < 3:
        return None

    first_value = values[0]
    last_value = values[-1]
    value_range = max(values) - min(values)

    if value_range == 0:
        direction = "flat"
    elif last_value < first_value:
        direction = "decreasing"
    else:
        direction = "increasing"

    monotonic_up = all(b >= a for a, b in zip(values, values[1:]))
    monotonic_down = all(b <= a for a, b in zip(values, values[1:]))
    is_monotonic = monotonic_up or monotonic_down

    tail_start = max(1, len(values) * 2 // 3)
    tail = values[tail_start:]
    if len(tail) >= 2 and value_range > 0:
        tail_mean = sum(tail) / len(tail)
        tail_std = (sum((v - tail_mean) ** 2 for v in tail) / len(tail)) ** 0.5
        is_plateau = tail_std / value_range < 0.10
    else:
        is_plateau = False

    return {
        "first_value": first_value,
        "last_value": last_value,
        "direction": direction,
        "is_monotonic": is_monotonic,
        "is_plateau": is_plateau,
    }


def build_episode_curves(artifacts: RunArtifacts) -> dict[str, Any]:
    """Build the MF per-episode performance curve section."""
    episodes = [
        extract_episode_entry(item, artifacts.validation)
        for item in artifacts.primary_episodes
    ]
    trend_summary = {}
    for metric_name in TREND_METRICS:
        trend = compute_metric_trend(episodes, metric_name)
        if trend is not None:
            trend_summary[metric_name] = trend

    return {
        "best_episode_path": relative_to_run(artifacts.paths, artifacts.best_episode.path),
        "metric_provenance": {
            "performance_metrics": "summary.metrics",
            "total_reward": "summary.total_reward",
            "mean_reward_per_agent": "derived_from_summary.total_reward",
            "end_of_episode_active_target_count": "derived_from_trace.final_step.info.target_active",
        },
        "trend_summary": trend_summary,
        "episodes": episodes,
    }


def build_policy_summary(artifacts: RunArtifacts, episode_curves: dict[str, Any]) -> dict[str, Any]:
    """Build the canonical MF policy summary anchored on the best episode."""
    episodes = episode_curves["episodes"]
    best_episode_entry = extract_episode_entry(
        artifacts.best_episode,
        artifacts.validation,
    )
    return {
        "best_episode": best_episode_entry,
        "training_episode_count": len(episodes),
        "total_training_steps": artifacts.primary_summary.get("total_steps"),
        "total_steps_to_best": artifacts.primary_summary.get("total_steps_to_best"),
        "aggregate_training": {
            "avg_steps": mean_or_none(
                [float(item["steps"]) for item in episodes if isinstance(item.get("steps"), (int, float))]
            ),
            "avg_ammo_eff": mean_or_none(
                [float(item["ammo_eff"]) for item in episodes if isinstance(item.get("ammo_eff"), (int, float))]
            ),
            "avg_dmg_eff": mean_or_none(
                [float(item["dmg_eff"]) for item in episodes if isinstance(item.get("dmg_eff"), (int, float))]
            ),
            "avg_total_ammo_used": mean_or_none(
                [
                    float(item["total_ammo_used"])
                    for item in episodes
                    if isinstance(item.get("total_ammo_used"), (int, float))
                ]
            ),
            "avg_total_collisions": mean_or_none(
                [
                    float(item["total_collisions"])
                    for item in episodes
                    if isinstance(item.get("total_collisions"), (int, float))
                ]
            ),
            "avg_total_overkill": mean_or_none(
                [
                    float(item["total_overkill"])
                    for item in episodes
                    if isinstance(item.get("total_overkill"), (int, float))
                ]
            ),
            "avg_total_reward": mean_or_none(
                [
                    float(item["total_reward"])
                    for item in episodes
                    if isinstance(item.get("total_reward"), (int, float))
                ]
            ),
            "avg_mean_reward_per_agent": mean_or_none(
                [
                    float(item["mean_reward_per_agent"])
                    for item in episodes
                    if isinstance(item.get("mean_reward_per_agent"), (int, float))
                ]
            ),
            "success_rate_pct": mean_or_none(
                [100.0 if item.get("success") else 0.0 for item in episodes if item.get("success") is not None]
            ),
        },
    }


def summarize_checkpoint_agent(agent_entry: dict[str, Any], top_k: int = 3) -> dict[str, Any]:
    """Summarize one agent's prediction state for checkpoint aggregation."""
    ranked_targets = agent_entry.get("match", {}).get("ranked_targets", [])
    predicted_rewards = agent_entry.get("match", {}).get("predicted_rewards", [])
    top_targets = ranked_targets[:top_k] if isinstance(ranked_targets, list) else []
    top_scores = [
        predicted_rewards[target]
        for target in top_targets
        if isinstance(target, int)
        and 0 <= target < len(predicted_rewards)
    ] if isinstance(predicted_rewards, list) else []

    summary = {
        "agent_idx": agent_entry.get("agent_idx"),
        "best_target": agent_entry.get("match", {}).get("best_target"),
        "top_k_targets": top_targets,
        "top_k_predicted_rewards": top_scores,
    }

    integration = agent_entry.get("integration_matrix")
    if isinstance(integration, dict):
        counts = integration.get("M_count")
        avg_matrix = integration.get("M_avg")
        pred_matrix = integration.get("M_pred")
        if isinstance(counts, list) and counts and isinstance(counts[0], list):
            total = sum(len(row) for row in counts)
            covered = sum(1 for row in counts for value in row if isinstance(value, (int, float)) and value > 0)
            coverage_ratio = (covered / total) if total else None
        else:
            coverage_ratio = None

        agreement_values: list[float] = []
        if (
            isinstance(counts, list)
            and isinstance(avg_matrix, list)
            and isinstance(pred_matrix, list)
        ):
            for row_idx, row in enumerate(counts):
                if not isinstance(row, list):
                    continue
                for col_idx, count in enumerate(row):
                    if not isinstance(count, (int, float)) or count <= 0:
                        continue
                    try:
                        avg_value = avg_matrix[row_idx][col_idx]
                        pred_value = pred_matrix[row_idx][col_idx]
                    except (IndexError, TypeError):
                        continue
                    if isinstance(avg_value, (int, float)) and isinstance(pred_value, (int, float)):
                        agreement_values.append(abs(float(avg_value) - float(pred_value)))

        summary["integration_matrix"] = {
            "coverage_ratio": coverage_ratio,
            "mean_absolute_agreement_gap": mean_or_none(agreement_values),
        }

    return summary


def build_checkpoint_summary(state: LoadedLearningState) -> dict[str, Any]:
    """Build an aggregated checkpoint summary from one learning-state artifact."""
    data = state.data
    agents = data.get("episode_state", {}).get("agents", [])
    epsilon_values = [
        float(agent["epsilon"])
        for agent in agents
        if isinstance(agent, dict) and isinstance(agent.get("epsilon"), (int, float))
    ]
    step_counts = [
        float(agent["step_count"])
        for agent in agents
        if isinstance(agent, dict) and isinstance(agent.get("step_count"), (int, float))
    ]
    agent_summaries = [
        summarize_checkpoint_agent(agent)
        for agent in agents
        if isinstance(agent, dict)
    ]
    best_targets = [
        item["best_target"]
        for item in agent_summaries
        if isinstance(item.get("best_target"), int)
    ]
    top1_rewards = [
        item["top_k_predicted_rewards"][0]
        for item in agent_summaries
        if item.get("top_k_predicted_rewards")
        and isinstance(item["top_k_predicted_rewards"][0], (int, float))
    ]
    coverage_ratios = [
        item["integration_matrix"]["coverage_ratio"]
        for item in agent_summaries
        if isinstance(item.get("integration_matrix", {}).get("coverage_ratio"), (int, float))
    ]
    agreement_gaps = [
        item["integration_matrix"]["mean_absolute_agreement_gap"]
        for item in agent_summaries
        if isinstance(
            item.get("integration_matrix", {}).get("mean_absolute_agreement_gap"),
            (int, float),
        )
    ]
    return {
        "episode": data.get("episode_num"),
        "path": state.path.name,
        "epsilon_mean": mean_or_none(epsilon_values),
        "step_count_mean": mean_or_none(step_counts),
        "agent_count": len(agent_summaries),
        "unique_best_target_count": len(set(best_targets)),
        "mean_top1_predicted_reward": mean_or_none(top1_rewards),
        "integration_matrix": {
            "mean_coverage_ratio": mean_or_none(coverage_ratios),
            "mean_absolute_agreement_gap": mean_or_none(agreement_gaps),
        },
    }


def build_learning_summary(artifacts: RunArtifacts) -> dict[str, Any] | None:
    """Build the MF learning progression section."""
    if not artifacts.learning_states:
        return None

    by_episode = {
        int(item.data.get("episode_num")): item
        for item in artifacts.learning_states
        if isinstance(item.data.get("episode_num"), int)
    }
    episode_numbers = sorted(by_episode)
    if not episode_numbers:
        artifacts.validation.add_warning("Learning-state artifacts exist but have no usable episode numbers")
        return {
            "available": False,
            "reason": "learning_state_episode_numbers_unavailable",
        }

    first_episode = episode_numbers[0]
    middle_episode = episode_numbers[len(episode_numbers) // 2]
    final_episode = episode_numbers[-1]
    best_episode_num = artifacts.best_episode.data.get("episode_num")

    checkpoint_numbers = []
    for episode_num in (first_episode, middle_episode, final_episode, best_episode_num):
        if isinstance(episode_num, int) and episode_num in by_episode and episode_num not in checkpoint_numbers:
            checkpoint_numbers.append(episode_num)

    final_state = by_episode[final_episode]
    final_agents = final_state.data.get("episode_state", {}).get("agents", [])
    epsilon_progression = [
        {
            "episode": episode_num,
            "epsilon_mean": mean_or_none(
                [
                    float(agent["epsilon"])
                    for agent in by_episode[episode_num].data.get("episode_state", {}).get("agents", [])
                    if isinstance(agent, dict) and isinstance(agent.get("epsilon"), (int, float))
                ]
            ),
        }
        for episode_num in episode_numbers
    ]
    final_epsilon = mean_or_none(
        [
            float(agent["epsilon"])
            for agent in final_agents
            if isinstance(agent, dict) and isinstance(agent.get("epsilon"), (int, float))
        ]
    )
    final_step_count = mean_or_none(
        [
            float(agent["step_count"])
            for agent in final_agents
            if isinstance(agent, dict) and isinstance(agent.get("step_count"), (int, float))
        ]
    )

    best_is_final = (
        isinstance(best_episode_num, int)
        and best_episode_num == final_episode
    )
    if best_is_final:
        convergence_status = "potentially_undertrained"
    elif isinstance(best_episode_num, int) and best_episode_num <= final_episode * 0.5:
        convergence_status = "early_peak"
    else:
        convergence_status = "converged"

    return {
        "available": True,
        "total_training_episodes": len(episode_numbers),
        "final_episode": final_episode,
        "final_epsilon": final_epsilon,
        "epsilon_progression": epsilon_progression,
        "final_step_count": final_step_count,
        "convergence_assessment": {
            "best_is_final": best_is_final,
            "best_episode": best_episode_num,
            "convergence_status": convergence_status,
        },
        "checkpoint_strategy": ["first", "middle", "final", "best_if_distinct"],
        "checkpoints": [
            build_checkpoint_summary(by_episode[episode_num])
            for episode_num in checkpoint_numbers
        ],
    }


def compare_metric(
    metric_name: str,
    mf_value: Any,
    baseline_value: Any,
    *,
    higher_is_better: bool,
    category: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic metric comparison entry."""
    relative_change_pct = compute_relative_change(mf_value, baseline_value)
    interpretation = "unavailable"
    if isinstance(mf_value, (int, float)) and isinstance(baseline_value, (int, float)):
        if mf_value == baseline_value:
            interpretation = "no_change"
        elif higher_is_better:
            interpretation = "improved" if mf_value > baseline_value else "worsened"
        else:
            interpretation = "improved" if mf_value < baseline_value else "worsened"
    return {
        "metric": metric_name,
        "category": category or METRIC_CATEGORIES.get(metric_name, "other"),
        "mf_value": mf_value,
        "baseline_value": baseline_value,
        "relative_change_pct": relative_change_pct,
        "higher_is_better": higher_is_better,
        "directionality": interpretation,
    }


def build_comparison_vs_baseline(artifacts: RunArtifacts) -> dict[str, Any]:
    """Compare the MF best episode against same-run baselines and references."""
    if not artifacts.baselines:
        return {"baselines": {}, "reference_policies": {}}

    mf_entry = extract_episode_entry(artifacts.best_episode, artifacts.validation)
    baseline_comparisons: dict[str, Any] = {}
    reference_comparisons: dict[str, Any] = {}
    for name, baseline in artifacts.baselines.items():
        baseline_entry = extract_episode_entry(
            baseline.episode,
            artifacts.validation,
        )
        metric_entries = [
            compare_metric("steps", mf_entry.get("steps"), baseline_entry.get("steps"), higher_is_better=False),
            compare_metric(
                "total_ammo_used",
                mf_entry.get("total_ammo_used"),
                baseline_entry.get("total_ammo_used"),
                higher_is_better=False,
            ),
            compare_metric(
                "shots_per_target",
                mf_entry.get("shots_per_target"),
                baseline_entry.get("shots_per_target"),
                higher_is_better=False,
            ),
            compare_metric(
                "total_collisions",
                mf_entry.get("total_collisions"),
                baseline_entry.get("total_collisions"),
                higher_is_better=False,
            ),
            compare_metric(
                "total_overkill",
                mf_entry.get("total_overkill"),
                baseline_entry.get("total_overkill"),
                higher_is_better=False,
            ),
            compare_metric(
                "total_net_damage",
                mf_entry.get("total_net_damage"),
                baseline_entry.get("total_net_damage"),
                higher_is_better=True,
            ),
            compare_metric(
                "total_gross_damage",
                mf_entry.get("total_gross_damage"),
                baseline_entry.get("total_gross_damage"),
                higher_is_better=False,
            ),
            compare_metric(
                "total_latent_mismatch",
                mf_entry.get("total_latent_mismatch"),
                baseline_entry.get("total_latent_mismatch"),
                higher_is_better=False,
            ),
            compare_metric(
                "latent_mismatch_ratio",
                mf_entry.get("latent_mismatch_ratio"),
                baseline_entry.get("latent_mismatch_ratio"),
                higher_is_better=False,
            ),
            compare_metric(
                "targets_neutralized",
                mf_entry.get("targets_neutralized"),
                baseline_entry.get("targets_neutralized"),
                higher_is_better=True,
            ),
            compare_metric("success", mf_entry.get("success"), baseline_entry.get("success"), higher_is_better=True),
        ]

        improved = sum(1 for item in metric_entries if item["directionality"] == "improved")
        worsened = sum(1 for item in metric_entries if item["directionality"] == "worsened")
        overall_label = "mixed"
        if improved > 0 and worsened == 0:
            overall_label = "improved"
        elif worsened > 0 and improved == 0:
            overall_label = "worsened"

        category_directions: dict[str, str] = {}
        for cat in dict.fromkeys(METRIC_CATEGORIES.values()):
            cat_metrics = [m for m in metric_entries if m.get("category") == cat]
            cat_improved = sum(1 for m in cat_metrics if m["directionality"] == "improved")
            cat_worsened = sum(1 for m in cat_metrics if m["directionality"] == "worsened")
            if cat_improved > 0 and cat_worsened == 0:
                category_directions[cat] = "improved"
            elif cat_worsened > 0 and cat_improved == 0:
                category_directions[cat] = "worsened"
            elif cat_improved > 0 and cat_worsened > 0:
                category_directions[cat] = "mixed"
            else:
                category_directions[cat] = "no_change"

        target = baseline_comparisons if name == "random" else reference_comparisons
        target[name] = {
            "baseline_policy": baseline.policy_type,
            "mf_episode": mf_entry["episode"],
            "baseline_episode": baseline_entry["episode"],
            "overall_label": overall_label,
            "category_directions": category_directions,
            "metrics": metric_entries,
        }

    return {
        "baselines": baseline_comparisons,
        "reference_policies": reference_comparisons,
    }


def build_key_findings(
    comparison: dict[str, Any],
    episode_curves: dict[str, Any],
    learning_summary: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Surface the most actionable findings from the assembled report sections."""
    findings: list[dict[str, Any]] = []

    all_comparisons: dict[str, dict[str, Any]] = {}
    for section_key in ("baselines", "reference_policies"):
        for name, comp in comparison.get(section_key, {}).items():
            all_comparisons[name] = comp

    if len(all_comparisons) >= 2:
        metric_names: set[str] = set()
        for comp in all_comparisons.values():
            for m in comp.get("metrics", []):
                metric_names.add(m["metric"])
        for metric_name in sorted(metric_names):
            worsened_vs = []
            for name, comp in all_comparisons.items():
                for m in comp.get("metrics", []):
                    if m["metric"] == metric_name and m["directionality"] == "worsened":
                        worsened_vs.append(name)
            if len(worsened_vs) == len(all_comparisons):
                findings.append({
                    "type": "cross_baseline_regression",
                    "metric": metric_name,
                    "category": METRIC_CATEGORIES.get(metric_name, "other"),
                    "worsened_vs": worsened_vs,
                })

    trend_summary = episode_curves.get("trend_summary", {})
    for metric_name, trend in trend_summary.items():
        higher_is_better = metric_name in ("ammo_eff", "dmg_eff", "mean_reward_per_agent")
        degrading = (
            (trend["direction"] == "decreasing" and higher_is_better)
            or (trend["direction"] == "increasing" and not higher_is_better)
        )
        if degrading and not trend.get("is_plateau", False):
            findings.append({
                "type": "active_degradation",
                "metric": metric_name,
                "category": METRIC_CATEGORIES.get(metric_name, "other"),
                "direction": trend["direction"],
                "first_value": trend["first_value"],
                "last_value": trend["last_value"],
            })

    if isinstance(learning_summary, dict) and learning_summary.get("available"):
        convergence = learning_summary.get("convergence_assessment", {})
        if convergence.get("convergence_status") == "potentially_undertrained":
            findings.append({
                "type": "convergence_warning",
                "status": "potentially_undertrained",
                "best_episode": convergence.get("best_episode"),
            })

        checkpoints = learning_summary.get("checkpoints", [])
        for cp in checkpoints:
            agent_count = cp.get("agent_count", 0)
            unique_targets = cp.get("unique_best_target_count", agent_count)
            if agent_count > 0 and unique_targets < agent_count * 0.6:
                findings.append({
                    "type": "target_crowding",
                    "episode": cp.get("episode"),
                    "agent_count": agent_count,
                    "unique_best_target_count": unique_targets,
                })

    return findings


def build_metric_definitions() -> dict[str, dict[str, Any]]:
    """Return canonical metric definitions with formulas and interpretations."""
    return {
        "steps": {
            "description": "Number of timesteps from episode start to termination",
            "formula": "count(timesteps)",
            "source": "summary.total_steps",
            "direction": "lower_is_better",
            "category": "efficiency",
        },
        "targets_neutralized": {
            "description": "Number of targets reduced to zero HP",
            "formula": "count(targets where hp <= 0)",
            "source": "metrics.targets_neutralized",
            "direction": "higher_is_better",
            "category": "task_completion",
        },
        "total_ammo_used": {
            "description": "Total shots fired by all agents across the episode",
            "formula": "sum(ammo_used per agent)",
            "source": "metrics.total_ammo_used",
            "direction": "lower_is_better",
            "category": "efficiency",
        },
        "total_collisions": {
            "description": "Number of times multiple agents targeted the same target simultaneously",
            "formula": "sum(collision events)",
            "source": "metrics.total_collisions",
            "direction": "lower_is_better",
            "category": "coordination",
        },
        "total_overkill": {
            "description": "Cumulative wasted damage dealt to targets already at zero HP",
            "formula": "sum(max(0, damage - remaining_hp) per shot)",
            "source": "metrics.total_overkill",
            "direction": "lower_is_better",
            "category": "precision",
        },
        "total_net_damage": {
            "description": "Total effective damage that reduced target HP (excludes overkill)",
            "formula": "sum(min(damage, remaining_hp) per shot)",
            "source": "metrics.total_net_damage",
            "direction": "higher_is_better",
            "category": "precision",
        },
        "total_gross_damage": {
            "description": "Total raw damage dealt including overkill",
            "formula": "total_net_damage + total_overkill",
            "source": "metrics.total_gross_damage",
            "direction": "context_dependent",
            "category": "precision",
        },
        "shots_per_target": {
            "description": "Average number of shots required per neutralized target",
            "formula": "total_ammo_used / targets_neutralized",
            "source": "metrics.shots_per_target (computed in EpisodeMetrics.__post_init__)",
            "direction": "lower_is_better",
            "category": "efficiency",
        },
        "total_reward": {
            "description": "Sum of all agent rewards across the episode",
            "formula": "sum(agent_rewards.values())",
            "source": "derived from metrics.agent_rewards or summary.total_reward",
            "direction": "higher_is_better",
            "category": "reward",
        },
        "mean_reward_per_agent": {
            "description": "Average reward earned per agent",
            "formula": "total_reward / num_agents",
            "source": "derived from total_reward",
            "direction": "higher_is_better",
            "category": "reward",
        },
        "success": {
            "description": "Boolean indicating whether all targets were neutralized",
            "formula": "termination_reason == 'all_targets_neutralized'",
            "source": "summary.success",
            "direction": "higher_is_better",
            "category": "task_completion",
        },
        "total_latent_mismatch": {
            "description": "Cumulative damage shortfall due to suboptimal drone-target pairing. For each shot at an active target, this is the difference between the best possible damage (from the optimally matched drone based on latent compatibility) and the actual damage dealt. Measures unrealized damage potential from assignment decisions.",
            "formula": "sum(max(0, max_damage_any_drone_for_target - actual_damage) per shot at active targets)",
            "source": "metrics.total_latent_mismatch (accumulated from env step latent_mismatch)",
            "direction": "lower_is_better",
            "category": "precision",
        },
        "latent_mismatch_ratio": {
            "description": "Fraction of optimal damage potential lost to suboptimal drone-target pairing. Normalizes latent mismatch as a proportion of total achievable damage. A value of 0 means every shot was fired by the best-matched drone; values approaching 1 indicate severe mismatching.",
            "formula": "total_latent_mismatch / (total_gross_damage + total_latent_mismatch)",
            "source": "metrics.latent_mismatch_ratio (computed in EpisodeMetrics.__post_init__)",
            "direction": "lower_is_better",
            "category": "precision",
        },
    }


def build_report(artifacts: RunArtifacts) -> dict[str, Any]:
    """Assemble the canonical policy report JSON structure."""
    episode_curves = build_episode_curves(artifacts)
    report = {
        "report_version": REPORT_VERSION,
        "builder_version": BUILDER_VERSION,
        "scenario_id": artifacts.environment.get("scenario_id", artifacts.paths.run_dir.name),
        "report_type": "policy_evaluation",
        "primary_policy": PRIMARY_POLICY,
        "report_focus": {
            "primary_anchor": "best_episode_path",
            "baselines": {
                name: "control"
                for name in sorted(artifacts.baselines)
                if name == "random"
            },
            "reference_policies": {
                name: "ceiling"
                for name in sorted(artifacts.baselines)
                if name != "random"
            },
        },
        "config_snapshot": build_config_snapshot(artifacts),
        "artifacts": build_artifacts_section(artifacts),
        "validation": {
            "status": artifacts.validation.status(),
            "warnings": artifacts.validation.warnings,
            "errors": artifacts.validation.errors,
            "unavailable_optional_fields": artifacts.validation.unavailable_optional_fields,
        },
        "episode_curves": episode_curves,
        "policy_summary": build_policy_summary(artifacts, episode_curves),
        "learning_summary": build_learning_summary(artifacts),
        "comparison_vs_baseline": build_comparison_vs_baseline(artifacts),
    }
    report["metric_definitions"] = build_metric_definitions()
    report["key_findings"] = build_key_findings(
        report["comparison_vs_baseline"],
        report["episode_curves"],
        report.get("learning_summary"),
    )
    return report


def write_report(report: dict[str, Any], output_path: Path) -> None:
    """Write the report JSON with stable formatting."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve artifacts, build the report, and write policy_report.json."""
    args = parse_args(argv)

    try:
        paths = resolve_builder_paths(Path(__file__), args.run)
        artifacts = collect_run_artifacts(paths)
    except ReportBuilderError as exc:
        print(f"ERROR: {exc}")
        return 1

    output_path = Path(args.output).resolve() if args.output else artifacts.paths.run_dir / "policy_report.json"
    report = build_report(artifacts)
    write_report(report, output_path)

    payload = {
        "status": "written",
        "run_dir": str(artifacts.paths.run_dir),
        "environment_path": str(artifacts.paths.environment_path),
        "primary_policy_dir": str(artifacts.paths.primary_policy_dir),
        "primary_summary_path": str(artifacts.paths.primary_summary_path),
        "primary_episode_count": len(artifacts.primary_episodes),
        "best_episode_path": str(artifacts.best_episode.path),
        "learning_state_count": len(artifacts.learning_states),
        "baselines": {
            name: str(artifact.episode.path)
            for name, artifact in artifacts.baselines.items()
        },
        "output_path": str(output_path),
        "validation": report["validation"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
