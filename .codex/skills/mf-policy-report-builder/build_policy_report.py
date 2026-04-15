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
    "avg_latent_match_quality": "precision",
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


REPORT_VERSION = "2.0"
BUILDER_VERSION = "0.7.0"


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
            "Optional output path. In single-file mode, path to the JSON file. "
            "In split mode (default), path to the report/ directory."
        ),
    )
    parser.add_argument(
        "--single-file",
        action="store_true",
        dest="single_file",
        help="Write a single monolithic policy_report.json instead of split files.",
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
    
    # Support new policies/ structure (preferred) and old flat structure (fallback)
    new_structure_dir = run_dir / "policies" / PRIMARY_POLICY
    old_structure_dir = run_dir / PRIMARY_POLICY
    
    if new_structure_dir.exists():
        primary_policy_dir = new_structure_dir
    elif old_structure_dir.exists():
        primary_policy_dir = old_structure_dir
    else:
        raise ReportBuilderError(
            f"Required matrix_factorization_cf artifacts are missing: "
            f"checked {new_structure_dir} and {old_structure_dir}"
        )
    
    primary_summary_path = primary_policy_dir / "episodes_summary.json"

    missing_paths = [
        path
        for path in (environment_path, primary_summary_path)
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
    
    # Determine policy search directory (new policies/ structure or old flat structure)
    policies_dir = paths.run_dir / "policies"
    search_dir = policies_dir if policies_dir.exists() else paths.run_dir
    
    for child in sorted(search_dir.iterdir(), key=lambda path: path.name):
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


def std_or_none(values: list[float]) -> float | None:
    """Return the population standard deviation for non-empty numeric lists."""
    if len(values) < 2:
        return None
    mu = sum(values) / len(values)
    return (sum((x - mu) ** 2 for x in values) / len(values)) ** 0.5


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

    targets_neutralized = metrics.get("targets_neutralized")
    total_ammo_used = metrics.get("total_ammo_used")
    total_net_damage = metrics.get("total_net_damage")
    total_gross_damage = metrics.get("total_gross_damage")

    entry = {
        "episode": metrics.get("episode", data.get("episode_num")),
        "steps": metrics.get("steps", summary.get("total_steps", len(data.get("steps", [])))),
        "success": summary.get("success"),
        "targets_neutralized": targets_neutralized,
        "total_ammo_used": total_ammo_used,
        "total_collisions": metrics.get("total_collisions"),
        "total_overkill": metrics.get("total_overkill"),
        "total_gross_damage": total_gross_damage,
        "total_net_damage": total_net_damage,
        "shots_per_target": metrics.get("shots_per_target"),
        "total_latent_mismatch": metrics.get("total_latent_mismatch"),
        "avg_latent_match_quality": metrics.get("avg_latent_match_quality"),
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
    "steps", "total_collisions", "total_overkill", "mean_reward_per_agent",
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
        "metric_schema": EPISODE_METRIC_SCHEMA,
        "metric_provenance": {
            "performance_metrics": "summary.metrics",
            "total_reward": "summary.total_reward",
            "mean_reward_per_agent": "derived_from_summary.total_reward",
            "end_of_episode_active_target_count": "derived_from_trace.final_step.info.target_active",
        },
        "trend_summary": trend_summary,
        "episodes": episodes,
    }


def _extract_float_series(episodes: list[dict[str, Any]], key: str) -> list[float]:
    """Extract a numeric series from episode entries."""
    return [float(item[key]) for item in episodes if isinstance(item.get(key), (int, float))]


def build_policy_summary(artifacts: RunArtifacts, episode_curves: dict[str, Any]) -> dict[str, Any]:
    """Build the canonical MF policy summary anchored on the best episode."""
    episodes = episode_curves["episodes"]
    best_episode_entry = extract_episode_entry(
        artifacts.best_episode,
        artifacts.validation,
    )

    steps_vals = _extract_float_series(episodes, "steps")
    ammo_vals = _extract_float_series(episodes, "total_ammo_used")
    collisions_vals = _extract_float_series(episodes, "total_collisions")
    overkill_vals = _extract_float_series(episodes, "total_overkill")
    reward_vals = _extract_float_series(episodes, "total_reward")
    mean_reward_vals = _extract_float_series(episodes, "mean_reward_per_agent")
    success_vals = [100.0 if item.get("success") else 0.0 for item in episodes if item.get("success") is not None]

    return {
        "best_episode": best_episode_entry,
        "training_episode_count": len(episodes),
        "total_training_steps": artifacts.primary_summary.get("total_steps"),
        "total_steps_to_best": artifacts.primary_summary.get("total_steps_to_best"),
        "aggregate_training": {
            "avg_steps": mean_or_none(steps_vals),
            "std_steps": std_or_none(steps_vals),
            "avg_total_ammo_used": mean_or_none(ammo_vals),
            "std_total_ammo_used": std_or_none(ammo_vals),
            "avg_total_collisions": mean_or_none(collisions_vals),
            "std_total_collisions": std_or_none(collisions_vals),
            "avg_total_overkill": mean_or_none(overkill_vals),
            "std_total_overkill": std_or_none(overkill_vals),
            "avg_total_reward": mean_or_none(reward_vals),
            "std_total_reward": std_or_none(reward_vals),
            "avg_mean_reward_per_agent": mean_or_none(mean_reward_vals),
            "std_mean_reward_per_agent": std_or_none(mean_reward_vals),
            "success_rate_pct": mean_or_none(success_vals),
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


METRIC_DEFINITIONS = {
    "steps": {
        "display_name": "Steps",
        "description": "Number of timesteps from episode start to termination",
        "formula": "count(timesteps)",
        "direction": "lower_is_better",
        "unit": "steps",
    },
    "targets_neutralized": {
        "display_name": "Targets Neutralized",
        "description": "Number of targets reduced to zero HP",
        "formula": "count(targets where hp <= 0)",
        "direction": "higher_is_better",
        "unit": "targets",
    },
    "total_ammo_used": {
        "display_name": "Total Ammo Used",
        "description": "Total shots fired by all agents across the episode",
        "formula": "sum(ammo_used per agent)",
        "direction": "lower_is_better",
        "unit": "shots",
    },
    "total_collisions": {
        "display_name": "Total Collisions",
        "description": "Number of times multiple agents targeted the same target simultaneously",
        "formula": "sum(collision events)",
        "direction": "lower_is_better",
        "unit": "events",
    },
    "total_overkill": {
        "display_name": "Total Overkill",
        "description": "Cumulative wasted damage dealt to targets already at zero HP",
        "formula": "sum(max(0, damage - remaining_hp) per shot)",
        "direction": "lower_is_better",
        "unit": "HP",
    },
    "total_net_damage": {
        "display_name": "Total Net Damage",
        "description": "Total effective damage that reduced target HP (excludes overkill)",
        "formula": "sum(min(damage, remaining_hp) per shot)",
        "direction": "higher_is_better",
        "unit": "HP",
    },
    "total_gross_damage": {
        "display_name": "Total Gross Damage",
        "description": "Total raw damage dealt including overkill",
        "formula": "total_net_damage + total_overkill",
        "direction": "context_dependent",
        "unit": "HP",
    },
    "shots_per_target": {
        "display_name": "Shots per Target",
        "description": "Average number of shots required per neutralized target",
        "formula": "total_ammo_used / targets_neutralized",
        "direction": "lower_is_better",
        "unit": "shots/target",
    },
    "total_latent_mismatch": {
        "display_name": "Total Latent Mismatch",
        "description": "Cumulative damage shortfall due to suboptimal drone-target pairing. For each shot at an active target, this is the difference between the best possible damage (from the optimally matched drone based on latent compatibility) and the actual damage dealt. Measures unrealized damage potential from assignment decisions.",
        "formula": "sum(max(0, max_damage_any_drone_for_target - actual_damage) per shot at active targets)",
        "direction": "lower_is_better",
        "unit": "HP",
    },
    "avg_latent_match_quality": {
        "display_name": "Avg. Match Quality",
        "description": "Average match quality per shot: fraction of optimal damage achieved through drone-target pairing decisions. Measures how well each shot utilized the best available drone for each target. A value of 1.0 means every shot was fired by the optimally matched drone; lower values indicate suboptimal pairing.",
        "formula": "total_gross_damage / total_optimal_potential, where total_optimal_potential = sum of optimal damages for all shots fired",
        "direction": "higher_is_better",
        "unit": "ratio",
    },
    "success": {
        "display_name": "Success",
        "description": "Boolean indicating whether all targets were neutralized",
        "formula": "termination_reason == 'all_targets_neutralized'",
        "direction": "higher_is_better",
        "unit": "bool",
    },
}

EPISODE_METRIC_SCHEMA: dict[str, dict[str, str]] = {
    **{
        name: {
            "display_name": defn["display_name"],
            "unit": defn["unit"],
            "direction": defn["direction"],
        }
        for name, defn in METRIC_DEFINITIONS.items()
    },
    "total_reward": {
        "display_name": "Total Reward",
        "unit": "reward",
        "direction": "higher_is_better",
    },
    "mean_reward_per_agent": {
        "display_name": "Mean Reward per Agent",
        "unit": "reward",
        "direction": "higher_is_better",
    },
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
    """Compare the MF best episode against same-run baselines, organized by metric."""
    if not artifacts.baselines:
        return {}

    mf_entry = extract_episode_entry(artifacts.best_episode, artifacts.validation)
    
    baseline_entries: dict[str, dict[str, Any]] = {}
    for name, baseline in artifacts.baselines.items():
        baseline_entries[name] = extract_episode_entry(
            baseline.episode,
            artifacts.validation,
        )

    metrics_to_compare = [
        ("steps", False),
        ("total_ammo_used", False),
        ("shots_per_target", False),
        ("total_collisions", False),
        ("total_overkill", False),
        ("total_net_damage", True),
        ("total_gross_damage", False),
        ("total_latent_mismatch", False),
        ("avg_latent_match_quality", True),
        ("targets_neutralized", True),
        ("success", True),
    ]

    comparison: dict[str, Any] = {}
    for metric_name, higher_is_better in metrics_to_compare:
        mf_value = mf_entry.get(metric_name)
        metric_def = METRIC_DEFINITIONS.get(metric_name, {})
        
        metric_obj: dict[str, Any] = {
            "display_name": metric_def.get("display_name", metric_name),
            "description": metric_def.get("description", ""),
            "formula": metric_def.get("formula", ""),
            "category": METRIC_CATEGORIES.get(metric_name, "other"),
            "direction": metric_def.get("direction", "higher_is_better" if higher_is_better else "lower_is_better"),
            "unit": metric_def.get("unit", ""),
            "mf_value": mf_value,
        }
        
        for baseline_name, baseline_entry in baseline_entries.items():
            baseline_value = baseline_entry.get(metric_name)
            metric_obj[baseline_name] = baseline_value
            
            relative_change = compute_relative_change(mf_value, baseline_value)
            if relative_change is not None:
                metric_obj[f"mf_vs_{baseline_name}_pct"] = relative_change
        
        comparison[metric_name] = metric_obj

    return comparison


def build_key_findings(
    comparison: dict[str, Any],
    episode_curves: dict[str, Any],
    learning_summary: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Surface the most actionable findings from the assembled report sections."""
    findings: list[dict[str, Any]] = []

    baseline_names = []
    for metric_name, metric_data in comparison.items():
        if isinstance(metric_data, dict):
            for key in metric_data.keys():
                if key not in ("description", "formula", "category", "direction", "mf_value") and not key.startswith("mf_vs_"):
                    if key not in baseline_names:
                        baseline_names.append(key)

    if len(baseline_names) >= 2:
        for metric_name, metric_data in comparison.items():
            if not isinstance(metric_data, dict):
                continue
            
            mf_value = metric_data.get("mf_value")
            direction = metric_data.get("direction", "")
            if direction == "context_dependent":
                continue
            higher_is_better = "higher" in direction

            worsened_vs = []
            for baseline_name in baseline_names:
                baseline_value = metric_data.get(baseline_name)
                if isinstance(mf_value, (int, float)) and isinstance(baseline_value, (int, float)):
                    if mf_value != baseline_value:
                        is_worse = (mf_value < baseline_value) if higher_is_better else (mf_value > baseline_value)
                        if is_worse:
                            worsened_vs.append(baseline_name)
            
            if len(worsened_vs) == len(baseline_names):
                findings.append({
                    "type": "cross_baseline_regression",
                    "metric": metric_name,
                    "category": metric_data.get("category", "other"),
                    "worsened_vs": worsened_vs,
                })

    trend_summary = episode_curves.get("trend_summary", {})
    for metric_name, trend in trend_summary.items():
        higher_is_better = metric_name in ("mean_reward_per_agent",)
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




REPORT_FILES = {
    "config": {
        "filename": "config.json",
        "description": "Experiment configuration (paper: Methods section)",
    },
    "comparison": {
        "filename": "comparison.json",
        "description": "MF vs baseline results and key findings (paper: Results table)",
    },
    "episode_curves": {
        "filename": "episode_curves.json",
        "description": "Per-episode time-series metrics (viz: learning curve plots)",
    },
    "learning_progression": {
        "filename": "learning_progression.json",
        "description": "Epsilon decay, checkpoints, convergence (viz: learning state plots)",
    },
    "policy_summary": {
        "filename": "policy_summary.json",
        "description": "Aggregate stats and best episode (paper: Results summary)",
    },
}


def build_report_files(artifacts: RunArtifacts) -> dict[str, dict[str, Any]]:
    """Assemble the split report as a dict of filename -> content."""
    episode_curves = build_episode_curves(artifacts)
    policy_summary = build_policy_summary(artifacts, episode_curves)
    learning_summary = build_learning_summary(artifacts)
    comparison = build_comparison_vs_baseline(artifacts)
    key_findings = build_key_findings(comparison, episode_curves, learning_summary)

    validation = {
        "status": artifacts.validation.status(),
        "warnings": artifacts.validation.warnings,
        "errors": artifacts.validation.errors,
        "unavailable_optional_fields": artifacts.validation.unavailable_optional_fields,
    }

    manifest = {
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
        "validation": validation,
        "artifacts": build_artifacts_section(artifacts),
        "files": {
            key: {"path": info["filename"], "description": info["description"]}
            for key, info in REPORT_FILES.items()
        },
    }

    return {
        "report_manifest.json": manifest,
        "config.json": build_config_snapshot(artifacts),
        "comparison.json": {
            "comparison_vs_baseline": comparison,
            "key_findings": key_findings,
        },
        "episode_curves.json": episode_curves,
        "learning_progression.json": learning_summary if learning_summary else {"available": False},
        "policy_summary.json": policy_summary,
    }


def build_report_single(artifacts: RunArtifacts) -> dict[str, Any]:
    """Assemble the monolithic policy report JSON (legacy single-file mode)."""
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
    report["key_findings"] = build_key_findings(
        report["comparison_vs_baseline"],
        report["episode_curves"],
        report.get("learning_summary"),
    )
    return report


def _write_json(data: dict[str, Any], path: Path) -> None:
    """Write a single JSON file with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def write_report_files(files: dict[str, dict[str, Any]], report_dir: Path) -> list[str]:
    """Write split report files into report_dir. Returns list of written paths."""
    written = []
    for filename, content in files.items():
        path = report_dir / filename
        _write_json(content, path)
        written.append(str(path))
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve artifacts, build the report, and write output."""
    args = parse_args(argv)

    try:
        paths = resolve_builder_paths(Path(__file__), args.run)
        artifacts = collect_run_artifacts(paths)
    except ReportBuilderError as exc:
        print(f"ERROR: {exc}")
        return 1

    if args.single_file:
        output_path = Path(args.output).resolve() if args.output else artifacts.paths.run_dir / "policy_report.json"
        report = build_report_single(artifacts)
        _write_json(report, output_path)
        validation = report["validation"]
        written_paths = [str(output_path)]
    else:
        report_dir = Path(args.output).resolve() if args.output else artifacts.paths.run_dir / "report"
        files = build_report_files(artifacts)
        written_paths = write_report_files(files, report_dir)
        validation = files["report_manifest.json"]["validation"]

    payload = {
        "status": "written",
        "mode": "single_file" if args.single_file else "split",
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
        "output_paths": written_paths,
        "validation": validation,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
