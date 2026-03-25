from pathlib import Path

import pytest

from tabula_drone.logging import EnvironmentLogger, EpisodeLogger


def test_environment_logger_start_policy_creates_active_episode_logger(tmp_path: Path) -> None:
    env_logger = EnvironmentLogger(
        output_dir=str(tmp_path),
        scenario_id="scenario_alpha",
        mode="continuous",
    )

    episode_logger = env_logger.start_policy("random_policy", is_deterministic=False)

    expected_policy_dir = tmp_path / "scenario_alpha" / "random_policy"

    assert isinstance(episode_logger, EpisodeLogger)
    assert episode_logger is env_logger.active_episode_logger
    assert Path(episode_logger.output_dir) == expected_policy_dir / "episodes"
    assert Path(episode_logger.analysis_dir) == expected_policy_dir / "analysis"
    assert episode_logger.mode == "continuous"
    assert (expected_policy_dir / "episodes").is_dir()
    assert (expected_policy_dir / "analysis").is_dir()
    assert (expected_policy_dir / "learning_state").is_dir()


def test_environment_logger_save_policy_episodes_persists_selected_episode(tmp_path: Path) -> None:
    env_logger = EnvironmentLogger(
        output_dir=str(tmp_path),
        scenario_id="scenario_beta",
    )
    env_logger.start_policy("oracle_policy", is_deterministic=True)

    env_logger.record_episode_data({"episode": "payload"}, steps=4)

    result = env_logger.save_policy_episodes()

    episode_path = tmp_path / "scenario_beta" / "oracle_policy" / "episodes" / "episode_ep01.json"

    assert result["steps"] == {"first": 4}
    assert result["files"] == [".../episode_ep01.json"]
    assert episode_path.is_file()
    assert episode_path.read_text() == '{\n  "episode": "payload"\n}'

    with pytest.raises(ValueError, match="start_policy\\(\\) must be called"):
        _ = env_logger.active_episode_logger


def test_environment_logger_persist_episode_outputs_uses_active_logger(tmp_path: Path) -> None:
    env_logger = EnvironmentLogger(
        output_dir=str(tmp_path),
        scenario_id="scenario_gamma",
    )
    env_logger.start_policy("policy_x", is_deterministic=True)

    class StubEpisodeLogger:
        def get_analysis_data(self) -> dict:
            return {"summary": {"count": 1}}

        def to_dict(self) -> dict:
            return {"episode": "artifact"}

    env_logger._active_episode_logger = StubEpisodeLogger()

    env_logger.record_episode(steps=7)
    env_logger.save_analysis(episode_num=1)
    result = env_logger.save_policy_episodes()

    analysis_path = tmp_path / "scenario_gamma" / "policy_x" / "analysis" / "analysis_ep01.json"
    episode_path = tmp_path / "scenario_gamma" / "policy_x" / "episodes" / "episode_ep01.json"

    assert analysis_path.is_file()
    assert analysis_path.read_text() == '{\n  "summary": {\n    "count": 1\n  }\n}'
    assert episode_path.is_file()
    assert episode_path.read_text() == '{\n  "episode": "artifact"\n}'
    assert result["steps"] == {"first": 7}


def test_environment_logger_save_policy_episodes_saves_all_matrix_factorization_episodes_and_summary(
    tmp_path: Path,
) -> None:
    env_logger = EnvironmentLogger(
        output_dir=str(tmp_path),
        scenario_id="scenario_mf",
    )
    env_logger.start_policy("matrix_factorization_cf", is_deterministic=False)

    env_logger.record_episode_data({"episode": 1}, steps=8)
    env_logger.record_episode_data({"episode": 2}, steps=5)
    env_logger.record_episode_data({"episode": 3}, steps=7)

    result = env_logger.save_policy_episodes()

    policy_dir = tmp_path / "scenario_mf" / "matrix_factorization_cf"
    episodes_dir = policy_dir / "episodes"
    summary_path = policy_dir / "episodes_summary.json"

    assert result["files"] == [
        ".../episode_ep01.json",
        ".../episode_ep02.json",
        ".../episode_ep03.json",
    ]
    assert result["steps"] == {"first": 8, "final": 7}
    assert result["best_episode_num"] == 2
    assert result["milestones"] == {"first": 1, "best": 2}

    assert (episodes_dir / "episode_ep01.json").read_text() == '{\n  "episode": 1\n}'
    assert (episodes_dir / "episode_ep02.json").read_text() == '{\n  "episode": 2\n}'
    assert (episodes_dir / "episode_ep03.json").read_text() == '{\n  "episode": 3\n}'
    assert summary_path.read_text() == (
        '{\n'
        '  "total_episodes": 3,\n'
        '  "total_steps": 20,\n'
        '  "total_steps_to_best": 13,\n'
        '  "best_episode_path": "episodes/episode_ep02.json"\n'
        '}'
    )


def test_environment_logger_start_episode_persists_shared_environment_artifact(tmp_path: Path) -> None:
    env_logger = EnvironmentLogger(
        output_dir=str(tmp_path),
        scenario_id="scenario_theta",
    )
    env_logger.start_policy("policy_env", is_deterministic=True)

    class StubEpisodeLogger:
        VERSION = "1.2"

        def _build_shared_config_snapshot(self, env) -> dict:
            return {
                "world_size": [10.0, 20.0],
                "max_steps": 5,
                "scenario_id": env.scenario_id,
                "class_attribute_mapping": {"A": {"hp": 1.0}},
                "weapon_damage_profile_mapping": {"light": {"hp": 1.0}},
            }

        def _build_scenario_snapshot(self, env, reset_info, seed) -> dict:
            return {
                "num_drones": 1,
                "num_targets": 1,
                "drone_positions": [[1.0, 2.0]],
                "target_positions": [[3.0, 4.0]],
                "weapon_assignments": {"drone_0": "light"},
                "target_classes": ["A"],
            }

        def start_episode(self, **kwargs) -> None:
            return None

    class StubEnv:
        scenario_id = "scenario_theta"

    env_logger._active_episode_logger = StubEpisodeLogger()

    env_logger.start_episode(
        env=StubEnv(),
        reset_info={"step_index": 0},
        seed=11,
        episode_num=1,
        total_episodes=1,
    )

    environment_path = tmp_path / "scenario_theta" / "environment.json"
    assert environment_path.is_file()
    assert environment_path.read_text() == (
        '{\n'
        '  "version": "1.2",\n'
        '  "scenario_id": "scenario_theta",\n'
        '  "config": {\n'
        '    "world_size": [\n'
        '      10.0,\n'
        '      20.0\n'
        '    ],\n'
        '    "max_steps": 5,\n'
        '    "scenario_id": "scenario_theta",\n'
        '    "class_attribute_mapping": {\n'
        '      "A": {\n'
        '        "hp": 1.0\n'
        '      }\n'
        '    },\n'
        '    "weapon_damage_profile_mapping": {\n'
        '      "light": {\n'
        '        "hp": 1.0\n'
        '      }\n'
        '    }\n'
        '  },\n'
        '  "scenario": {\n'
        '    "num_drones": 1,\n'
        '    "num_targets": 1,\n'
        '    "drone_positions": [\n'
        '      [\n'
        '        1.0,\n'
        '        2.0\n'
        '      ]\n'
        '    ],\n'
        '    "target_positions": [\n'
        '      [\n'
        '        3.0,\n'
        '        4.0\n'
        '      ]\n'
        '    ],\n'
        '    "weapon_assignments": {\n'
        '      "drone_0": "light"\n'
        '    },\n'
        '    "target_classes": [\n'
        '      "A"\n'
        '    ]\n'
        '  }\n'
        '}'
    )


def test_environment_logger_handle_flush_persists_learning_state_checkpoint(tmp_path: Path) -> None:
    env_logger = EnvironmentLogger(
        output_dir=str(tmp_path),
        scenario_id="scenario_delta",
        mode="continuous",
    )
    env_logger.start_policy("policy_y", is_deterministic=False)

    env_logger.configure_continuous_flush(
        episode_num=3,
        learning_state_provider=lambda: {
            "target_classes": ["A", "B", "B", "C"],
            "agents": [{"agent_lv": [1.0, 2.0]}],
        },
        num_agents=2,
        num_targets=4,
        latent_dim=2,
    )

    env_logger.handle_flush(10)

    learning_state_path = (
        tmp_path
        / "scenario_delta"
        / "policy_y"
        / "learning_state"
        / "learning_state_ep03_step_00010.json"
    )

    assert learning_state_path.is_file()
    assert learning_state_path.read_text() == (
        '{\n'
        '  "version": "1.0",\n'
        '  "scenario_id": "scenario_delta",\n'
        '  "episode_num": 3,\n'
        '  "policy_type": "policy_y",\n'
        '  "num_agents": 2,\n'
        '  "num_targets": 4,\n'
        '  "latent_dim": 2,\n'
        '  "episode_state": {\n'
        '    "target_classes": [\n'
        '      "A",\n'
        '      "B",\n'
        '      "B",\n'
        '      "C"\n'
        '    ],\n'
        '    "agents": [\n'
        '      {\n'
        '        "agent_lv": [\n'
        '          1.0,\n'
        '          2.0\n'
        '        ]\n'
        '      }\n'
        '    ]\n'
        '  }\n'
        '}'
    )


def test_environment_logger_episode_lifecycle_methods_delegate_to_active_logger(tmp_path: Path) -> None:
    env_logger = EnvironmentLogger(
        output_dir=str(tmp_path),
        scenario_id="scenario_epsilon",
    )
    env_logger.start_policy("policy_z", is_deterministic=True)

    calls = []

    class StubEpisodeLogger:
        def _build_shared_config_snapshot(self, env) -> dict:
            return {"scenario_id": "scenario_epsilon"}

        def _build_scenario_snapshot(self, env, reset_info, seed) -> dict:
            return {"num_drones": 0, "num_targets": 0}

        def start_episode(self, **kwargs) -> None:
            calls.append(("start", kwargs))

        def log_step(self, **kwargs) -> None:
            calls.append(("step", kwargs))

        def flush(self, step_num: int) -> str:
            calls.append(("flush", step_num))
            return f"flush-{step_num}"

        def end_episode(self, **kwargs) -> None:
            calls.append(("end", kwargs))

    env_logger._active_episode_logger = StubEpisodeLogger()

    env_logger.start_episode(env="env", reset_info={"step_index": 0}, seed=7, episode_num=2, total_episodes=9)
    env_logger.log_step(
        step_num=3,
        actions={"drone_0": 1},
        rewards={"drone_0": 2.5},
        terminated=False,
        truncated=False,
        info={"target_hps": [1.0]},
    )
    flush_result = env_logger.flush_episode(3)
    env_logger.end_episode(total_rewards={"drone_0": 4.0}, done_reason="done")

    assert flush_result == "flush-3"
    assert calls == [
        (
            "start",
            {
                "env": "env",
                "reset_info": {"step_index": 0},
                "seed": 7,
                "episode_num": 2,
                "total_episodes": 9,
                "environment_path": "../../environment.json",
            },
        ),
        (
            "step",
            {
                "step_num": 3,
                "actions": {"drone_0": 1},
                "rewards": {"drone_0": 2.5},
                "terminated": False,
                "truncated": False,
                "info": {"target_hps": [1.0]},
            },
        ),
        ("flush", 3),
        (
            "end",
            {
                "total_rewards": {"drone_0": 4.0},
                "done_reason": "done",
            },
        ),
    ]
