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


def test_environment_logger_finalize_policy_persists_selected_episode(tmp_path: Path) -> None:
    env_logger = EnvironmentLogger(
        output_dir=str(tmp_path),
        scenario_id="scenario_beta",
    )
    env_logger.start_policy("oracle_policy", is_deterministic=True)

    env_logger.record_episode({"episode": "payload"}, steps=4)

    result = env_logger.finalize_policy()

    episode_path = tmp_path / "scenario_beta" / "oracle_policy" / "episodes" / "episode_first_ep01.json"

    assert result["steps"] == {"first": 4}
    assert result["files"] == [".../episode_first_ep01.json"]
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

    env_logger.persist_episode_outputs(episode_num=1, steps=7)
    result = env_logger.finalize_policy()

    analysis_path = tmp_path / "scenario_gamma" / "policy_x" / "analysis" / "analysis_ep01.json"
    episode_path = tmp_path / "scenario_gamma" / "policy_x" / "episodes" / "episode_first_ep01.json"

    assert analysis_path.is_file()
    assert analysis_path.read_text() == '{\n  "summary": {\n    "count": 1\n  }\n}'
    assert episode_path.is_file()
    assert episode_path.read_text() == '{\n  "episode": "artifact"\n}'
    assert result["steps"] == {"first": 7}


def test_environment_logger_handle_flush_persists_learning_state_checkpoint(tmp_path: Path) -> None:
    env_logger = EnvironmentLogger(
        output_dir=str(tmp_path),
        scenario_id="scenario_delta",
        mode="continuous",
    )
    env_logger.start_policy("policy_y", is_deterministic=False)

    env_logger.configure_continuous_flush(
        episode_num=3,
        learning_state_provider=lambda: {"agents": [{"agent_lv": [1.0, 2.0]}]},
        num_agents=2,
        num_targets=4,
        latent_dim=2,
        entities={"agents": [{"agent_id": "drone_0"}]},
    )

    env_logger.handle_flush(10)

    learning_state_path = (
        tmp_path
        / "scenario_delta"
        / "policy_y"
        / "learning_state"
        / "learning_state_step_00010.json"
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
        '  "pre_episode": null,\n'
        '  "post_episode": {\n'
        '    "agents": [\n'
        '      {\n'
        '        "agent_lv": [\n'
        '          1.0,\n'
        '          2.0\n'
        '        ]\n'
        '      }\n'
        '    ]\n'
        '  },\n'
        '  "entities": {\n'
        '    "agents": [\n'
        '      {\n'
        '        "agent_id": "drone_0"\n'
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
