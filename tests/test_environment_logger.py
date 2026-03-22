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
