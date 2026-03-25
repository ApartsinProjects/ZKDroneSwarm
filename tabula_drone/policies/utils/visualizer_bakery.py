import json
import os
import re
import numpy as np
from typing import Dict, Any, List, Tuple

try:
    from sklearn.manifold import TSNE
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

TSNE_MODE_PER_EPISODE = "per_episode"
TSNE_MODE_PER_EPISODE_ALIGNED = "per_episode_aligned"

def compute_tsne_2d_offline(P: np.ndarray, U: np.ndarray, agent_idx: int) -> np.ndarray:
    """Compute a per-episode t-SNE block for one agent and its targets."""
    agent_row = P[agent_idx].reshape(1, -1)
    targets = U.T
    combined = np.vstack([agent_row, targets])

    if not _HAS_SKLEARN:
        return combined[:, :2]

    n_samples = combined.shape[0]
    perplexity = min(5, max(1, n_samples - 1))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        init="pca",
        learning_rate=50,
    )
    return tsne.fit_transform(combined)


def _align_points_to_reference(reference_points: np.ndarray, current_points: np.ndarray) -> np.ndarray:
    """Rotate and translate the current episode embedding toward the previous frame."""
    if reference_points.shape != current_points.shape:
        return current_points
    if current_points.shape[0] < 2:
        return current_points

    reference_center = reference_points.mean(axis=0)
    current_center = current_points.mean(axis=0)
    reference_centered = reference_points - reference_center
    current_centered = current_points - current_center

    reference_norm = np.linalg.norm(reference_centered)
    current_norm = np.linalg.norm(current_centered)
    if reference_norm < 1e-12 or current_norm < 1e-12:
        return current_centered + reference_center

    reference_unit = reference_centered / reference_norm
    current_unit = current_centered / current_norm

    u, _singular_values, vt = np.linalg.svd(current_unit.T @ reference_unit)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt

    return current_centered @ rotation + reference_center


def _learning_state_sort_key(filepath: str) -> Tuple[int, int, str]:
    """Sort learning-state files by episode or step progression for alignment."""
    filename = os.path.basename(filepath)

    episode_match = re.search(r"ep(\d+)", filename)
    if episode_match:
        return (0, int(episode_match.group(1)), filename)

    step_match = re.search(r"step_(\d+)", filename)
    if step_match:
        return (1, int(step_match.group(1)), filename)

    if "continuous_final" in filename:
        return (2, 0, filename)

    return (3, 0, filename)


def _iter_learning_state_files(directorypath: str) -> List[str]:
    """Return ordered learning-state files for a directory."""
    if not os.path.isdir(directorypath):
        return []

    return sorted(
        (
            os.path.join(directorypath, filename)
            for filename in os.listdir(directorypath)
            if filename.startswith("learning_state_") and filename.endswith(".json")
        ),
        key=_learning_state_sort_key,
    )


def _apply_embedding(agent_entry: Dict[str, Any], embedding: np.ndarray) -> None:
    """Write a 2D embedding back into an agent entry."""
    agent_entry["agent_lv"] = embedding[0].tolist()
    agent_entry["target_lv"] = embedding[1:].tolist()


def _enrich_learning_state_dir_per_episode(directorypath: str) -> None:
    """Compute independent per-episode t-SNE for each learning-state file."""
    state_files = _iter_learning_state_files(directorypath)
    if not state_files:
        return

    for filepath in state_files:
        with open(filepath, 'r') as f:
            data = json.load(f)

        stage_data = data.get("episode_state")
        if isinstance(stage_data, dict):
            agents_data = stage_data.get("agents", [])
            for agent_entry in agents_data:
                if "P" not in agent_entry or "U" not in agent_entry:
                    continue

                P = np.array(agent_entry["P"])
                U = np.array(agent_entry["U"])
                agent_idx = agent_entry.get("agent_idx", 0)
                embedding = compute_tsne_2d_offline(P, U, agent_idx)
                _apply_embedding(agent_entry, embedding)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


def _enrich_learning_state_dir_per_episode_aligned(directorypath: str) -> None:
    """Compute per-episode t-SNE and align each episode to the previous frame."""
    state_files = _iter_learning_state_files(directorypath)
    if not state_files:
        return

    previous_embeddings_by_agent: Dict[int, np.ndarray] = {}
    for filepath in state_files:
        with open(filepath, 'r') as f:
            data = json.load(f)

        stage_data = data.get("episode_state")
        if isinstance(stage_data, dict):
            agents_data = stage_data.get("agents", [])
            for agent_entry in agents_data:
                if "P" not in agent_entry or "U" not in agent_entry:
                    continue

                P = np.array(agent_entry["P"])
                U = np.array(agent_entry["U"])
                agent_idx = agent_entry.get("agent_idx", 0)

                embedding = compute_tsne_2d_offline(P, U, agent_idx)
                reference = previous_embeddings_by_agent.get(agent_idx)
                if reference is not None:
                    embedding = _align_points_to_reference(reference, embedding)

                _apply_embedding(agent_entry, embedding)
                previous_embeddings_by_agent[agent_idx] = embedding

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


def enrich_learning_state_dir(directorypath: str, mode: str = TSNE_MODE_PER_EPISODE_ALIGNED) -> None:
    """Run the selected offline t-SNE enrichment mode for a learning-state directory."""
    if mode == TSNE_MODE_PER_EPISODE:
        _enrich_learning_state_dir_per_episode(directorypath)
        return

    if mode == TSNE_MODE_PER_EPISODE_ALIGNED:
        _enrich_learning_state_dir_per_episode_aligned(directorypath)
        return

    raise ValueError(f"Unsupported t-SNE enrichment mode: {mode}")


def enrich_learning_state_file(filepath: str, mode: str = TSNE_MODE_PER_EPISODE_ALIGNED):
    """
    Enrich one file by delegating to the selected directory-level t-SNE pass.
    """
    if not os.path.exists(filepath):
        print(f"Warning: Enrichment file not found: {filepath}")
        return

    enrich_learning_state_dir(os.path.dirname(filepath), mode=mode)
