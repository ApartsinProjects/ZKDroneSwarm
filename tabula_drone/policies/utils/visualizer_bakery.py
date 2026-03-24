import json
import os
import numpy as np
from typing import Dict, Any, List, Tuple

try:
    from sklearn.manifold import TSNE
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

def compute_tsne_2d_offline(P: np.ndarray, U: np.ndarray, agent_idx: int) -> Tuple[List[float], List[List[float]]]:
    """
    Compute joint t-SNE 2D projection for a given state offline.
    
    Args:
        P: Agent matrix (num_agents, latent_dim)
        U: Target matrix (latent_dim, num_targets)
        agent_idx: Index of the current agent in P
        
    Returns:
        (agent_2d, targets_2d)
    """
    if not _HAS_SKLEARN:
        return P[agent_idx][:2].tolist(), U.T[:, :2].tolist()

    agent_row = P[agent_idx].reshape(1, -1)
    targets = U.T
    combined = np.vstack([agent_row, targets])

    n_samples = combined.shape[0]
    perplexity = min(5, max(1, n_samples - 1))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        init="pca",
        learning_rate="auto",
    )
    embeddings_2d = tsne.fit_transform(combined)

    agent_2d = embeddings_2d[0].tolist()
    targets_2d = embeddings_2d[1:].tolist()
    return agent_2d, targets_2d

def enrich_learning_state_file(filepath: str):
    """
    Read a learning state JSON, compute t-SNE 2D projections from P/U,
    and update the agent_lv/target_lv fields in place.
    """
    if not os.path.exists(filepath):
        print(f"Warning: Enrichment file not found: {filepath}")
        return

    with open(filepath, 'r') as f:
        data = json.load(f)

    stage_data = data.get("episode_state")
    if isinstance(stage_data, dict):
        agents_data = stage_data.get("agents", [])

        for agent_entry in agents_data:
            # We need P and U matrices
            if "P" not in agent_entry or "U" not in agent_entry:
                continue

            P = np.array(agent_entry["P"])
            U = np.array(agent_entry["U"])
            agent_idx = agent_entry.get("agent_idx", 0)

            # Compute real t-SNE
            a_2d, t_2d = compute_tsne_2d_offline(P, U, agent_idx)

            # Update fields
            agent_entry["agent_lv"] = a_2d
            agent_entry["target_lv"] = t_2d

    # Save back
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
