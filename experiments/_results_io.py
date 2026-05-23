"""Complete experimental-data persistence (so runs can be re-analyzed, not just
read from summaries). Every experiment should save its FULL per-seed/per-config
results via save_results(); then add a row to docs/DATA_CATALOGUE.md.
"""
import json
import os
import time
import subprocess

# Repo root (parent of experiments/). Anchor relative results_dir here so results
# always land in <root>/results/pilots regardless of the cwd an experiment is
# launched from (avoids the cwd-relative "results/pilots" under experiments/ bug).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def save_results(name, payload, results_dir="results/pilots"):
    """Save COMPLETE results to results/pilots/<name>_<UTCstamp>.json.

    payload should include EVERYTHING needed to re-analyze without re-running:
      - "meta": params, config grid, seeds, metric definitions, date, git commit
      - "raw":  per-seed / per-config metric values (lists, not just means)
      - optional "trajectories": per-round series for AUC / anytime curves
    Returns the saved path.
    """
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(_ROOT, results_dir)
    os.makedirs(results_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    payload = dict(payload)
    payload.setdefault("meta", {})
    payload["meta"]["saved_utc"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    payload["meta"]["git_commit"] = _git_commit()
    payload["meta"]["name"] = name
    path = os.path.join(results_dir, f"{name}_{stamp}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    return path
