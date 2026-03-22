"""
Utilities for reading and transforming engagement analysis artifacts.
"""

import json
import re
from typing import Any, Dict, List, Tuple


def load_analysis(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def extract_drone_engagement_counts(
    analysis_data: Dict[str, Any],
) -> Dict[str, Dict[str, int]]:
    summary = analysis_data.get("summary", {})
    counts = summary.get("drone_engagement_counts", {})
    return {
        str(drone_id): {str(target_id): int(v) for target_id, v in per_target.items()}
        for drone_id, per_target in counts.items()
    }


def _id_sort_key(value: str) -> Tuple[str, int, str]:
    match = re.search(r"(\d+)$", value)
    if match is None:
        return (value, -1, value)
    return (value[: match.start(1)], int(match.group(1)), value)


def build_target_x_drone_table(
    drone_engagement_counts: Dict[str, Dict[str, int]],
) -> Tuple[List[str], List[List[Any]]]:
    drone_ids = sorted(drone_engagement_counts.keys(), key=_id_sort_key)

    target_id_set = set()
    for per_target in drone_engagement_counts.values():
        target_id_set.update(per_target.keys())
    target_ids = sorted(target_id_set, key=_id_sort_key)

    headers: List[str] = ["Target"] + drone_ids
    rows: List[List[Any]] = []

    for target_id in target_ids:
        row: List[Any] = [target_id]
        for drone_id in drone_ids:
            row.append(drone_engagement_counts.get(drone_id, {}).get(target_id, 0))
        rows.append(row)

    return headers, rows
