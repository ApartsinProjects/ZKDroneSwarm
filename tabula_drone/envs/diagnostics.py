"""
Typed diagnostics snapshots for environment-level telemetry.

These snapshots are internal project models. They can be serialized to plain
dicts for logging and other boundary-facing consumers, but they are not part of
the PettingZoo `info` contract themselves.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import copy


@dataclass
class EnvDiagnosticsSnapshot:
    """
    Snapshot of shared environment diagnostics for a reset or step boundary.

    Optional fields capture step-specific telemetry that may be absent during
    reset or in early migration stages.
    """

    step_index: int
    scenario_id: str
    actions: Dict[str, int]
    ammo_used: Dict[str, int]
    weapon_types: List[str]
    target_hps: List[float]
    target_attributes: List[Dict[str, float]]
    target_classes: List[str]
    target_active: List[bool]
    processing_order: Optional[List[str]] = None
    net_damage: Optional[float] = None
    neutralizations_this_step: Optional[int] = None
    cumulative_neutralizations: Optional[int] = None
    collisions: Optional[int] = None
    target_selections: Optional[Dict[int, List[str]]] = None
    overkill: Optional[Dict[int, float]] = None
    done_reason: Optional[str] = None
    total_gross_damage: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this snapshot into a plain dict for boundary consumers."""
        payload: Dict[str, Any] = {
            "step_index": self.step_index,
            "scenario_id": self.scenario_id,
            "actions": copy.deepcopy(self.actions),
            "ammo_used": copy.deepcopy(self.ammo_used),
            "weapon_types": list(self.weapon_types),
            "target_hps": list(self.target_hps),
            "target_attributes": copy.deepcopy(self.target_attributes),
            "target_classes": list(self.target_classes),
            "target_active": list(self.target_active),
        }

        optional_fields = {
            "processing_order": self.processing_order,
            "net_damage": self.net_damage,
            "neutralizations_this_step": self.neutralizations_this_step,
            "cumulative_neutralizations": self.cumulative_neutralizations,
            "collisions": self.collisions,
            "target_selections": self.target_selections,
            "overkill": self.overkill,
            "done_reason": self.done_reason,
            "total_gross_damage": self.total_gross_damage,
        }

        for key, value in optional_fields.items():
            if value is not None:
                payload[key] = copy.deepcopy(value)

        return payload
