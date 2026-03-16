"""
Centralized Metrics Calculation for ZK-MRTA Swarm Performance.

Translates raw environment counters into human-readable KPIs for 
both Episodic and Continuous operational modes.
"""

from typing import Any, Dict, Optional, Union

def calculate_derived_metrics(
    raw_metrics: Dict[str, Any],
    mode: str = "episodic"
) -> Dict[str, Any]:
    """
    Calculate KPIs from raw metrics dictionary.
    
    Args:
        raw_metrics: Dictionary containing:
            - steps
            - targets_neutralized
            - total_ammo_used
            - total_effective_damage
            - total_potential_damage
            - total_collisions
        mode: 'episodic' or 'continuous'
        
    Returns:
        Dictionary of calculated KPIs and formatted strings.
    """
    m = raw_metrics
    steps = m.get("steps", 0)
    neutralized = m.get("targets_neutralized", 0)
    ammo = m.get("total_ammo_used", 0)
    eff_dmg = m.get("total_effective_damage", 0.0)
    pot_dmg = m.get("total_potential_damage", 0.0)
    collisions = m.get("total_collisions", 0)

    # Common metrics
    ammo_eff = neutralized / ammo if ammo > 0 else 0.0
    dmg_eff = eff_dmg / pot_dmg if pot_dmg > 0 else 0.0
    
    # Mode-specific metrics
    results = {
        "ammo_eff": ammo_eff,
        "dmg_eff": dmg_eff,
    }
    
    if mode == "continuous":
        # Throughput: Neutralizations per 100 steps
        results["throughput"] = (neutralized / steps * 100) if steps > 0 else 0.0
        
        # Coordination Score: Neutralizations per Collision
        if collisions == 0:
            results["coordination_score"] = float('inf')
            results["coordination_str"] = "N/A"
        else:
            score = neutralized / collisions
            results["coordination_score"] = score
            results["coordination_str"] = f"{score:.2f}"
    else:
        # Episodic metrics
        results["shots_per_target"] = ammo / neutralized if neutralized > 0 else 0.0
        results["best_steps"] = steps

    return results

def format_metric_display(val: Union[float, str], fmt: str = "{}") -> str:
    """Helper to format numeric values safely."""
    if isinstance(val, str):
        return val
    if val == float('inf'):
        return "N/A"
    return fmt.format(val)
