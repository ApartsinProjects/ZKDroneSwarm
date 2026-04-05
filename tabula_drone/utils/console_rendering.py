"""Console printer for the demo orchestrator."""

import builtins
from typing import Any, Dict, List, TextIO, Tuple

from tabulate import tabulate


class ConsolePrinter:
    """Straightforward console output helper for demo/reporting text."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream

    def _emit(self, text: str, *, end: str = "\n") -> None:
        if self._stream is None:
            builtins.print(text, end=end)
            return
        builtins.print(text, end=end, file=self._stream)

    def separator(self, width: int = 60) -> None:
        self._emit("=" * width)

    def target_class_profile(
        self,
        class_attribute_mapping: Dict[str, Dict[str, float]],
    ) -> None:
        attributes = list(next(iter(class_attribute_mapping.values())).keys())
        attr_short = [attribute[:6] for attribute in attributes]
        class_headers = ["Class"] + attr_short + ["Dominant Attr"]
        class_rows = []
        for class_name, attrs in sorted(class_attribute_mapping.items()):
            dominant_attr = max(attrs.items(), key=lambda item: item[1])[0]
            row = [class_name] + [int(attrs[attribute]) for attribute in attributes] + [dominant_attr[:6]]
            class_rows.append(row)
        self._emit(
            "\nTarget Class Profile (HP):\n"
            + tabulate(class_rows, headers=class_headers, tablefmt="simple")
            + "\n",
            end="",
        )

    def weapon_damage_profile(
        self,
        weapon_damage_profile_mapping: Dict[str, Dict[str, float]],
        class_attribute_mapping: Dict[str, Dict[str, float]],
    ) -> None:
        attributes = list(next(iter(class_attribute_mapping.values())).keys())
        attr_short = [attribute[:6] for attribute in attributes]
        weapon_headers = ["Weapon"] + attr_short
        weapon_rows = []
        for weapon, profile in sorted(weapon_damage_profile_mapping.items()):
            row = [weapon] + [int(profile[attribute]) for attribute in attributes]
            weapon_rows.append(row)
        self._emit(
            "\nWeapon Damage Profile:\n"
            + tabulate(weapon_rows, headers=weapon_headers, tablefmt="simple")
            + "\n",
            end="",
        )

    def drone_setup(
        self,
        drones_config: List[Dict[str, Any]],
    ) -> None:
        drone_headers = ["Drone", "Weapon"]
        drone_rows = [[f"D{index}", cfg["weapon_type"]] for index, cfg in enumerate(drones_config)]
        self._emit(
            "\nDrone Setup:\n"
            + tabulate(drone_rows, headers=drone_headers, tablefmt="simple")
            + "\n",
            end="",
        )

    def target_setup(
        self,
        targets_config: List[Dict[str, Any]],
        class_attribute_mapping: Dict[str, Dict[str, float]],
    ) -> None:
        target_headers = ["Class", "Targets", "Count", "Sum", "Dominant Attr"]
        class_to_targets: Dict[str, List[str]] = {}
        for index, target_cfg in enumerate(targets_config):
            class_type = target_cfg["class_type"]
            class_to_targets.setdefault(class_type, []).append(f"T{index}")

        target_rows = []
        for class_type in sorted(class_to_targets.keys()):
            targets = class_to_targets[class_type]
            attributes = class_attribute_mapping[class_type]
            dominant_attr = max(attributes.items(), key=lambda item: item[1])[0]
            class_total_hp = sum(attributes.values())
            total_hp = len(targets) * class_total_hp
            target_rows.append([
                class_type,
                ", ".join(targets),
                len(targets),
                int(total_hp) if float(total_hp).is_integer() else total_hp,
                dominant_attr[:6],
            ])

        self._emit(
            "\nTarget Setup:\n"
            + tabulate(target_rows, headers=target_headers, tablefmt="simple")
            + "\n",
            end="",
        )

    def initial_setup(
        self,
        class_attribute_mapping: Dict[str, Dict[str, float]],
        weapon_damage_profile_mapping: Dict[str, Dict[str, float]],
        drones_config: List[Dict[str, Any]],
        targets_config: List[Dict[str, Any]],
    ) -> None:
        """Print all initial setup information (target class profile, weapon damage profile, drone setup, target setup)."""
        self.target_class_profile(class_attribute_mapping)
        self.weapon_damage_profile(weapon_damage_profile_mapping, class_attribute_mapping)
        self.drone_setup(drones_config)
        self.target_setup(targets_config, class_attribute_mapping)

    def engagement_matrix(
        self,
        title: str,
        row_labels: List[str],
        column_labels: List[str],
        matrix: List[List[float]],
        row_header: str = "Drone",
        precision: int = 3,
    ) -> None:
        headers = [row_header] + column_labels
        rows = []
        for row_label, values in zip(row_labels, matrix):
            rows.append([row_label] + [f"{value:.{precision}f}" for value in values])

        self._emit(
            f"\n{title}:\n"
            + tabulate(rows, headers=headers, tablefmt="simple")
            + "\n",
            end="",
        )

    def optimal_engagement_prediction(
        self,
        drones_config: List[Dict[str, Any]],
        targets_config: List[Dict[str, Any]],
        class_attribute_mapping: Dict[str, Dict[str, float]],
        weapon_damage_profile_mapping: Dict[str, Dict[str, float]],
        title: str = "Optimal Engagement Prediction (Greedy)",
        score_matrix: List[List[float]] | None = None,
        value_header: str = "Damage",
        value_matrix: List[List[float]] | None = None,
        value_precision: int = 0,
    ) -> None:
        assignments = []
        for drone_idx, drone_cfg in enumerate(drones_config):
            weapon_type = drone_cfg["weapon_type"]
            weapon_damage = weapon_damage_profile_mapping[weapon_type]

            for target_idx, target_cfg in enumerate(targets_config):
                class_type = target_cfg["class_type"]
                target_attrs = class_attribute_mapping[class_type]
                dominant_attr = max(target_attrs.items(), key=lambda item: item[1])[0]
                damage_to_dominant = weapon_damage[dominant_attr]

                assignments.append({
                    "drone_id": f"D{drone_idx}",
                    "target_id": f"T{target_idx}",
                    "weapon": weapon_type,
                    "target_class": class_type,
                    "dominant_attr": dominant_attr[:6],
                    "value": (
                        value_matrix[drone_idx][target_idx]
                        if value_matrix is not None
                        else damage_to_dominant
                    ),
                    "efficiency": (
                        score_matrix[drone_idx][target_idx]
                        if score_matrix is not None
                        else damage_to_dominant
                    ),
                })

        assigned_drones = set()
        assigned_targets = set()
        optimal_assignments = []
        assignments.sort(key=lambda item: item["efficiency"], reverse=True)

        for assignment in assignments:
            if assignment["drone_id"] not in assigned_drones and assignment["target_id"] not in assigned_targets:
                optimal_assignments.append(assignment)
                assigned_drones.add(assignment["drone_id"])
                assigned_targets.add(assignment["target_id"])

                if len(optimal_assignments) == min(len(drones_config), len(targets_config)):
                    break

        headers = ["Drone", "→", "Target", "Target Class", "Weapon", "Target Attr", value_header]
        rows = []
        for assignment in sorted(optimal_assignments, key=lambda item: item["drone_id"]):
            value = assignment["value"]
            if value_precision == 0:
                rendered_value: Any = int(value)
            else:
                rendered_value = f"{value:.{value_precision}f}"
            rows.append([
                assignment["drone_id"],
                "→",
                assignment["target_id"],
                assignment["target_class"],
                assignment["weapon"][:6],
                assignment["dominant_attr"],
                rendered_value,
            ])

        self._emit(
            f"\n{title}:\n"
            + tabulate(rows, headers=headers, tablefmt="simple")
            + "\n",
            end="",
        )

    def episode_start(
        self,
        episode_num: int,
        num_drones: int,
        num_targets: int,
        target_classes: List[str],
        weapon_types: List[str],
        target_hps: List[float],
    ) -> None:
        self._emit(
            f"\n{'=' * 60}\n"
            f"EPISODE {episode_num} START\n"
            f"{'=' * 60}\n"
            f"Drones: {num_drones}, Targets: {num_targets}\n"
            f"Target classes: {target_classes}\n"
            f"Drone weapons: {weapon_types}\n"
            f"Initial HPs: {target_hps}\n\n",
            end="",
        )

    def episode_step(
        self,
        step_count: int,
        actions: Dict[str, Any],
        target_hps: List[float],
        target_active: List[bool],
        rewards: Dict[str, float],
        overkill: Dict[int, float] | None = None,
    ) -> None:
        lines = [
            f"Step {step_count}:",
            f"  Actions: {actions}",
            f"  Target HPs: {target_hps}",
            f"  Target Active: {target_active}",
            f"  Step Rewards: {rewards}",
        ]
        if overkill is not None:
            lines.append(f"  Overkill: {overkill}")
        self._emit("\n".join(lines))

    def episode_summary(
        self,
        episode_num: int,
        step_count: int,
        total_rewards: Dict[str, float],
        ammo_used: Dict[str, int],
        weapon_types: List[str],
        done_reason: str,
        targets_neutralized: int,
        total_ammo_used: int,
    ) -> None:
        lines = [
            "",
            "=" * 60,
            f"EPISODE {episode_num} SUMMARY",
            "=" * 60,
            f"Done Reason:          {done_reason}",
            f"Steps:                {step_count}",
            f"Targets Neutralized:  {targets_neutralized}",
            f"Total Ammo Used:      {total_ammo_used}",
            "",
            "Agent Performance:",
        ]
        for agent_id, reward in sorted(total_rewards.items()):
            agent_idx = int(agent_id.split("_")[1])
            weapon_type = weapon_types[agent_idx]
            lines.append(f"  {agent_id}: {reward:.1f} (weapon: {weapon_type}, ammo: {ammo_used[agent_id]})")
        lines.extend(["=" * 60, ""])
        self._emit("\n".join(lines), end="")

    def learning_path(
        self,
        agent_headers: List[str],
        agent_rows: List[List[str]],
        target_headers: List[str],
        target_rows: List[List[str]],
    ) -> None:
        self._emit(
            "    --- Learning Path ---\n"
            + tabulate(agent_rows, headers=agent_headers, tablefmt="simple")
            + "\n\n"
            + tabulate(target_rows, headers=target_headers, tablefmt="simple")
            + "\n",
            end="",
        )

    def agent_clustering(
        self,
        drone_weapon_map: List[str],
        similarity_lines: List[str],
    ) -> None:
        lines = ["", "--- Agent Latent Vector Analysis ---", "Agent Types:"]
        for index, weapon in enumerate(drone_weapon_map):
            lines.append(f"  Agent {index}: {weapon}")
        lines.extend(similarity_lines)
        self._emit("\n".join(lines), end="")

    def policy_performance_summary(
        self,
        table_data: List[List[str]],
        headers: List[str],
    ) -> None:
        self._emit(
            "\n"
            + "=" * 60
            + "\nPOLICY PERFORMANCE SUMMARY\n"
            + "=" * 60
            + "\n"
            + tabulate(table_data, headers=headers, tablefmt="grid")
            + "\n"
            + "=" * 60,
            end="",
        )

    def policy_performance_comparison(
        self,
        mappings_file: str,
        headers: List[str],
        cmp_data: List[List[str]],
    ) -> None:
        metric_lines = [
            "Ammo Eff = targets / ammo (higher = fewer wasted shots)",
            "Dmg Eff  = net_dmg / gross_dmg (higher = less overkill)",
        ]
        lines = [
            "",
            "=" * 60,
            "POLICY PERFORMANCE COMPARISON (vs Random Baseline)",
            "=" * 60,
            f"Mappings: {mappings_file}",
            *metric_lines,
            tabulate(cmp_data, headers=headers, tablefmt="grid"),
            "=" * 60,
        ]
        self._emit("\n".join(lines), end="")

    def engagement_table(
        self,
        policy_type: str,
        headers: List[str] | None,
        payload: Any,
    ) -> None:
        lines = [f"\nActual Engagement Summary (Best Episode) - {policy_type}:"]
        if headers is None:
            lines.append(str(payload))
        else:
            lines.append(tabulate(payload, headers=headers, tablefmt="simple"))
        lines.append("")
        self._emit("\n".join(lines), end="")

    def demo_header(
        self,
        config_path: str,
        mappings_file: str,
        world_size: Any,
        seed: Any,
        policy_types: List[str],
        num_episodes: int,
    ) -> None:
        self._emit(
            "\n"
            + "=" * 60
            + "\nZK-MRTA ENVIRONMENT DEMO\n"
            + "=" * 60
            + f"\nConfig File: {config_path}"
            + f"\nMappings File: {mappings_file}"
            + f"\nWorld Size: {world_size}"
            + f"\nRandom Seed: {seed}"
            + f"\nPolicy Types: {policy_types}"
            + f"\nNumber of Episodes per Policy: {num_episodes}"
        )

    def policy_run_header(self, policy_type: str) -> None:
        self._emit(f"\n>>> Running policy: {policy_type}", end="")

    def continuous_run_progress(
        self,
        steps: int,
        throughput: float,
        coordination: str,
        ammo_eff: float,
        collisions: int,
    ) -> None:
        self._emit(
            f"  Continuous Run Progress: Steps={steps}, "
            f"Throughput={throughput:.1f} N/100, "
            f"Coordination={coordination}, "
            f"Ammo Eff={ammo_eff:.3f}, "
            f"Collisions={collisions}"
        )

    def episode_run_progress(
        self,
        episode_num: int,
        steps: int,
        targets_neutralized: int,
        total_net_damage: float,
        total_overkill: float,
        total_reward: float,
    ) -> None:
        self._emit(
            f"  Episode {episode_num}: Steps={steps}, "
            f"Total Neutralized={targets_neutralized}, "
            f"Total Net Damage={total_net_damage:.0f}, "
            f"Total Wasted HP={total_overkill:.0f}, "
            f"Reward={total_reward:.0f}"
        )

    def saved_episodes(self, files: List[str]) -> None:
        self._emit(f"  Total saved episodes: {len(files)}")

    def policy_steps_summary(self, first: Any, best: Any, mid: Any) -> None:
        self._emit(f"  Steps: first={first}, best={best}, mid={mid}")

    def policy_final_summary(
        self,
        steps: int,
        targets_neutralized: int,
        total_net_damage: float,
        total_overkill: float,
        total_collisions: int,
    ) -> None:
        self._emit(
            f"  Final Summary: Steps={steps}, "
            f"Total Neutralized={targets_neutralized}, "
            f"Total Net Damage={total_net_damage:.0f}, "
            f"Total Wasted HP={total_overkill:.0f}, "
            f"Collisions={total_collisions}"
        )

    def policy_first_step(self, first: Any) -> None:
        self._emit(f"  Steps: first={first}")

    def aggregate_statistics(
        self,
        total_episodes: int,
        num_episodes: int,
        policy_count: int,
        avg_steps: float,
        avg_targets: float,
        avg_ammo: float,
        avg_overkill: float,
        per_policy_rows: List[str],
        per_agent_rows: List[str],
    ) -> None:
        lines = [
            "",
            "=" * 60,
            "AGGREGATE STATISTICS",
            "=" * 60,
            f"Total Episodes: {total_episodes} ({num_episodes} per policy × {policy_count} policies)",
            f"Average Steps:              {avg_steps:.1f}",
            f"Average Targets Neutralized: {avg_targets:.1f}",
            f"Average Ammo Used:          {avg_ammo:.1f}",
            f"Average Overkill Damage:    {avg_overkill:.1f}",
            *per_policy_rows,
            "",
            "Per-Agent Average Rewards:",
            *per_agent_rows,
            "=" * 60,
        ]
        self._emit("\n".join(lines), end="")

    def demo_complete(self) -> None:
        self._emit("\nDemo complete! ✓", end="")

    def latent_world_debug(
        self,
        drones_config: List[Dict[str, Any]],
        targets_config: List[Dict[str, Any]],
        precision: int = 3,
        max_components: int = 6,
    ) -> None:
        """Display latent-world drone weapons and target attributes grouped by mode.

        Args:
            drones_config: List of drone configs with 'mode_id' and 'latent_vector' keys.
            targets_config: List of target configs with 'mode_id' and 'latent_vector' keys.
            precision: Decimal places for vector component display.
            max_components: Maximum latent vector components to show (truncates if longer).
        """
        from collections import defaultdict

        drones_by_mode: Dict[int, List[Tuple[int, Tuple[float, ...]]]] = defaultdict(list)
        targets_by_mode: Dict[int, List[Tuple[int, Tuple[float, ...]]]] = defaultdict(list)

        for idx, dcfg in enumerate(drones_config):
            mode_id = int(dcfg.get("mode_id", -1))
            latent_vector = tuple(float(v) for v in dcfg.get("latent_vector", []))
            drones_by_mode[mode_id].append((idx, latent_vector))

        for idx, tcfg in enumerate(targets_config):
            mode_id = int(tcfg.get("mode_id", -1))
            latent_vector = tuple(float(v) for v in tcfg.get("latent_vector", []))
            targets_by_mode[mode_id].append((idx, latent_vector))

        all_modes = sorted(set(drones_by_mode.keys()) | set(targets_by_mode.keys()))

        lines = ["", "=" * 60, "LATENT WORLD DEBUG (Gaussian Mixture)", "=" * 60]

        for mode_id in all_modes:
            lines.append(f"\nMode {mode_id}:")
            lines.append("-" * 40)

            drones = drones_by_mode.get(mode_id, [])
            if drones:
                lines.append(f"  Drones (Weapons): {len(drones)}")
                for d_idx, vec in drones:
                    vec_str = self._format_vector(vec, precision, max_components)
                    lines.append(f"    D{d_idx}: [{vec_str}]")
            else:
                lines.append("  Drones: None")

            targets = targets_by_mode.get(mode_id, [])
            if targets:
                lines.append(f"  Targets (Attributes): {len(targets)}")
                for t_idx, vec in targets:
                    vec_str = self._format_vector(vec, precision, max_components)
                    lines.append(f"    T{t_idx}: [{vec_str}]")
            else:
                lines.append("  Targets: None")

        lines.extend(["=" * 60, ""])
        self._emit("\n".join(lines), end="")

    def _format_vector(self, vec: Tuple[float, ...], precision: int, max_components: int) -> str:
        """Format a vector tuple for display with truncation if needed."""
        if not vec:
            return ""
        if len(vec) <= max_components:
            return ", ".join(f"{v:.{precision}f}" for v in vec)
        shown = vec[:max_components]
        formatted = ", ".join(f"{v:.{precision}f}" for v in shown)
        return f"{formatted}, ... ({len(vec) - max_components} more)"
