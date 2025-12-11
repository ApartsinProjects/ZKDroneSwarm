"""
Command-line interface for the FalconX snapshot viewer.

This module provides a command-line interface for visualizing snapshot files.
"""

import argparse
import csv
import sys
from typing import List, Optional, Tuple
import json

from viewer.io.io import load_snapshot, SnapshotValidationError
from viewer.draw import plot_world


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Args:
        args: Command-line arguments. If None, sys.argv[1:] will be used.
        
    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="FalconX Snapshot Viewer - Visualize simulation snapshots"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    subparsers.required = True
    
    # Show command
    show_parser = subparsers.add_parser("show", help="Show a snapshot visualization")
    show_parser.add_argument(
        "--snapshot",
        required=True,
        help="Path to the snapshot JSON file"
    )
    show_parser.add_argument(
        "--save",
        help="Path to save the visualization as a PNG image"
    )
    show_parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for the saved image (default: 150)"
    )
    show_parser.add_argument(
        "--export-csv",
        help="Path to export target data as CSV"
    )
    show_parser.add_argument(
        "--show-ranges",
        action="store_true",
        help="Show missile range circles on the map"
    )
    show_parser.add_argument(
        "--ranges-origin",
        nargs=2,
        type=float,
        default=[0.0, 0.0],
        metavar=('X', 'Y'),
        help="Origin coordinates for range circles (default: 0.0 0.0)"
    )
    show_parser.add_argument(
        "--show-spawn-regions",
        action="store_true",
        help="Show target spawn regions on the map"
    )
    
    return parser.parse_args(args)


def export_targets_to_csv(targets: List[dict], csv_path: str) -> None:
    """
    Export target data to a CSV file.
    
    Args:
        targets: List of target dictionaries.
        csv_path: Path to save the CSV file.
    """
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id', 'x', 'y', 'status', 'type']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for target in targets:
            writer.writerow({
                'id': target['id'],
                'x': target['x'],
                'y': target['y'],
                'status': target['status'],
                'type': target['type']
            })
    
    print(f"Exported {len(targets)} targets to {csv_path}")


def print_snapshot_summary(snap: dict) -> None:
    """
    Print a summary of the snapshot to stdout.
    
    Args:
        snap: The snapshot data dictionary.
    """
    # Extract metadata
    meta = snap["meta"]
    world = snap["world"]
    catalogs = snap["catalogs"]
    summary = snap["summary"]
    status_counts = summary["target_status_counts"]
    
    print("=== Snapshot Summary ===")
    print(f"World ID: {meta['world_id']}")
    print(f"Created: {meta['created_utc']}")
    print(f"Seed: {meta['seed']}")
    print(f"Time: {meta['t_s']:.2f} seconds")
    print(f"Version: {meta['version']}")
    print()
    
    print("World Bounds:")
    print(f"  Width: {world['bounds_m']['width']:.1f} meters")
    print(f"  Height: {world['bounds_m']['height']:.1f} meters")
    print()
    
    # Get target information from the new schema
    target_count = 0
    if "target_spawn_region" in world and "target_instances" in world["target_spawn_region"]:
        target_instances = world["target_spawn_region"]["target_instances"]
        target_count = target_instances["count"]
    
    print("Targets:")
    print(f"  Count: {target_count}")
    print(f"  Intact: {status_counts.get('intact', 0)}")
    print(f"  Damaged: {status_counts.get('damaged', 0)}")
    print(f"  Destroyed: {status_counts.get('destroyed', 0)}")
    
    # Print target type counts if available
    if "target_type_counts" in summary:
        print("  Types:")
        for target_type, count in sorted(summary["target_type_counts"].items()):
            print(f"    {target_type}: {count}")
    
    # Print resupply station info if present
    if "resupply_station" in world:
        print()
        print("Resupply Station:")
        station = world["resupply_station"]
        print(f"  Position: ({station['pos']['x']:.1f}, {station['pos']['y']:.1f})")
        
        if "stock" in station:
            print("  Stock:")
            for missile_type, count in sorted(station["stock"].items()):
                print(f"    {missile_type}: {count}")
    
    # Print target spawn region info if present
    if "target_spawn_region" in world:
        print()
        print("Target Spawn Region:")
        region = world["target_spawn_region"]
        if "y_fraction" in region:
            y_fractions = region["y_fraction"]
            if isinstance(y_fractions, list) and len(y_fractions) == 2:
                height = world['bounds_m']['height']
                y_min = y_fractions[0] * height
                y_max = y_fractions[1] * height
                print(f"  Y Range: {y_min:.1f} to {y_max:.1f} meters ({y_fractions[0]:.2f} to {y_fractions[1]:.2f} of height)")


#python3 -m viewer show --snapshot out/snapshot.json
def show_command(args: argparse.Namespace) -> int:
    """
    Show a snapshot visualization.
    
    Args:
        args: Command-line arguments.
        
    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    try:
        # Load snapshot
        snap = load_snapshot(args.snapshot)
        
        # Print summary
        print_snapshot_summary(snap)
        
        # Export targets to CSV if requested
        if args.export_csv:
            # Get targets from the new schema
            targets = []
            if "world" in snap and "target_spawn_region" in snap["world"]:
                if "target_instances" in snap["world"]["target_spawn_region"]:
                    target_instances = snap["world"]["target_spawn_region"]["target_instances"]
                    if "items" in target_instances:
                        targets = target_instances["items"]
            
            export_targets_to_csv(targets, args.export_csv)
        
        # Parse ranges origin
        ranges_origin = (args.ranges_origin[0], args.ranges_origin[1])
        
        # Show visualization
        plot_world(
            snap=snap,
            save_path=args.save,
            dpi=args.dpi,
            show_ranges=args.show_ranges,
            ranges_origin=ranges_origin,
            show_spawn_regions=args.show_spawn_regions
        )
        
        return 0
        
    except SnapshotValidationError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        return 1
        
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in snapshot file", file=sys.stderr)
        return 1
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for the command-line interface.
    
    Args:
        args: Command-line arguments. If None, sys.argv[1:] will be used.
        
    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    parsed_args = parse_args(args)
    
    if parsed_args.command == "show":
        return show_command(parsed_args)
    else:
        print(f"Unknown command: {parsed_args.command}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
