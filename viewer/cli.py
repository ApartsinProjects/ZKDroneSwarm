"""
Command-line interface for TabulaDrone Episode Viewer.
"""

import argparse
import glob
import json
import os
import sys

from viewer.state_adapter import load_episode, extract_initial_state
from viewer.draw import display_viewer

LOGS_DIR = "logs"


def _get_episode_step_count(file_path: str) -> int:
    """
    Extract step count from episode log file.
    
    Args:
        file_path: Path to episode JSON file
        
    Returns:
        Number of steps in the episode, or 0 if not found
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            return len(data.get('steps', []))
    except (json.JSONDecodeError, IOError):
        return 0


def _find_latest_scenario() -> str:
    """
    Find the most recent scenario folder (run_*) in logs directory.
    
    Returns:
        Path to the most recent scenario folder
        
    Raises:
        SystemExit: If no scenario folders found
    """
    if not os.path.isdir(LOGS_DIR):
        print(f"Error: Logs directory '{LOGS_DIR}' not found.", file=sys.stderr)
        sys.exit(1)
    
    scenario_folders = glob.glob(os.path.join(LOGS_DIR, "run_*"))
    scenario_folders = [f for f in scenario_folders if os.path.isdir(f)]
    
    if not scenario_folders:
        print(f"Error: No scenario folders (run_*) found in '{LOGS_DIR}'.", file=sys.stderr)
        sys.exit(1)
    
    # Sort descending by folder name (timestamp in name)
    scenario_folders.sort(reverse=True)
    return scenario_folders[0]


def find_all_episodes() -> list[str]:
    """
    Find all episode log files from the most recent scenario.
    
    Episodes are grouped by policy (alphabetical order), and within each
    policy group sorted by step count descending (higher to lower).
    
    Returns:
        List of episode file paths
        
    Raises:
        SystemExit: If logs directory doesn't exist or contains no episodes
    """
    scenario_path = _find_latest_scenario()
    
    # Find all policy subfolders (alphabetical order)
    policy_folders = sorted([
        d for d in os.listdir(scenario_path)
        if os.path.isdir(os.path.join(scenario_path, d))
    ])
    
    all_episodes = []
    
    for policy in policy_folders:
        episodes_dir = os.path.join(scenario_path, policy, "episodes")
        if not os.path.isdir(episodes_dir):
            continue
        
        # Find all episode files in this policy
        pattern = os.path.join(episodes_dir, "episode_*.json")
        policy_episodes = glob.glob(pattern)
        
        # Sort by step count descending (higher to lower)
        policy_episodes.sort(key=_get_episode_step_count, reverse=True)
        all_episodes.extend(policy_episodes)
    
    if not all_episodes:
        print(f"Error: No episode files found in '{scenario_path}'.", file=sys.stderr)
        sys.exit(1)
    
    return all_episodes


def find_latest_episode() -> str:
    """
    Find the latest episode log file in the logs directory.
    
    Returns:
        Path to the latest episode log file
        
    Raises:
        SystemExit: If logs directory doesn't exist or contains no episodes
    """
    return find_all_episodes()[0]


def show_command(episode_path: str) -> None:
    """
    Execute the show command.
    
    Args:
        episode_path: Path to episode log JSON file
    """
    episode_files = find_all_episodes()
    current_index = episode_files.index(episode_path)
    
    episode_data = load_episode(episode_path)
    state = extract_initial_state(episode_data, episode_path=episode_path)
    
    print(f"Loading episode: {episode_path}")
    print(f"  Version: {state['version']}")
    print(f"  World size: {state['world_size']}")
    print(f"  Drones: {len(state['drones'])}")
    print(f"  Targets: {len(state['targets'])}")
    
    display_viewer(state, episode_files, current_index)


def main():
    """Main entry point for viewer CLI."""
    parser = argparse.ArgumentParser(
        prog="viewer",
        description="TabulaDrone Episode Viewer"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # show command
    show_parser = subparsers.add_parser("show", help="Display episode visualization")
    show_parser.add_argument(
        "--episode",
        help="Path to episode log JSON file (default: latest in logs/)"
    )
    
    args = parser.parse_args()
    
    if args.command == "show":
        episode_path = args.episode or find_latest_episode()
        show_command(episode_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
