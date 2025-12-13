"""
Command-line interface for TabulaDrone Episode Viewer.
"""

import argparse
import glob
import os
import sys

from viewer.state_adapter import load_episode, extract_initial_state
from viewer.draw import display_viewer

LOGS_DIR = "logs"


def find_latest_episode() -> str:
    """
    Find the latest episode log file in the logs directory.
    
    Returns:
        Path to the latest episode log file
        
    Raises:
        SystemExit: If logs directory doesn't exist or contains no episodes
    """
    if not os.path.isdir(LOGS_DIR):
        print(f"Error: Logs directory '{LOGS_DIR}' not found.", file=sys.stderr)
        sys.exit(1)
    
    pattern = os.path.join(LOGS_DIR, "episode_*.json")
    episode_files = glob.glob(pattern)
    
    if not episode_files:
        print(f"Error: No episode files found in '{LOGS_DIR}'.", file=sys.stderr)
        sys.exit(1)
    
    # Sort descending by filename (contains timestamp)
    episode_files.sort(reverse=True)
    return episode_files[0]


def show_command(episode_path: str) -> None:
    """
    Execute the show command.
    
    Args:
        episode_path: Path to episode log JSON file
    """
    episode_data = load_episode(episode_path)
    state = extract_initial_state(episode_data)
    
    print(f"Loading episode: {episode_path}")
    print(f"  Version: {state['version']}")
    print(f"  World size: {state['world_size']}")
    print(f"  Drones: {len(state['drones'])}")
    print(f"  Targets: {len(state['targets'])}")
    
    display_viewer(state)


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
