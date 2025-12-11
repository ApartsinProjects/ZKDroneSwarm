"""
Main entry point for the FalconX snapshot viewer.

This module allows the viewer to be run as a Python module:
python -m viewer show --snapshot /path/to/snapshot.json
"""

import sys
from viewer.cli import main

if __name__ == "__main__":
    sys.exit(main())
