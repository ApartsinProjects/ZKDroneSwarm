# TabulaDrone

Reinforcement Learning environments for drone target engagement scenarios, built with Gymnasium.

## Overview

TabulaDrone provides minimal, well-defined RL environments for training and evaluating drone engagement policies. This project starts with a foundational single-drone, single-target scenario and will scale to multi-agent, multi-target environments.

## Environments

### DroneEngageSingleTarget-v0 (MVP - Step 0)

A minimal environment where:
- Single static drone with limited ammunition
- Single static target with hit points
- 2D world space
- Simple action space: Idle or Fire
- Sparse reward: +1.0 for target neutralization

**Status:** In Development

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
TabulaDrone/
├── tabula_drone/           # Main package
│   ├── envs/              # Environment implementations
│   └── utils/             # Shared utilities
├── tests/                 # Test suite
├── spec_0.md             # Environment specification
└── requirements.txt       # Dependencies
```

## Development

This project follows the Baby Steps methodology - each feature is developed in small, atomic, validated increments.

## License

TBD
