# TabulaDrone

Reinforcement Learning environments for drone target engagement scenarios, built with Gymnasium.

## Overview

TabulaDrone provides minimal, well-defined RL environments for training and evaluating drone engagement policies. This project starts with a foundational single-drone, single-target scenario and will scale to multi-agent, multi-target environments.

## Environments

### DroneEngageSingleTarget-v0

A minimal environment where:
- Single static drone with limited ammunition
- Single static target with hit points
- 2D world space
- Simple action space: Idle or Fire
- Sparse reward: +1.0 for target neutralization

**Status:** ✅ Complete

### DroneEngageMultiTarget-v0

A multi-target engagement environment where:
- Single static drone with limited ammunition
- Multiple static targets (supports 20+ targets)
- Configurable target classes (A/B/C with different HP values)
- Zone-based target organization
- Dynamic action space: Idle or Fire at specific target
- Per-target incremental rewards: +1.0 for each target neutralized
- Partial success allowed: Episode can end with some targets remaining
- Termination: All targets neutralized OR no ammo remaining
- Truncation: Maximum time steps reached

**Key Features:**
- Configurable target list with position, class, and zone
- Dynamic observation space based on number of targets
- Per-step info dict with detailed target status arrays
- Deterministic with seed support
- Fully Gymnasium-compliant

**Status:** ✅ Complete

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Single Target Environment

```python
from tabula_drone.envs.drone_engage_single_target_v0 import DroneEngageSingleTargetV0

env = DroneEngageSingleTargetV0(
    drone_ammo_max=5,
    drone_damage_per_shot=35.0,
    target_class_type="A",  # 100 HP
    max_steps=50,
)

obs, info = env.reset(seed=42)
terminated = False
truncated = False

while not (terminated or truncated):
    action = env.action_space.sample()  # 0=Idle, 1=Fire
    obs, reward, terminated, truncated, info = env.step(action)
```

### Multi-Target Environment

```python
from tabula_drone.envs.drone_engage_multi_target_v0 import DroneEngageMultiTargetV0

env = DroneEngageMultiTargetV0(
    drone_ammo_max=15,
    drone_damage_per_shot=35.0,
    targets_config=[
        {'position': (200, 200), 'class_type': 'A', 'zone_id': 'zone_1'},
        {'position': (800, 200), 'class_type': 'B', 'zone_id': 'zone_2'},
        {'position': (500, 800), 'class_type': 'C', 'zone_id': 'zone_3'},
    ],
    max_steps=100,
)

obs, info = env.reset(seed=42)
terminated = False
truncated = False

while not (terminated or truncated):
    # 0=Idle, 1-3=Fire at specific target
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Access per-target info
    print(f"Target HPs: {info['target_hps']}")
    print(f"Target Active: {info['target_active']}")
```

### Running Demos

```bash
# Single target demo
python main.py

# Multi-target demo
python main_multi_target.py
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
