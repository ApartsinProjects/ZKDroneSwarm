# TabulaDrone

Reinforcement Learning environments for drone target engagement scenarios, built with PettingZoo.

## Overview

TabulaDrone provides a multi-agent RL environment for Zero-Knowledge Multi-Robot Task Allocation (ZK-MRTA) research. Multiple drones engage multiple targets without prior knowledge of task attributes, agent capabilities, or inter-agent communication.

## Environment

### DroneEngageZKMRTA-v0

A multi-agent PettingZoo environment where:
- Multiple static drones with unlimited ammunition
- Multiple static targets with configurable classes (A/B/C with different HP values)
- Zero-knowledge constraints: agents don't know target HP, classes, or their own damage capabilities
- No communication between agents
- Zone-based target organization
- Action space per agent: Noop or Fire at specific target
- Shared reward: +1.0 for each target neutralized (distributed to all agents)
- Observations show only target positions and binary active status

**Key Features:**
- Configurable drones with weapon types (light/medium/heavy)
- Configurable targets with position, class, and zone
- ZK-MRTA compliant observations (no HP, no classes, no ammo)
- Parallel execution (all agents act simultaneously)
- Deterministic with seed support
- Fully PettingZoo-compliant

**Status:** ✅ Production

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### ZK-MRTA Multi-Agent Environment

```python
from tabula_drone.envs.drone_engage_zk_mrta_v0 import DroneEngageZKMRTA

# Create environment with multiple drones and targets
env = DroneEngageZKMRTA(
    drones_config=[
        {'position': (100, 100), 'weapon_type': 'light'},
        {'position': (900, 100), 'weapon_type': 'medium'},
        {'position': (500, 900), 'weapon_type': 'heavy'},
    ],
    targets_config=[
        {'position': (200, 200), 'class_type': 'A', 'zone_id': 'zone_1'},
        {'position': (800, 200), 'class_type': 'B', 'zone_id': 'zone_2'},
        {'position': (500, 800), 'class_type': 'C', 'zone_id': 'zone_3'},
    ],
    max_steps=100,
)

# Reset returns observations for all agents
observations, infos = env.reset(seed=42)

# Parallel execution: all agents act simultaneously
while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
```

### Running Demo

```bash
# ZK-MRTA demo with random policy
python main_zk_mrta.py
```

## Project Structure

```
TabulaDrone/
├── tabula_drone/           # Main package
│   ├── core/              # Shared state representations
│   ├── envs/              # Environment implementations
│   ├── policies/          # Policy implementations
│   └── scenarios/         # Scenario utilities
├── tests/                 # Test suite
├── main_zk_mrta.py       # Demo script
└── requirements.txt       # Dependencies
```

## Development

This project follows the Baby Steps methodology - each feature is developed in small, atomic, validated increments.

## License

TBD
