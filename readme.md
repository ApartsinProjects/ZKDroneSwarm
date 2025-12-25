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
- Action space per agent: Noop or Fire at specific target
- Reward: +1.0 for killing blow on target

**Observation Modes:**

The environment supports two observation modes controlled by `observation_mode` parameter:

| Mode | Description | Use Case |
|------|-------------|----------|
| `minimal` (default) | Target positions + active status only | Pure ZK-MRTA, no inter-agent visibility |
| `collaborative` | Adds other agents' actions and rewards | Collaborative filtering, multi-agent learning |

*Minimal mode observations:*
- Target positions (x, y) and binary active status

*Collaborative mode adds:*
- `selected_targets`: Which target each agent fired at last step
- `observed_rewards`: What reward each agent received last step

**Noise Parameters (collaborative mode):**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reward_noise` | float | 0.0 | Gaussian noise σ added to actual rewards |
| `observation_noise` | float | 0.0 | Additional Gaussian noise σ when observing other agents' rewards |

When both noise parameters are non-zero, an agent observing another agent's reward sees:
`observed = true_reward + N(0, reward_noise) + N(0, observation_noise)`

Own rewards have only `reward_noise`; observed rewards from others have both noise sources.

**Note:** Both modes maintain zero-knowledge about *capabilities* — agents never see HP values, damage profiles, weapon types, or class types. Collaborative mode only reveals *actions and outcomes*, not *why* those outcomes occurred.

**Key Features:**
- Configurable drones with weapon types (light/medium/heavy)
- Configurable targets with position and class
- ZK-MRTA compliant observations (no HP, no classes, no damage capabilities)
- Configurable observation mode (minimal or collaborative)
- Parallel execution (all agents act simultaneously)
- Deterministic with seed support
- Fully PettingZoo-compliant

**Status:** ✅ Production

## Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip3 install -r requirements.txt
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
        {'position': (200, 200), 'class_type': 'A'},
        {'position': (800, 200), 'class_type': 'B'},
        {'position': (500, 800), 'class_type': 'C'},
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
python3 main_zk_mrta.py
```

### Episode Viewer

Visualize episode logs with the built-in viewer:

```bash
# Show latest episode (auto-discovers from logs/)
python3 -m viewer show

# Show specific episode
python3 -m viewer show --episode logs/episode_xxx.json
```

The viewer displays a split-panel view (60/40 layout) with:
- **Left panel:** Map with world bounds, grid, drones (triangles by weapon type), targets (circles)
- **Right panel:** Tabbed info panel for additional details

## Project Structure

```
TabulaDrone/
├── tabula_drone/           # Main package
│   ├── core/              # Shared state representations
│   ├── envs/              # Environment implementations
│   ├── logging/           # Episode logging
│   ├── policies/          # Policy implementations
│   └── scenarios/         # Scenario utilities
├── viewer/                # Episode visualization CLI
├── tests/                 # Test suite
├── logs/                  # Episode log output (generated)
├── config/                # Scenario configuration files
├── main_zk_mrta.py       # Demo script
└── requirements.txt       # Dependencies
```

## Development

This project follows the Baby Steps methodology - each feature is developed in small, atomic, validated increments.

## License

TBD
