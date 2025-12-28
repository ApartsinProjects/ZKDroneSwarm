# TabulaDrone Project Statistics Report

**Generated:** December 28, 2025  
**Project Started:** November 13, 2025  
**Last Commit:** December 28, 2025  

---

## 📊 Lines of Code Summary

| Category | Lines of Code | Files |
|----------|---------------|-------|
| **Total (Production + Tests)** | **11,581** | **48** |
| Core Package (`tabula_drone/`) | 3,924 | 20 |
| Viewer (`viewer/`) | 2,763 | 17 |
| Tests (`tests/`) | 4,129 | 10 |
| Main Script | 765 | 1 |

### Breakdown by Module

| Module | Lines | Description |
|--------|-------|-------------|
| `tabula_drone/policies/` | 1,287 | Policy implementations (7 policies) |
| `tabula_drone/scenarios/` | 800 | Scenario building & weapon assignment |
| `tabula_drone/envs/` | 641 | Environment implementations |
| `tabula_drone/logging/` | 405 | Episode logging + decentralized state |
| `tabula_drone/config/` | 390 | Configuration loading |
| `tabula_drone/core/` | 140 | Core state representations |

### Largest Files

| Rank | File | Lines |
|------|------|-------|
| 1 | `tests/test_drone_engage_zk_mrta_v0.py` | 1,034 |
| 2 | `tabula_drone/envs/drone_engage_zk_mrta_v0.py` | 638 |
| 3 | `tabula_drone/scenarios/scenario_builder.py` | 630 |
| 4 | `tests/test_episode_logger.py` | 450 |
| 5 | `tests/test_scenario_builder.py` | 425 |

---

## 🏗️ Architecture & Design

### Design Patterns Used

| Pattern | Implementation | Location |
|---------|----------------|----------|
| **Dataclass** | Immutable state representations | `core/states.py`, `config/config_loader.py` |
| **Strategy Pattern** | Policy abstractions (Random, Oracle) | `policies/` |
| **Builder Pattern** | Scenario construction | `scenarios/scenario_builder.py` |
| **Adapter Pattern** | State format conversion for viewer | `viewer/state_adapter.py` |
| **Component-Based UI** | Hierarchical viewer components | `viewer/components/` |
| **ParallelEnv** | PettingZoo multi-agent API | `envs/drone_engage_zk_mrta_v0.py` |

### Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                        │
│  main_zk_mrta.py (entry point, orchestration)               │
├─────────────────────────────────────────────────────────────┤
│                      Policy Layer                            │
│  RandomPolicy, OracleTimeToKillPolicy                        │
├─────────────────────────────────────────────────────────────┤
│                   Environment Layer                          │
│  DroneEngageZKMRTA (PettingZoo ParallelEnv)                 │
├─────────────────────────────────────────────────────────────┤
│                      Core Layer                              │
│  DroneState, TargetState, WorldState, AttributeProfile       │
├─────────────────────────────────────────────────────────────┤
│                  Infrastructure Layer                        │
│  ConfigLoader, ScenarioBuilder, EpisodeLogger               │
├─────────────────────────────────────────────────────────────┤
│                   Visualization Layer                        │
│  Viewer CLI, MapPanel, InfoPanel, ResultsPanel, SummaryPanel │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧱 Class Structure

### Core Classes (13 Dataclasses)

**State Representations:**
- `DroneState` - Drone position, weapon type, ammo, damage profile
- `TargetState` - Target position, class, attributes, active status
- `WorldState` - World size, time step, max steps, scenario ID
- `AttributeProfile` - Multi-attribute health/damage system

**Configuration:**
- `WorldConfig`, `DronesConfig`, `TargetsConfig`
- `EnvironmentConfig`, `PolicyConfig`, `ExecutionConfig`
- `LoggingConfig`, `MappingsConfig`, `ScenarioConfig`

### Policy Classes (7)
- `RandomPolicy` - ZK-compliant random action selection
- `MaxDamageOracle` - Oracle maximizing damage per step
- `MinTTKOracle` - Oracle minimizing time-to-kill
- `EpGreedyCFPolicy` - Centralized ε-greedy collaborative filtering
- `DecentralizedEpGreedyCFPolicy` - Decentralized ε-greedy CF (ZK-MRTA compliant)
- `UCBCFPolicy` - Upper Confidence Bound collaborative filtering

### Environment Class (1)
- `DroneEngageZKMRTA` - Main PettingZoo ParallelEnv

### Viewer Components (9)
- `BaseComponent` - Abstract base for all UI components
- `MapPanel` - World visualization with drones/targets
- `TabContainer` - Tab navigation system
- `InfoPanel`, `ResultsPanel`, `SummaryPanel`, `EmptyPanel`
- `TrainingPathPanel` - Latent space visualization for CF policies

---

## 🧪 Testing Statistics

| Metric | Value |
|--------|-------|
| **Total Test Functions** | 136 |
| **Test Files** | 6 |
| **Test LOC** | 2,594 |
| **Test-to-Code Ratio** | 1:1.97 |

### Test Coverage by Module

| Test File | Tests | Lines |
|-----------|-------|-------|
| `test_drone_engage_zk_mrta_v0.py` | 45+ | 1,034 |
| `test_episode_logger.py` | 25+ | 450 |
| `test_scenario_builder.py` | 25+ | 425 |
| `test_config_loader.py` | 20+ | 339 |
| `test_oracle_policy.py` | 12+ | 239 |
| `test_state_adapter.py` | 8+ | 104 |

---

## 📦 Dependencies

| Category | Package | Version |
|----------|---------|---------|
| **Core** | gymnasium | ≥0.26.0 |
| **Core** | numpy | ≥1.21.0 |
| **Core** | pettingzoo | ≥1.24.0 |
| **Visualization** | matplotlib | ≥3.5.0 |
| **Testing** | pytest | ≥7.0.0 |

---

## 📁 Project Structure

```
TabulaDrone/
├── tabula_drone/           # Main package (2,609 LOC)
│   ├── core/              # State representations (140 LOC)
│   ├── config/            # Configuration loading (390 LOC)
│   ├── envs/              # Environment implementations (641 LOC)
│   ├── logging/           # Episode logging (326 LOC)
│   ├── policies/          # Policy implementations (306 LOC)
│   ├── scenarios/         # Scenario utilities (800 LOC)
│   └── utils/             # Utility functions
├── viewer/                # Episode visualization (2,168 LOC)
│   ├── components/
│   │   ├── base/          # Base component class
│   │   ├── containers/    # MapPanel, TabContainer
│   │   └── panels/        # Info, Results, Summary panels
│   ├── cli.py             # CLI interface
│   ├── draw.py            # Drawing utilities
│   └── state_adapter.py   # State format conversion
├── tests/                 # Test suite (2,594 LOC)
├── config/                # Configuration files
│   ├── scenario.json      # Scenario configuration
│   └── mappings.json      # Class/weapon mappings
├── specs/                 # Documentation
├── logs/                  # Episode log output
├── main_zk_mrta.py        # Demo script (322 LOC)
└── requirements.txt       # Dependencies
```

---

## 🎯 Key Features

1. **Zero-Knowledge Multi-Robot Task Allocation (ZK-MRTA)**
   - No prior knowledge of task attributes (HP, classes)
   - No knowledge of agent capabilities (damage)
   - No communication between agents

2. **Multi-Agent Environment**
   - PettingZoo ParallelEnv API compliant
   - Configurable drones with weapon types (light/medium/heavy)
   - Configurable targets with classes (A/B/C)

3. **Multi-Attribute Damage System**
   - Targets have multiple damage attributes
   - Weapons deal damage to specific attributes
   - All attributes must be depleted to neutralize

4. **Comprehensive Logging**
   - Episode logging with full state history
   - JSON-based episode files
   - Viewer for post-hoc analysis

5. **Policy Framework**
   - Random policy (ZK-compliant baseline)
   - Oracle policies (MaxDamage, MinTTK with full knowledge)
   - Collaborative Filtering policies (centralized & decentralized)
   - Extensible policy interface

6. **Decentralized Learning**
   - Per-agent private latent vectors (agent_lv, target_lv, other_agents_lv)
   - SGD-based matrix factorization with ε-greedy exploration
   - Learning state logging per episode for post-analysis
   - Scenario-based folder organization for learning states

---

## 📈 Code Quality Metrics

| Metric | Value |
|--------|-------|
| **Functions/Methods** | 450+ |
| **Dataclasses** | 13 |
| **Test Functions** | 150+ |
| **Documentation Files** | 10 (.md in docs/) |
| **Type Hints** | Extensive (TypedDict, dataclass) |

---

## 🔄 Development Activity

- **Total Commits:** 55+
- **Development Period:** ~7 weeks
- **Methodology:** Baby Steps (small, atomic, validated increments)

### Recent Commits

```
d9146f8 fixing training path tab
7d6a575 log reorganized
184eaed research proposal
e925741 doc update
08d9c65 + prefix changes for episode logs
c29c8ff + different mappings support
11dda5f + training path
8afcf42 learning path is now part of the episode json
d1ab507 learning path details #1
cc584aa fixing the reward mechanism
```

---

*Report generated by Cascade AI*
