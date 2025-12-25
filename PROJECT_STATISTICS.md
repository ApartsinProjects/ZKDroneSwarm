# TabulaDrone Project Statistics Report

**Generated:** December 23, 2025  
**Project Started:** November 13, 2025  
**Last Commit:** December 21, 2025  

---

## 📊 Lines of Code Summary

| Category | Lines of Code | Files |
|----------|---------------|-------|
| **Total (Production + Tests)** | **7,693** | **37** |
| Core Package (`tabula_drone/`) | 2,609 | 17 |
| Viewer (`viewer/`) | 2,168 | 16 |
| Tests (`tests/`) | 2,594 | 6 |
| Main Script | 322 | 1 |

### Breakdown by Module

| Module | Lines | Description |
|--------|-------|-------------|
| `tabula_drone/envs/` | 641 | Environment implementations |
| `tabula_drone/scenarios/` | 800 | Scenario building & weapon assignment |
| `tabula_drone/config/` | 390 | Configuration loading |
| `tabula_drone/logging/` | 326 | Episode logging |
| `tabula_drone/policies/` | 306 | Policy implementations |
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

### Policy Classes (2)
- `RandomPolicy` - ZK-compliant random action selection
- `OracleTimeToKillPolicy` - Optimal oracle with full knowledge

### Environment Class (1)
- `DroneEngageZKMRTA` - Main PettingZoo ParallelEnv

### Viewer Components (8)
- `BaseComponent` - Abstract base for all UI components
- `MapPanel` - World visualization with drones/targets
- `TabContainer` - Tab navigation system
- `InfoPanel`, `ResultsPanel`, `SummaryPanel`, `EmptyPanel`

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
   - Oracle policy (optimal with full knowledge)
   - Extensible policy interface

---

## 📈 Code Quality Metrics

| Metric | Value |
|--------|-------|
| **Functions/Methods** | 364 |
| **Dataclasses** | 13 |
| **Test Functions** | 136 |
| **Documentation Files** | 4 (.md in specs/) |
| **Type Hints** | Extensive (TypedDict, dataclass) |

---

## 🔄 Development Activity

- **Total Commits:** 40+
- **Development Period:** ~6 weeks
- **Methodology:** Baby Steps (small, atomic, validated increments)

### Recent Commits

```
e2abbba + summary panel
257355b empty circle for destroyed target
84e960c bug fix - oracle policy
5e7ce1d line between the drone and the target
3af2faf play pause animation
2430322 +hp as title colors changes
a77da2f sort by timestamp (episode)
32bad4f show reward in a better way
37205bf display all drone rewards
575e5ba support multiple policies execution
```

---

*Report generated by Cascade AI*
