## ZK-MRTA World & Random Policy - High-Level Specification

> **Last Updated:** December 2024

### 1. Scope

This specification defines the minimal world model and behavioral constraints for a **Zero-Knowledge Multi-Robot Task Allocation (ZK-MRTA)** simulation with:

* Static drones engaging static targets in a 2D world.
* Agents operating under strict **zero-knowledge and no-communication** assumptions.
* A single baseline decision strategy: **RANDOM policy** (no learning, no prior knowledge).

The goal is to provide a reproducible environment in which the Random policy can be evaluated against ZK-MRTA performance metrics (e.g., task completion efficiency, utility) without baking any strategy-specific assumptions into the world model.

---

### 2. Entity Model

#### 2.1 Target Attributes (`TargetState`)

Each target is represented as:

* `id: str`
  Unique string identifier for the target (e.g., `"target_0"`).

* `position: Tuple[float, float]`
  Static 2D coordinates `(x, y)` in world space.

* `class_type: str`
  Discrete class label (e.g., `"A"`, `"B"`, `"C"`) that determines initial attribute values via configurable mapping.

* `attributes: AttributeProfile`
  Multi-attribute health profile containing named attributes (e.g., `{"armor": 100.0, "shields": 50.0}`).
  See Section 2.4 for details.

* `is_active: bool`
  Indicates whether the target is still "alive" in the scenario.
  **Invariant**:
  * `is_active == True  ⇔` at least one attribute > 0
  * `is_active == False ⇔` all attributes ≤ 0

**Backward Compatibility Properties:**
* `hp_current` → Sum of all current attribute values
* `hp_initial` → Sum of all initial attribute values

Targets are static; positions do not change within an episode.

---

#### 2.2 Drone Attributes (`DroneState`)

Each drone is represented as:

* `id: str`
  Unique string identifier for the drone (e.g., `"drone_0"`).

* `position: Tuple[float, float]`
  Static 2D coordinates `(x, y)` in world space. Drones do not move.

* `ammo_used: int`
  Running count of shots fired (starts at 0).
  **Note:** Drones have **unlimited ammo**. This field is a counter for metrics, not a constraint.

* `weapon_type: str`
  Weapon category (e.g., `"light"`, `"medium"`, `"heavy"`) that determines damage profile via configurable mapping.

* `damage_profile: Dict[str, float]`
  Damage per attribute per shot (e.g., `{"armor": 10.0, "shields": 5.0}`).
  Looked up from `weapon_damage_profile_mapping` based on `weapon_type`.

**Backward Compatibility Property:**
* `damage_per_shot` → Sum of all damage profile values

Drones are static shooters that choose targets but do not relocate.

---

#### 2.3 World Attributes (`WorldState`)

The world is represented as:

* `world_size: Tuple[float, float]`
  `(width, height)` defining the 2D bounds of the environment. All drone/target positions must lie within these bounds.

* `time_step: int`
  Current discrete time-step index within the episode.
  **Invariant**: `0 ≤ time_step ≤ max_steps`.

* `max_steps: int`
  Maximum allowed steps per episode. If `time_step == max_steps`, the episode terminates by truncation.

* `scenario_id: str`
  Identifier of the scenario configuration (e.g., number and classes of targets, number of drones). Used for reproducibility and analysis.

* `seed: Optional[int]`
  Random seed to ensure reproducible initialization and stochastic behavior (especially important for the Random policy).

The world contains collections of `DroneState` and `TargetState` instances and enforces the termination conditions (e.g., all targets neutralized or `max_steps` reached).

---

#### 2.4 Attribute Profile (`AttributeProfile`)

Multi-attribute health system for targets:

* `attributes: Dict[str, float]`
  Current values for each named attribute (mutable).
  Example: `{"armor": 75.0, "shields": 25.0}`

* `initial_values: Dict[str, float]`
  Original values at creation (immutable reference).

**Methods:**
* `apply_damage(damage_profile)` → Reduces each attribute by corresponding damage value (clamped at 0)
* `is_depleted()` → Returns `True` if ALL attributes are ≤ 0
* `get_total()` → Returns sum of all current attribute values

**Depletion Rule:** A target is considered neutralized when ALL attributes reach zero or below.

---

### 3. ZK-MRTA Problem Assumptions

The environment must enforce the following **Zero-Knowledge Multi-Robot Task Allocation (ZK-MRTA)** assumptions:

1. **No prior knowledge of task attributes**

   * Drones do **not** receive explicit information about target class types, HP values, or any semantic labels *before* interacting with the world.
   * Any knowledge about target difficulty or class must emerge only through observed outcomes (e.g., whether a target is neutralized after being shot).

2. **No knowledge of own capabilities**

   * Drones are not given explicit parameters such as their own `damage_profile` or detailed performance models.
   * They may observe simple outcomes (e.g., a shot was fired, target changed state), but they are not told "your damage is 20 HP per shot" as static prior knowledge.

3. **No knowledge of other agents' capabilities**

   * Drones are not informed about the damage or policies of other drones.
   * Any inference about others comes only from indirect world-state changes (e.g., a target becomes inactive even though this drone did not fire).

4. **No communication between agents**

   * Drones cannot directly exchange messages, signals, or synchronized plans.
   * The environment must not provide any explicit "who did what" attribution; world updates are observable, but not which drone caused them.

5. **Observation constraints**

   * Drones:

     * **Can observe outcomes of their own actions** (e.g., "I fired at target X; now is_active changed").
     * **Can only indirectly observe outcomes of others' actions** via global world state changes (e.g., target becomes inactive without this drone firing) but without attribution.
   * The observation model must be designed so that:

     * The world state is visible at the level needed for evaluation (e.g., which targets remain active),
     * But does not break the zero-knowledge assumptions by exposing pre-encoded knowledge about capabilities or performing centralized planning.

6. **Performance metrics independent of strategy**

   * Metrics such as total utility, mission completion time, number of neutralized targets, overkill, etc., are defined and computed **by the environment**, not by any specific policy.
   * This allows systematic comparison of different strategies (Random, Greedy, CF, etc.) on the same world and metrics without altering the world definition itself.

---

### 4. Action Model (Random Policy Context)

For this baseline, we assume a simple per-step action model:

* **Per Drone Action Space**

  * At each time step, a drone may:

    * `Engage(target_id)` – attempt to shoot a specific active target; or
    * `NoOp` – do nothing (optional, controlled by `allow_noop` parameter).
  * **Note:** Drones have unlimited ammo, so ammo constraints do not apply.

* **World Update (Sequential Processing)**

  Drones are processed **sequentially in random order** each step:

  1. Shuffle drone processing order randomly.
  2. For each drone in shuffled order:
     * If action == `NoOp`: skip.
     * If action == `Engage(target_id)`:
       * Increment `ammo_used` by 1.
       * If target `is_active`:
         * Apply `damage_profile` to target's `attributes`.
         * If `attributes.is_depleted()`: set `is_active = False`, award +1.0 reward to this drone.
       * Else: wasted shot (ammo counted, no damage/reward).

  **Note:** Sequential processing means only one drone can deliver the "killing blow" per target per step.

The exact encoding uses PettingZoo `ParallelEnv` with `Discrete(num_targets + 1)` action space per agent.

---

### 5. RANDOM Decision Policy - Requirements

For this baseline, each drone follows a **Random policy** consistent with the ZK-MRTA assumptions:

1. **Zero-knowledge compliance**

   * The policy **must not** use:

     * Target class labels (`class_type`),
     * Attribute values (current or initial),
     * Any embedded prior information about damage, success probabilities, or other drones.
   * It may observe which targets are currently active vs. inactive (since that is a direct/indirect outcome of actions), but it treats all active targets symmetrically.

2. **Action selection rule**

   * At each time step, for each drone:

     * Form the set `A` of **feasible actions** (all `Engage(target_id)` for targets with `is_active == True`; optionally include `NoOp` if `allow_noop=True`).
     * Select an action from `A` using a **uniform random distribution**.
   * **Note:** Drones have unlimited ammo, so ammo constraints do not apply.

3. **No memory-based strategy**

   * The Random policy **does not exploit memory** of past outcomes to bias action selection (e.g., "stick to targets that seemed easier").
   * Drones may maintain internal memory for logging or debugging, but this memory must not influence the Random decision rule.

4. **No coordination**

   * Drones independently sample their actions; there is no explicit or implicit coordination protocol.
   * The environment must not inject any coupling that would let drones systematically avoid or align on specific targets beyond what random independence produces.

5. **Reproducibility**

   * The Random policy must use the world `seed` (if provided) to ensure deterministic reproducibility of runs:

     * Same `scenario_id` + same `seed` ⇒ same sequences of random decisions and outcomes.

---

### 6. Episode Termination & Metrics (Baseline)

* **Termination conditions**:

  * All targets are inactive (`is_active == False` for all targets); or
  * `time_step` reaches `max_steps`.

* **Metrics to log (for Random policy evaluation)**:

  * `steps` — Number of steps until termination.
  * `targets_neutralized` — Count of targets with `is_active == False`.
  * `total_ammo_used` — Sum of all drones' `ammo_used`.
  * `total_overkill` — Sum of excess damage beyond neutralization.
  * `agent_rewards` — Cumulative reward per drone.
  * `done_reason` — `"all_targets_neutralized"` or `"max_steps"`.

Metrics are computed by the environment and collected for analysis; they do not feed back into the Random policy.

---

### 7. Configuration Requirements

The environment requires two configurable mappings (no hardcoded defaults):

* **`class_attribute_mapping: Dict[str, Dict[str, float]]`**
  Maps target class types to initial attribute values.
  Example:
  ```python
  {
      "A": {"armor": 100.0, "shields": 50.0},
      "B": {"armor": 150.0, "shields": 75.0},
      "C": {"armor": 200.0, "shields": 100.0},
  }
  ```

* **`weapon_damage_profile_mapping: Dict[str, Dict[str, float]]`**
  Maps weapon types to damage per attribute per shot.
  Example:
  ```python
  {
      "light": {"armor": 10.0, "shields": 5.0},
      "medium": {"armor": 25.0, "shields": 12.5},
      "heavy": {"armor": 50.0, "shields": 25.0},
  }
  ```

These mappings must be provided at environment initialization.

---

### 8. Reward Model

* **+1.0** to the drone that delivers the **killing blow** (neutralizes the target).
* Sequential processing ensures only one drone can neutralize a target per step.
* Shots at already-neutralized targets are wasted (ammo counted, no reward).
* No penalty for `NoOp` or missed shots.

---

### 9. Observation Model (ZK-Compliant)

Each drone observes a flat array of shape `(3 * num_targets,)`:

```
[target_0_x, target_0_y, target_0_active,
 target_1_x, target_1_y, target_1_active,
 ...]
```

**Exposed:**
* Target positions (x, y)
* Target active status (1.0 or 0.0)

**Hidden (ZK compliance):**
* Target attribute values (current or initial)
* Target class types
* Drone damage profiles
* Other drones' states or actions

All agents receive identical observations (global state visibility for positions/active status only).

---

*Specification updated to reflect current implementation. Last updated: December 2024.*
