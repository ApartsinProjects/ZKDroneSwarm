
## ZK-MRTA World & Random Policy – High-Level Specification (Draft)

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
  Unique string identifier for the target.

* `position: Tuple[float, float]`
  2D coordinates `(x, y)` in world space.

* `class_type: str`
  Discrete class label (e.g., `"A"`, `"B"`, `"C"`) that determines initial HP and possibly semantic category (e.g., “light”, “medium”, “heavy”).

* `zone_id: str`
  Identifier for spatial/logical zone (e.g., `"zone_1"`). Used for grouping targets and later for zone-level metrics or constraints.

* `hp_initial: float`
  Initial hit points of the target, determined by its `class_type`.
  **Invariant**: `hp_initial > 0`.

* `hp_current: float`
  Current remaining hit points.
  **Invariant**: `0 ≤ hp_current ≤ hp_initial`.

* `is_active: bool`
  Indicates whether the target is still “alive” in the scenario.
  **Invariant**:

  * `is_active == True  ⇔ hp_current > 0`
  * `is_active == False ⇔ hp_current ≤ 0`

Targets are static; positions do not change within an episode.

---

#### 2.2 Drone Attributes (`DroneState`)

Each drone is represented as:

* `id: str`
  Unique string identifier for the drone.

* `position: Tuple[float, float]`
  Static 2D coordinates `(x, y)` in world space. Drones do not move in this MVP.

* `ammo: int`
  Current ammunition count.
  **Invariant**: `0 ≤ ammo ≤ ammo_max`.

* `ammo_max: int`
  Maximum ammunition capacity.

* `damage_per_shot: float`
  Damage applied to a target on each successful shot by this drone.

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
  Identifier of the scenario configuration (e.g., number and classes of targets, number of drones, ammo settings). Used for reproducibility and analysis.

* `seed: Optional[int]`
  Random seed to ensure reproducible initialization and stochastic behavior (especially important for the Random policy).

The world contains collections of `DroneState` and `TargetState` instances and enforces the termination conditions (e.g., all targets neutralized or `max_steps` reached).

---

### 3. ZK-MRTA Problem Assumptions

The environment must enforce the following **Zero-Knowledge Multi-Robot Task Allocation (ZK-MRTA)** assumptions:

1. **No prior knowledge of task attributes**

   * Drones do **not** receive explicit information about target class types, HP values, or any semantic labels *before* interacting with the world.
   * Any knowledge about target difficulty or class must emerge only through observed outcomes (e.g., whether a target is neutralized after being shot).

2. **No knowledge of own capabilities**

   * Drones are not given explicit parameters such as their own `damage_per_shot` or detailed performance models.
   * They may observe simple outcomes (e.g., a shot was fired, ammo decreased, target changed state), but they are not told “your damage is 20 HP per shot” as static prior knowledge.

3. **No knowledge of other agents’ capabilities**

   * Drones are not informed about the damage, ammo, or policies of other drones.
   * Any inference about others comes only from indirect world-state changes (e.g., a target’s HP drops even though this drone did not fire).

4. **No communication between agents**

   * Drones cannot directly exchange messages, signals, or synchronized plans.
   * The environment must not provide any explicit “who did what” attribution; world updates are observable, but not which drone caused them.

5. **Observation constraints**

   * Drones:

     * **Can observe outcomes of their own actions** (e.g., “I fired at target X; now hp_current(X) changed / is_active changed / my ammo decreased”).
     * **Can only indirectly observe outcomes of others’ actions** via global world state changes (e.g., target HP reduced without this drone firing) but without attribution.
   * The observation model must be designed so that:

     * The world state is visible at the level needed for evaluation (e.g., which targets remain active),
     * But does not break the zero-knowledge assumptions by exposing pre-encoded knowledge about capabilities or performing centralized planning.

6. **Performance metrics independent of strategy**

   * Metrics such as total utility, mission completion time, number of neutralized targets, overkill, etc., are defined and computed **by the environment**, not by any specific policy.
   * This allows systematic comparison of different strategies (Random, Greedy, CF, etc.) on the same world and metrics without altering the world definition itself.

---

### 4. Action Model (Random Policy Context)

For this baseline, we assume a simple per-step action model:

* **Per Drone Action Space (conceptual)**

  * At each time step, an active drone with `ammo > 0` may:

    * `Engage(target_id)` – attempt to shoot a specific active target; or
    * `NoOp` – do nothing (optional for MVP; may be omitted if not needed).
  * If `ammo == 0`, the drone is effectively constrained to `NoOp`.

* **World Update**

  * For each `Engage(target_id)` action:

    * `ammo` decrements by 1.
    * The target’s `hp_current` decreases by the drone’s `damage_per_shot`.
    * If `hp_current ≤ 0`, set `is_active = False`.
  * Multiple drones may fire at the same target in the same step; cumulative damage is applied.

The exact encoding (Gym/PettingZoo spaces) is out of scope here; this spec only fixes the conceptual requirements.

---

### 5. RANDOM Decision Policy – Requirements

For this baseline, each drone follows a **Random policy** consistent with the ZK-MRTA assumptions:

1. **Zero-knowledge compliance**

   * The policy **must not** use:

     * Target class labels (`class_type`),
     * Precise HP values (`hp_initial`, `hp_current`),
     * Any embedded prior information about damage, success probabilities, or other drones.
   * It may observe which targets are currently active vs. inactive (since that is a direct/indirect outcome of actions), but it treats all active targets symmetrically.

2. **Action selection rule**

   * At each time step, for each drone with `ammo > 0`:

     * Form the set `A` of **feasible actions** (e.g., all `Engage(target_id)` for targets with `is_active == True`; optionally include `NoOp` if defined).
     * Select an action from `A` using a **uniform random distribution** (or another simple, explicitly specified distribution that does not use any semantic task knowledge).
   * For `ammo == 0`, the drone always performs `NoOp` (or simply has no action).

3. **No memory-based strategy**

   * The Random policy **does not exploit memory** of past outcomes to bias action selection (e.g., “stick to targets that seemed easier”).
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

  * Number of targets neutralized (overall and per `class_type` / `zone_id`).
  * Number of steps until termination (task completion time).
  * Total shots fired, ammo consumed, residual ammo.
  * (Optional) Overkill-related metrics, e.g., total damage above minimum required to neutralize each target.

Metrics are computed by the environment and collected for analysis; they do not feed back into the Random policy.
