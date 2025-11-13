## Environment Spec – `DroneEngageSingleTarget-v0` (Gymnasium)

### 1. Scenario Overview

* **Name:** `DroneEngageSingleTarget-v0`
* **Framework:** Implemented as a custom `gymnasium.Env`
* **Role in project:**
  Step 0 of the MVP – a minimal but meaningful environment where a single static drone, with limited resources, engages a single static target in a 2D world. This step validates:

  * the Gymnasium interface (`reset`, `step`, `observation_space`, `action_space`),
  * basic world dynamics (ammo, HP, damage),
  * and the logging structure that will later generalize to multiple targets and multiple drones.

---

### 2. World & Entities

#### 2.1 World (global state)

* `world_size`: `{width, height}` ∈ ℝ⁺² – bounds of the 2D area (for coordinates and future scaling).
* `time_step` ∈ ℕ – current step index (starts at 0).
* `max_steps` ∈ ℕ – maximum allowed steps in the episode.
* `scenario_id` – optional string identifier for the scenario configuration.
* `seed` – random seed for reproducibility.

The world contains exactly:

* one **drone** object, and
* one **target** object.

#### 2.2 Drone (single agent)

* `id` – e.g., `"drone_1"`.
* `position` = `(x_d, y_d)` ∈ ℝ² – fixed for the entire episode.
* `ammo` ∈ ℕ – current number of shots remaining.
* `ammo_max` ∈ ℕ – maximum ammo at episode start.
* `damage_per_shot` ∈ ℝ⁺ – damage inflicted on the target when firing.

#### 2.3 Target (single target, but with attributes we’ll reuse later)

* `id` – e.g., `"target_1"`.
* `position` = `(x_t, y_t)` ∈ ℝ² – fixed.
* `class_type` – categorical label (e.g., `"A"`, `"B"`, `"C"`) representing a simple type that maps to initial HP.
* `zone_id` – simple zone label (e.g., `"zone_1"`) to keep the notion of regions, for later multi-target worlds.
* `hp_initial` ∈ ℝ⁺ – initial hit points derived from `class_type`.
* `hp_current` ∈ ℝ⁺ – current hit points.
* `is_active` ∈ {True, False} – active iff `hp_current > 0`.

---

### 3. Gymnasium Interface

This environment **conforms to Gymnasium**:

* `observation_space`: `Box` with shape `(4,)` (float32).
* `action_space`: `Discrete(2)`.

#### 3.1 Observation (agent’s view)

At each `reset()` and `step()`, the environment returns a fully observable 4D vector:

1. `ammo_normalized` = `ammo / ammo_max` ∈ [0, 1]
2. `hp_normalized`   = `hp_current / hp_initial` ∈ [0, 1]
3. `distance`        = Euclidean distance between drone and target
   `distance = sqrt((x_d - x_t)² + (y_d - y_t)²)` ∈ ℝ₊
4. `time_progress`   = `time_step / max_steps` ∈ [0, 1]

Shape: `(4,)`.

This keeps the obs minimal, but already introduces:

* resource status,
* target status,
* spatial structure,
* and temporal context.

#### 3.2 Action Space

* `action_space = Discrete(2)`
* Meaning:

  * `0` → **Idle**: drone does nothing.
  * `1` → **Fire**: drone attempts to fire one shot at the target.

---

### 4. Episode Initialization (`reset`)

On `reset()`:

* `time_step` ← 0
* `ammo`      ← `ammo_max` (env parameter)
* `class_type`, `zone_id`, `hp_initial`, and positions are set according to the chosen scenario/config (or simple defaults).
* `hp_current` ← `hp_initial`
* `is_active`  ← True

The initial observation vector is computed as in Section 3.1.

---

### 5. Transition Dynamics (`step`)

Given current state and `action ∈ {0,1}`:

1. **Action handling**

   * If `action == 0` (Idle):

     * No change to ammo or HP.
   * If `action == 1` (Fire):

     * If `ammo > 0` and `is_active`:

       * `ammo ← ammo - 1`
       * `hp_current ← hp_current - damage_per_shot`
       * If `hp_current <= 0`:

         * `hp_current ← 0`
         * `is_active ← False`
     * If `ammo == 0` before firing, the action has no effect.

2. **Time progression**

   * `time_step ← time_step + 1`

3. **Observation, reward, done flags**

   * New observation computed (Section 3.1).
   * Reward and termination logic as below.

---

### 6. Termination, Truncation, and Reward

#### 6.1 Termination (`terminated`)

`terminated = True` if:

1. **Target neutralized:**

   * `is_active = False` (i.e., `hp_current <= 0`).

2. **No ammo with active target:**

   * `ammo == 0` and `is_active = True`.

#### 6.2 Truncation (`truncated`)

`truncated = True` if:

* `time_step >= max_steps`.

(If both occur on the same step, convention: prioritize `terminated=True` when the target is neutralized exactly at the limit.)

#### 6.3 Reward Function

Per step:

* If the target becomes neutralized **at this step**:

  * `reward = +1.0`
* Otherwise:

  * `reward = 0.0`

This is intentionally simple: the goal is just “neutralize the target at some point before resources/time run out”.

---

### 7. Info Dictionary (for logging / analysis)

`info` returned by `step()` includes:

* `step_index` – current `time_step` after the transition.
* `done_reason` – only set at episode end:

  * `"target_neutralized"`
  * `"no_ammo"`
  * `"max_steps"`
* `scenario_id` – the scenario used for this episode.
* (Optional snapshot fields for debugging):

  * `ammo`
  * `hp_current`
  * `class_type`
  * `zone_id`
