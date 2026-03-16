# Continuous Training Mode Requirements

## 1. Overview
The current ZK-MRTA environment operates strictly in **Episodic Mode**: a fixed set of targets is spawned at initialization, and the episode terminates when all targets are neutralized or a maximum step limit is reached. 

This document defines the requirements for a new **Continuous Training Mode**. In this mode, the environment simulates a single, endless engagement where neutralized targets are dynamically replaced, allowing policies to train over massive continuous step counts (e.g., 10,000+ steps) without regular domain resets.

## 2. Core Capabilities

### 2.1 Configuration
The `environment` section in `scenario.json` (and corresponding dataclasses) will be restructured to support toggleable modes.
A new `mode` attribute will be introduced under `environment`, defining the active training mode:
- `mode` (string): Either `"episodic"` or `"continuous"`.

Alongside `mode`, two new configuration objects will be added under `environment`, each holding mode-specific parameters:
- `episodic`: Contains configurations relevant only to episodic mode (e.g., `num_episodes`).
- `continuous`: Contains configurations relevant only to continuous mode, such as:
  - `logging_interval_steps` (int): Determines how frequently the environment takes a snapshot of the current state and flushes step history to disk (e.g., every 500 steps).

### 2.2 Asynchronous Target Respawning
- When `continuous_mode` is enabled, the environment's `step()` function will actively monitor for target neutralizations.
- The environment will never terminate due to "all targets neutralized"; it will only terminate when the absolute `max_steps` limit is reached.

#### 2.2.1 Target Identity Semantics
When a target is destroyed and subsequently respawned, the new target will **occupy the same internal state slot** (index) as the destroyed target. The environment will assign the new target a randomized position, health profile, and class type precisely dictated by the original distributions parsed at initialization (maintaining the total original ratio of classes). To policies and visualizers observing the environment, it will simply appear as if the target at index `i` sprang back to life at a new location with a new health profile.

#### 2.2.2 Respawn Timing
Target respawn calculations will execute at the **very end of the current engine tick** (`step()`). If a target drops to 0 HP during tick $T$, it is marked as inactive and its neutralization is recorded. Just before tick $T$ completes, the environment spawns the new target attributes into the inactive slot and flips the slot back to active. Thus, drones will observe the fully refreshed target at the very start of tick $T+1$.

### 2.3 Safe Rejection Sampling & Fallbacks
Because `ScenarioBuilder` relies on rejection sampling to find collision-free coordinates, the implementation must cap respawn attempts to maintain deterministic simulation speeds. 
- If the board is highly dense and a valid coordinate cannot be found within the maximum attempts (e.g., 1000), the environment **will simply log a warning that a target failed to find a valid coordinate, skip the respawn for that specific slot, and continue the engine tick**. The slot will remain inactive during tick $T+1$ and the environment will re-attempt to spawn a target into that slot at the end of the next tick.

## 3. Logging & Metrics Architecture

Continuous mode introduces severe memory and aggregation challenges. Logging 10,000+ per-tick observations into a single memory array will cause Out-of-Memory (OOM) errors. 

### 3.1 Chunked Memory Management
- `EpisodeLogger` will shift from a "dump on close" model to a "flushing" model.
- Every `logging_interval_steps`, the logger will save its internal `_steps` array to disk and then **clear the buffer**. This prevents memory bloating.

### 3.2 Dual Metric Aggregation
Inside `main_zk_mrta.py`, metrics (such as `targets_neutralized`, `total_ammo_used`, etc.) must be tracked in two parallel structures:
1. **Cumulative Metrics**: A running total spanning the entire continuous run (e.g., Step 0 to Step 10,000).
2. **Interval Metrics (Windowed)**: A local total specific to the current snapshot window (e.g., Steps 501 to 1000).

### 3.3 RunManager Artifact Tracking
Because the board never clears, the concept of a "shortest step count to zero targets" no longer exists. 
- When in continuous mode, `RunManager.finalize_policy()` will bypass its standard "first / best / mid" episode sorting logic.
- Instead, it will simply track and catalog the sequential list of snapshot files generated during the continuous run.

## 4. Log File Structure

In standard **Episodic Mode**, the `RunManager` creates three subfolders per policy: `episodes/`, `learning_state/`, and `analysis/`. 

For **Continuous Mode**, the exact same three-folder structure is preserved to maintain downstream compatibility, but the contents bypass single-file generation to support rolling intervals.

**Folder & File Naming Convention:**
Instead of `episode_first_ep01.json` or `episode_best.json`, the chunks will be sequentially numbered based on the step count at the time of flushing, and placed in their respective folders:
- `episodes/`
  - `episode_continuous_step_00500.json`
  - `episode_continuous_step_01000.json`
- `learning_state/` (if the policy tracks internal structure, like CF)
  - `learning_state_step_00500.json`
  - `learning_state_step_01000.json`
- `analysis/`
  - `analysis_step_00500.json`
  - `analysis_step_01000.json`

## 5. Log Content & Model Structure

Continuous mode distributes data across two parallel artifacts per interval: the episode log (environment interactions) and the learning state log (model structure).

### 5.1 Episode Log Content (`episode_continuous_step_X.json`)
The JSON schema will remain structurally compatible with offline visualizers but will represent data windows:
- `config` & `scenario`: Captured identically to Episodic mode (representing the active configuration).
- `steps`: **Windowed Data**. This array will only contain the step-by-step actions and rewards for the specific logging interval (e.g., exactly 500 entries).
- `summary`: **Split Payload**. Will be expanded to contain both `interval_metrics` (performance during this exact chunk, like ammo used or targets killed) and `cumulative_metrics` (overall performance since step 0).

### 5.2 Learning State Content (`learning_state_step_X.json`)
Policies that maintain complex internal structures (such as Matrix Factorization CF policies) will save their structures at the exact same intervals as the episode logs. The file naming will mirror the step count (e.g., `learning_state_step_00500.json`).
- **Model Structure**: The schema will continue to track the `version`, `num_agents`, and `num_targets`.
- **Pre/Post Boundaries**: The `pre_episode` block will reflect the internal structure (such as latent vectors) exactly as they were at the *start* of the interval window. The `post_episode` block will reflect the structure after the policy has trained up to the *end* of the interval window.

## 6. Compatibility Constraints
- **Strict Fallback:** The default `environment.mode` will be `"episodic"`. Existing unit tests, scripts, and visualizers that rely on episodic boundaries must not break.
- **Offline Replay:** Visualizers (e.g., Matrix Factorization replay) should be able to load successive chunk files natively if they wish to animate the entire continuous run.
