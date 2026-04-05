# Planning Output Manifest: Continuous Mode Removal

**Plan ID:** PLAN-2026-04-05-001  
**Origin:** From-Analyzer  
**Created:** 2026-04-05  
**Workflow:** Planning → Execution

---

## Goal

Remove all continuous mode support from the TabulaDrone codebase, keeping only episodic mode functionality. This includes eliminating the mode toggle, continuous-specific configuration objects, conditional branching logic, flush intervals, chunked logging, and continuous-specific metrics.

---

## Implementation Scope

### Execution Starting Point
Step 1: Remove continuous mode from config JSON

### Affected Files/Modules
1. `config/scenario.json`
2. `tabula_drone/config/config_loader.py`
3. `main_zk_mrta.py`
4. `tabula_drone/utils/metrics_manager.py`
5. `tabula_drone/envs/drone_engage_latent_mrta.py`
6. `tabula_drone/logging/environment_logger.py`
7. `tests/test_metrics_manager.py`
8. `docs/continuous_training_requirements.md`

### Acceptance Criteria
- All continuous mode code removed from codebase
- Episodic mode functionality unchanged and working
- All episodic mode tests pass
- Configuration parses successfully for episodic scenarios
- Main runner executes successfully for episodic mode
- No references to continuous mode remain in active code

---

## Approved Step Sequence

Execute these steps **in order**, validating each before proceeding to the next:

### Configuration Layer (Steps 1-5)

**Step 1: Remove continuous mode from config JSON**
- **File:** `config/scenario.json`
- **Action:** Remove `"mode": "episodic"` line (line 20) and entire `"continuous"` object (lines 24-26)
- **Validation:** JSON parses successfully; `jq . config/scenario.json` succeeds
- **Rationale:** Start with data layer; independent change

**Step 2: Remove ContinuousModeConfig dataclass**
- **File:** `tabula_drone/config/config_loader.py`
- **Action:** Remove `ContinuousModeConfig` dataclass (lines 44-48)
- **Validation:** File imports successfully; no syntax errors
- **Rationale:** Remove unused dataclass before removing its usage

**Step 3: Remove continuous field from EnvironmentConfig**
- **File:** `tabula_drone/config/config_loader.py`
- **Action:** Remove `continuous: Optional[ContinuousModeConfig] = None` from `EnvironmentConfig` (line 58)
- **Validation:** File imports successfully; dataclass instantiates
- **Rationale:** Remove field reference to deleted dataclass

**Step 4: Remove mode field from EnvironmentConfig**
- **File:** `tabula_drone/config/config_loader.py`
- **Action:** Remove `mode: str` from `EnvironmentConfig` (line 54)
- **Validation:** File imports successfully; dataclass instantiates
- **Rationale:** Remove mode toggle field

**Step 5: Simplify _parse_environment_config to remove mode validation**
- **File:** `tabula_drone/config/config_loader.py`
- **Action:** Remove mode validation and continuous parsing logic (lines 229-256); keep only episodic parsing
- **Validation:** Config loads successfully for episodic scenario; `load_config("config/scenario.json")` succeeds
- **Rationale:** Remove mode-dependent parsing logic

### Main Runner Layer (Steps 6-14)

**Step 6: Remove mode-based episode count logic from main runner**
- **File:** `main_zk_mrta.py`
- **Action:** Replace lines 599-603 with `num_episodes = config.environment.episodic.num_episodes`
- **Validation:** Script runs without errors; episode count set correctly
- **Rationale:** Simplify episode count determination

**Step 7: Remove flush interval configuration from main runner**
- **File:** `main_zk_mrta.py`
- **Action:** Remove flush interval logic (lines 688-691) and `flush_interval` parameter from `run_episode()` call (line 718)
- **Validation:** Script runs without errors; no flush interval passed
- **Rationale:** Remove continuous mode flush configuration

**Step 8: Remove configure_continuous_flush call from main runner**
- **File:** `main_zk_mrta.py`
- **Action:** Remove `configure_continuous_flush()` call (lines 693-708)
- **Validation:** Script runs without errors; logger not configured for continuous mode
- **Rationale:** Stop calling continuous flush configuration

**Step 9: Remove periodic flush logic from run_episode**
- **File:** `main_zk_mrta.py`
- **Action:** Remove periodic flush logic (lines 292-295) including `flush_interval` parameter
- **Validation:** Episode runs without errors; no periodic flushing occurs
- **Rationale:** Remove continuous mode flush execution

**Step 10: Remove mode-specific progress reporting from main runner**
- **File:** `main_zk_mrta.py`
- **Action:** Replace lines 749-765 with only episodic branch (lines 757-765)
- **Validation:** Progress reporting displays correctly for episodic mode
- **Rationale:** Remove continuous mode reporting

**Step 11: Remove mode-specific analysis file naming from main runner**
- **File:** `main_zk_mrta.py`
- **Action:** Replace lines 811-820 with only episodic branch (lines 816-820)
- **Validation:** Analysis files use episodic naming pattern
- **Rationale:** Remove continuous mode file naming

**Step 12: Remove mode parameter from EnvironmentLogger constructor call**
- **File:** `main_zk_mrta.py`
- **Action:** Remove `mode=config.environment.mode` from `EnvironmentLogger()` call (line 624)
- **Validation:** Logger instantiates successfully without mode parameter
- **Rationale:** Prepare for logger simplification

**Step 13: Remove mode parameter from MetricsManager constructor call**
- **File:** `main_zk_mrta.py`
- **Action:** Remove `mode` argument from `MetricsManager()` call (line 667)
- **Validation:** MetricsManager instantiates successfully without mode parameter
- **Rationale:** Prepare for metrics simplification

**Step 14: Remove mode parameter from environment constructor call**
- **File:** `main_zk_mrta.py`
- **Action:** Remove `mode=config.environment.mode` from `DroneEngageLatentMRTA()` call (line 648)
- **Validation:** Environment instantiates successfully without mode parameter
- **Rationale:** Prepare for environment simplification

### Metrics Layer (Steps 15-20)

**Step 15: Remove throughput and coordination fields from EpisodeMetrics**
- **File:** `tabula_drone/utils/metrics_manager.py`
- **Action:** Remove `throughput`, `coordination_score`, `coordination_str` field declarations (lines 39-41)
- **Validation:** Dataclass instantiates successfully
- **Rationale:** Remove continuous-specific metric fields

**Step 16: Remove mode field from EpisodeMetrics**
- **File:** `tabula_drone/utils/metrics_manager.py`
- **Action:** Remove `mode: str` field (line 24)
- **Validation:** Dataclass instantiates successfully
- **Rationale:** Remove mode field

**Step 17: Remove continuous mode calculations from EpisodeMetrics.__post_init__**
- **File:** `tabula_drone/utils/metrics_manager.py`
- **Action:** Remove lines 54-63 (continuous mode calculations); keep only episodic calculations (lines 51-53)
- **Validation:** Metrics calculate correctly for episodic mode
- **Rationale:** Remove mode-specific calculation logic

**Step 18: Remove mode parameter from MetricsManager.__init__**
- **File:** `tabula_drone/utils/metrics_manager.py`
- **Action:** Remove `mode` parameter and validation (lines 98-101); remove `self.mode` assignment
- **Validation:** MetricsManager instantiates without mode parameter
- **Rationale:** Remove mode awareness from manager

**Step 19: Simplify representative episode selection in MetricsManager**
- **File:** `tabula_drone/utils/metrics_manager.py`
- **Action:** Replace lines 155-159 with `representative_episode = min(normalized, key=lambda item: item.steps)`
- **Validation:** Representative episode selected correctly
- **Rationale:** Always use min steps selection

**Step 20: Remove mode field from PolicyRunSummary**
- **File:** `tabula_drone/utils/metrics_manager.py`
- **Action:** Remove `mode: str` field (line 79)
- **Validation:** Dataclass instantiates successfully
- **Rationale:** Remove mode field from summary

### Environment Layer (Steps 21-22)

**Step 21: Remove mode parameter from environment constructor**
- **File:** `tabula_drone/envs/drone_engage_latent_mrta.py`
- **Action:** Remove `mode: str = "episodic"` parameter (line 62) and `self.mode` assignment (line 80)
- **Validation:** Environment instantiates without mode parameter
- **Rationale:** Remove mode awareness from environment

**Step 22: Remove mode parameter from EpisodeMetrics creation in environment**
- **File:** `tabula_drone/envs/drone_engage_latent_mrta.py`
- **Action:** Remove `mode=self.mode` from `EpisodeMetrics()` call (line 344)
- **Validation:** Metrics created successfully without mode
- **Rationale:** Align with simplified metrics

### Logging Layer (Steps 23-29)

**Step 23: Remove mode parameter from EnvironmentLogger.__init__**
- **File:** `tabula_drone/logging/environment_logger.py`
- **Action:** Remove `mode: str = "episodic"` parameter (line 29) and `self.mode` assignment (line 34)
- **Validation:** Logger instantiates without mode parameter
- **Rationale:** Remove mode awareness from logger

**Step 24: Remove continuous flush instance variables from EnvironmentLogger**
- **File:** `tabula_drone/logging/environment_logger.py`
- **Action:** Remove continuous flush instance variables (lines 54-58)
- **Validation:** Logger instantiates successfully
- **Rationale:** Remove unused instance variables

**Step 25: Remove configure_continuous_flush method from EnvironmentLogger**
- **File:** `tabula_drone/logging/environment_logger.py`
- **Action:** Remove `configure_continuous_flush()` method (lines 205-223)
- **Validation:** File imports successfully; no syntax errors
- **Rationale:** Remove unused method

**Step 26: Remove flush_episode method from EnvironmentLogger**
- **File:** `tabula_drone/logging/environment_logger.py`
- **Action:** Remove `flush_episode()` method (lines 275-294)
- **Validation:** File imports successfully; no syntax errors
- **Rationale:** Remove unused method

**Step 27: Remove handle_flush method from EnvironmentLogger**
- **File:** `tabula_drone/logging/environment_logger.py`
- **Action:** Remove `handle_flush()` method (lines 307-329)
- **Validation:** File imports successfully; no syntax errors
- **Rationale:** Remove unused method

**Step 28: Remove _clear_continuous_flush_context method from EnvironmentLogger**
- **File:** `tabula_drone/logging/environment_logger.py`
- **Action:** Remove `_clear_continuous_flush_context()` method (lines 462-467)
- **Validation:** File imports successfully; no syntax errors
- **Rationale:** Remove unused method

**Step 29: Simplify save_policy_episodes to remove continuous mode branch**
- **File:** `tabula_drone/logging/environment_logger.py`
- **Action:** Remove continuous mode branch (lines 380-385) from `save_policy_episodes()`
- **Validation:** Episodes save correctly for episodic mode
- **Rationale:** Remove mode-specific save logic

### Test Layer (Steps 30-31)

**Step 30: Remove mode parameter from build_metrics helper in tests**
- **File:** `tests/test_metrics_manager.py`
- **Action:** Remove `mode: str = "episodic"` parameter from `build_metrics()` (line 10)
- **Validation:** Tests run successfully
- **Rationale:** Align test helper with simplified metrics

**Step 31: Remove continuous mode test cases**
- **File:** `tests/test_metrics_manager.py`
- **Action:** Remove `test_calc_episode_metrics_continuous_formulas()` (lines 57-66) and `test_calc_total_episodes_metrics_continuous_representative_is_last()` (lines 85-94)
- **Validation:** Remaining tests pass
- **Rationale:** Remove tests for removed functionality

### Documentation Layer (Step 32)

**Step 32: Archive continuous mode documentation**
- **File:** `docs/continuous_training_requirements.md`
- **Action:** Move to `docs/deprecated/continuous_training_requirements.md`
- **Validation:** File moved successfully; docs directory clean
- **Rationale:** Preserve historical context

---

## Constraints & Context

### Reviewer Notes/Constraints
- Follow caller-before-callee dependency order (main runner updates before infrastructure removal)
- Each step must be atomic and independently verifiable
- Maintain buildable state at every step
- No behavioral changes to episodic mode functionality
- Follow minimal-diff principle (pure removals only)

### Testing Touchpoints
- `tests/test_metrics_manager.py` - Verify episodic metrics calculations
- `tests/test_environment_logger.py` - Verify episodic logging
- `tests/test_env_diagnostics.py` - Verify environment diagnostics
- `tests/test_runner_diagnostics_integration.py` - Verify end-to-end episodic flow
- **Manual:** Run `main_zk_mrta.py` with episodic config

### Side-Effects
- **Config schema change:** Existing configs with `"mode": "continuous"` will fail
- **Metrics API change:** `EpisodeMetrics` no longer has `mode`, `throughput`, `coordination_score`, `coordination_str` fields
- **Logger API change:** `EnvironmentLogger` no longer accepts `mode` parameter or provides flush methods
- **Environment API change:** `DroneEngageLatentMRTA` no longer accepts `mode` parameter
- **Log structure:** Continuous mode logs won't be generated; episodic naming always used

### Planner Assumption Ledger (final)
- Documentation handling: Deferred to execution (Step 32 archives to deprecated/)
- Config simplification: Deferred to future work (keep nested episodic.num_episodes for now)
- Test coverage: Validated in Step 31 (only continuous mode tests removed)

---

## Execution Instructions

When starting the Execution Workflow:

1. Import this manifest as the approved plan
2. Begin with Step 1 and proceed sequentially
3. For each step:
   - Read the current file state
   - Quote the exact code to be removed/modified (Context Anchor)
   - Make the minimal change specified
   - Run the validation check
   - Record the outcome before proceeding
4. Do not skip steps or combine steps
5. If a step fails validation, stop and report the issue

---

**End of Planning Output Manifest**
