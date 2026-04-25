# Implementation Plan: SENTINEL

## Overview

Implement SENTINEL bottom-up: core data models and world state first, then the cascade engine and observability layer, then the incident library, then the agent layer, then the reward function, then the FastAPI server and training pipeline, and finally the Gradio dashboard and deployment artifacts. Each phase wires into the previous one so there is no orphaned code.

## Tasks

- [x] 1. Project scaffold and core data models
  - Create the directory structure: `sentinel/`, `sentinel/api/`, `sentinel/agents/`, `sentinel/training/`, `demo/`, `tests/unit/`, `tests/property/`, `tests/integration/`
  - Write `sentinel/models.py` containing all Pydantic v2 / dataclass definitions: `ServiceMetrics`, `IncidentState`, `IncidentTemplate`, `Action` (and all sub-action types), `Observation`, `TrajectoryStep`, `Trajectory`, `RewardBreakdown`, `RewardWeights`, `HypothesisNode`, `HypothesisTree`, `Alert`, `LogEntry`, `Trace`, `IncidentContext`, `TimelineEntry`, `FailureType` enum
  - Write `sentinel/config.py` to load and validate `env_spec.yaml` with Pydantic; fall back to defaults when file is absent
  - Create `env_spec.yaml` with the configuration values from the design
  - _Requirements: 1.7, 2.1, 2.2, 6.1, 7.1_

- [x] 2. NexaStack world state and CDG
  - [x] 2.1 Implement `NexaStackWorldState` in `sentinel/world_state.py`
    - Hard-code the 30-service topology (4 layers) and baseline `ServiceMetrics`
    - Build the `nx.DiGraph` CDG with edge weights in [0.0, 1.0]
    - Implement `restore_baseline()`, `apply_degradation()`, `snapshot()`, `to_json()`, `from_json()`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 18.1, 18.2_

  - [x] 2.2 Write property test: world state always has exactly 30 services with valid metrics (Property 4)
    - **Property 4: World state always contains exactly 30 services with valid metrics**
    - **Validates: Requirements 2.1**

  - [x] 2.3 Write property test: CDG edge weights always in [0.0, 1.0] (Property 5)
    - **Property 5: CDG edge weights are always in [0.0, 1.0]**
    - **Validates: Requirements 2.3**

  - [x] 2.4 Write property test: metric threshold crossing updates availability (Property 6)
    - **Property 6: Metric threshold crossing updates availability**
    - **Validates: Requirements 2.4**

  - [x] 2.5 Write property test: NexaStackWorldState serialization round-trip (Property 20)
    - **Property 20: NexaStackWorldState serialization round-trip**
    - **Validates: Requirements 18.1, 18.2**

  - [x] 2.6 Write unit tests for `NexaStackWorldState`
    - Test `restore_baseline()` resets all 30 services
    - Test `apply_degradation()` clamps severity to 1.0
    - Test `snapshot()` returns a JSON-serializable dict
    - _Requirements: 2.4, 2.5, 2.6_

- [x] 3. Cascade Engine
  - [x] 3.1 Implement `Cascade_Engine` in `sentinel/cascade_engine.py`
    - BFS failure propagation up to depth 6 with severity decay 0.7 per hop
    - `propagate_failure()`, `propagate_recovery()`, `get_blast_radius()`
    - Support all 6 `FailureType` values; clamp accumulated severity to 1.0
    - Raise `CascadeError` when root service is not in CDG; handle cycles via visited set
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

  - [x] 3.2 Write property test: cascade respects BFS depth ≤ 6 with exponential severity decay (Property 11)
    - **Property 11: Cascade propagation respects BFS depth ≤ 6 with exponential severity decay**
    - **Validates: Requirements 4.1, 4.2**

  - [x] 3.3 Write property test: recovery propagates through same paths as failure (Property 12)
    - **Property 12: Recovery propagates through the same paths as failure**
    - **Validates: Requirements 4.4**

  - [x] 3.4 Write unit tests for `Cascade_Engine`
    - Test specific BFS traversal examples and depth-6 boundary
    - Test severity decay values at each depth level
    - Test `CascadeError` on unknown root service
    - _Requirements: 4.1, 4.2, 4.5_

- [x] 4. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Incident Library and Generator
  - [x] 5.1 Create `incident_library.yaml` with the 6 named scenarios (E1, E2, M1, M2, H1, H2) plus at least 4 additional templates to reach the minimum of 10
    - Each template must include all required fields per the `IncidentTemplate` schema
    - _Requirements: 5.1, 5.2, 5.3, 5.6_

  - [x] 5.2 Implement `Incident_Generator` in `sentinel/incident_generator.py`
    - `sample()` weighted by difficulty distribution from config
    - `validate_template()` against `IncidentTemplate` schema
    - `add_template()` for ORACLE-generated templates
    - Parse `incident_library.yaml` at init; raise `IncidentLibraryError` on missing/malformed file
    - _Requirements: 5.1, 5.3, 5.4, 5.5_

  - [x] 5.3 Write property test: every incident template contains all required fields (Property 13)
    - **Property 13: Every incident template contains all required fields**
    - **Validates: Requirements 5.2, 5.5**

  - [x] 5.4 Write property test: incident sampling matches configured difficulty distribution (Property 14)
    - **Property 14: Incident sampling matches the configured difficulty distribution**
    - **Validates: Requirements 5.4**

  - [x] 5.5 Write property test: incident template YAML round-trip (Property 22)
    - **Property 22: Incident template YAML round-trip**
    - **Validates: Requirements 18.5**

  - [x] 5.6 Write unit tests for `Incident_Generator`
    - Test schema validation rejects templates with missing fields
    - Test `IncidentLibraryError` on missing YAML file
    - _Requirements: 5.2, 5.5_

- [x] 6. Observability Layer
  - [x] 6.1 Implement `Observability_Layer` in `sentinel/observability.py`
    - `sample_episode_params()` — sample `log_suppression_ratio` from U[0.0, 0.8] and `red_herring_count` from {1, 2, 3}
    - `build_observation()` — apply log suppression, zero out black-box rows in CDG matrix, inject red herring alerts, assemble full `Observation` dict
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

  - [x] 6.2 Write property test: log suppression ratio in [0.0, 0.8] and constant within episode (Property 8)
    - **Property 8: Log suppression ratio is in [0.0, 0.8] and constant within an episode**
    - **Validates: Requirements 3.1, 3.5**

  - [x] 6.3 Write property test: active incidents always have 1–3 red herring alerts (Property 9)
    - **Property 9: Active incidents always have 1–3 red herring alerts**
    - **Validates: Requirements 3.3**

  - [x] 6.4 Write property test: CDG snapshot zeros out black-box service rows (Property 10)
    - **Property 10: CDG snapshot zeros out black-box service rows**
    - **Validates: Requirements 3.4**

  - [x] 6.5 Write property test: red herring alerts are not labeled in observations (Property 15)
    - **Property 15: Red herring alerts are not labeled in observations**
    - **Validates: Requirements 6.4**

  - [x] 6.6 Write unit tests for `Observability_Layer`
    - Test black-box service masking produces null metrics
    - Test red herring injection count boundaries (1 and 3)
    - _Requirements: 3.2, 3.3_

- [x] 7. Reward Function
  - [x] 7.1 Implement `Reward_Function` in `sentinel/reward.py`
    - `_r1_root_cause_accuracy()` — returns 0.0, 0.5, or 1.0
    - `_r2_mttr()` — inversely proportional to MTTR steps, +0.1 bonus for pre-SLA resolution
    - `_r3_recovery_quality()` — fraction of services within 5% of baseline
    - `_r4_blast_radius()` — 1 - (final_br / peak_br)
    - `compute_step_reward()` — apply blast radius expansion penalty (-1.0) and healthy-service restart penalty
    - `compute_episode_reward()` — weighted sum + late resolution penalty (-0.5); return `RewardBreakdown`
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8_

  - [x] 7.2 Write property test: episode reward equals weighted sum of components plus penalties (Property 17)
    - **Property 17: Episode reward equals the weighted sum of components plus penalties**
    - **Validates: Requirements 13.1, 13.2, 13.3, 13.4, 13.5**

  - [x] 7.3 Write property test: blast radius expansion triggers -1.0 penalty (Property 18)
    - **Property 18: Blast radius expansion triggers a -1.0 penalty**
    - **Validates: Requirements 13.6**

  - [x] 7.4 Write property test: late resolution triggers -0.5 penalty (Property 19)
    - **Property 19: Late resolution triggers a -0.5 penalty**
    - **Validates: Requirements 13.7**

  - [x] 7.5 Write property test: RestartService on healthy service incurs blast_radius penalty (Property 16)
    - **Property 16: RestartService on a healthy service incurs a blast_radius penalty**
    - **Validates: Requirements 7.6**

  - [x] 7.6 Write unit tests for `Reward_Function`
    - Test exact R1/R2/R3/R4 values for known inputs
    - Test penalty application for blast radius expansion and late resolution
    - _Requirements: 13.1–13.7_

- [x] 8. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Agent implementations
  - [x] 9.1 Implement `BaseAgent` ABC and `ARGUS` in `sentinel/agents/argus.py`
    - Poll `NexaStackWorldState` each step; emit alerts with confidence scores
    - Enforce investigative + meta actions only; respect black-box constraints
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [x] 9.2 Implement `HOLMES` in `sentinel/agents/holmes.py`
    - Maintain `HypothesisTree`; implement `update_confidences()` and `get_primary_candidate(threshold=0.85)`
    - Emit `FormHypothesis` and investigative actions only
    - Wire to GRPOTrainer interface (stub `act()` that calls the fine-tuned model when available)
    - _Requirements: 9.1, 9.2, 9.3, 9.5, 9.6_

  - [x] 9.3 Implement `FORGE` in `sentinel/agents/forge.py`
    - Receive `HypothesisTree` + world state snapshot; produce ranked remediation plans
    - Simulate CDG effect to estimate blast radius before submitting action
    - Escalate to HERMES with warning flag when blast radius would expand
    - Emit remediation actions only
    - _Requirements: 10.1, 10.2, 10.3, 10.5, 10.6_

  - [x] 9.4 Implement `HERMES` in `sentinel/agents/hermes.py`
    - Execute canary deployments; monitor error_rate during observation window
    - Auto-rollback when error_rate exceeds pre-canary baseline
    - Validate SLA compliance after successful deployment
    - Record post-mortem timeline entries
    - Emit deployment + `CloseIncident` actions only
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6_

  - [x] 9.5 Write unit tests for HERMES canary logic
    - Test rollback trigger when error_rate exceeds baseline during observation window
    - Test successful promotion path
    - _Requirements: 11.2, 11.3_

  - [x] 9.6 Implement `ORACLE` in `sentinel/agents/oracle.py`
    - Analyze trajectory to identify worst-performing action category
    - Store trajectories and capability gap embeddings in ChromaDB (fall back to in-memory on connection failure)
    - Generate new `IncidentTemplate` via `GenerateNewScenario`; submit to `Incident_Generator` for validation
    - Retire oldest below-median templates when library exceeds 50 ORACLE-generated entries
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7_

  - [x] 9.7 Write unit tests for ORACLE retirement logic
    - Test that templates are retired when count exceeds 50
    - Test that below-median utility templates are removed first
    - _Requirements: 12.7_

- [x] 10. Sentinel_Env — top-level environment
  - [x] 10.1 Implement `Sentinel_Env` in `sentinel/env.py`
    - Wire together `NexaStackWorldState`, `Observability_Layer`, `Incident_Generator`, `Reward_Function`, and all agents
    - Implement `reset()`, `step()`, `render()`, `close()`
    - Define `observation_space` and `action_space` as valid `gymnasium.spaces.Dict` objects
    - Raise `gymnasium.error.ResetNeeded` when `step()` is called before `reset()`
    - Return `reward=-0.1` with appropriate `info.error` for unknown service or role violation actions
    - Raise `IncidentLibraryError` at init when `incident_library.yaml` is missing/malformed
    - Fall back to default config with warning log when `env_spec.yaml` is absent
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.7, 7.6, 7.7_

  - [x] 10.2 Write property test: reset() returns a structurally valid observation (Property 1)
    - **Property 1: reset() returns a structurally valid observation**
    - **Validates: Requirements 1.2, 6.1**

  - [x] 10.3 Write property test: step() returns a valid Gymnasium 5-tuple (Property 2)
    - **Property 2: step() returns a valid Gymnasium 5-tuple**
    - **Validates: Requirements 1.3**

  - [x] 10.4 Write property test: reset() restores all services to baseline (Property 7)
    - **Property 7: reset() restores all services to baseline**
    - **Validates: Requirements 2.6**

- [x] 11. Trajectory serialization
  - [x] 11.1 Implement `to_json()` and `from_json()` on `Trajectory` in `sentinel/models.py`
    - Ensure all observation dicts, action parameters, and reward values survive the round-trip
    - _Requirements: 18.3, 18.4_

  - [x] 11.2 Write property test: Trajectory serialization round-trip (Property 21)
    - **Property 21: Trajectory serialization round-trip**
    - **Validates: Requirements 18.3, 18.4**

- [x] 12. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 13. FastAPI server
  - [x] 13.1 Implement `sentinel/api/server.py` with endpoints `/reset`, `/step`, `/render`, `/close`, `/health`
    - All request/response models as Pydantic v2 models
    - Return HTTP 503 when `Sentinel_Env` is not yet initialized
    - Return HTTP 500 with traceback in debug mode on unhandled exceptions
    - _Requirements: 1.5, 1.6_

  - [x] 13.2 Write property test: malformed action payloads return HTTP 422 (Property 3)
    - **Property 3: Malformed action payloads return HTTP 422**
    - **Validates: Requirements 1.6**

  - [x] 13.3 Write integration test: full episode flow via FastAPI
    - Test `/reset` → `/step` (multiple actions) → `/close` end-to-end (3 example episodes)
    - _Requirements: 1.5, 1.6_

- [x] 14. GRPO training pipeline
  - [x] 14.1 Implement `sentinel/training/pipeline.py`
    - `build_grpo_trainer()` for HOLMES and FORGE using `FastLanguageModel` + LoRA (r=16, alpha=32)
    - Load `unsloth/Meta-Llama-3-8B-Instruct` in 4-bit with `max_seq_length=4096`
    - Use `Reward_Function.compute_episode_reward` as the sole GRPO reward signal
    - Log episode reward, MTTR, R1–R4, and loss to a persistent log file after each episode
    - Support checkpoint save and resume without resetting episode counter
    - Handle CUDA OOM by halving batch size and retrying; log warning
    - Handle corrupted checkpoint by falling back to last valid checkpoint
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 14.6_

  - [x] 14.2 Implement evaluation pipeline in `sentinel/training/evaluate.py`
    - Run 10 held-out episodes per difficulty tier (30 total)
    - Report mean ± std of R1, R2, R3, R4, total reward, and MTTR
    - _Requirements: 14.7_

- [x] 15. Gradio dashboard
  - [x] 15.1 Implement `demo/app.py` with `build_dashboard(env)` and `demo.launch()`
    - 30-service color-coded health grid (green → red)
    - Agent action feed: last 20 actions per agent with timestamps
    - Training progress chart: episode reward + MTTR + R1–R4 over last 50 episodes
    - "Inject Incident" control for any named incident in the library
    - Highlight affected services and current blast radius when incident is active
    - ORACLE capability gap display after each episode
    - Refresh at ≥ 1 update per 2 simulation steps
    - Pre-seeded simulation state for HuggingFace Spaces (no live training required)
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 15.7, 17.1, 17.2, 17.3_

- [x] 16. Docker and deployment artifacts
  - [x] 16.1 Write `Dockerfile` using Python 3.11, pin all dependencies in `requirements.txt`, include model weights download instructions
    - _Requirements: 16.1, 16.4_

  - [x] 16.2 Write `docker-compose.yml` defining SENTINEL server (port 8000), Gradio dashboard (port 7860), and ChromaDB services with correct networking and restart policy (max 3 retries)
    - _Requirements: 16.2, 16.3, 16.5_

- [x] 17. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for traceability
- Property tests use `hypothesis` with `@settings(max_examples=100)` and the tag format `# Feature: sentinel, Property {N}: {property_text}`
- Checkpoints ensure incremental validation at each major phase boundary
