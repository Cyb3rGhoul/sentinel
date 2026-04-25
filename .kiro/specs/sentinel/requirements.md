# Requirements Document

## Introduction

SENTINEL is a production-grade, OpenEnv-compliant reinforcement learning training environment for multi-agent autonomous incident response. It simulates NexaStack — a fictional but architecturally realistic 30-service e-commerce microservices platform — where five specialized AI agents cooperate to detect, diagnose, and remediate production incidents without human orchestration. SENTINEL is designed for the Meta PyTorch OpenEnv × Scaler Grand Finale hackathon and implements the Hyperagent self-improvement principle from Meta FAIR (arXiv:2603.19461). The system uses a 4-dimensional, 100% programmatically verifiable RLVR reward function and trains HOLMES and FORGE agents using TRL GRPOTrainer with Unsloth.

---

## Glossary

- **SENTINEL**: The overall RL training environment and multi-agent system described in this document.
- **NexaStack**: The simulated 30-service e-commerce microservices platform that serves as the environment world.
- **OpenEnv**: The open environment specification (gym.Env-compatible HTTP API) that SENTINEL implements.
- **ARGUS**: The Observer agent responsible for monitoring all 30 NexaStack services and detecting anomalies.
- **HOLMES**: The Root Cause Detective agent that performs causal reasoning on the CDG and maintains a Hypothesis Tree.
- **FORGE**: The Solutions Architect agent that designs and validates remediations and estimates blast radius.
- **HERMES**: The Deployment Agent responsible for canary deployments, automated rollbacks, and SLA validation.
- **ORACLE**: The Self-Improvement agent that analyzes completed incident trajectories and generates harder scenarios.
- **CDG**: Causal Dependency Graph — a directed weighted graph of the 30 NexaStack services and their dependencies.
- **NexaStackWorldState**: The complete world state object containing ServiceMetrics for all 30 services, the CDG, and incident state.
- **ServiceMetrics**: Per-service telemetry including CPU, memory, latency, error_rate, saturation, and availability.
- **Incident**: A simulated production failure event with a root cause, cascade effects, signals, red herrings, and a resolution timeline.
- **Incident_Library**: The collection of incident scenario templates, organized by difficulty tier.
- **Cascade_Engine**: The component that propagates failures through the CDG using BFS up to depth 6.
- **RLVR**: Reinforcement Learning with Verifiable Rewards — reward signals derived entirely from environment state, with no LLM-as-judge.
- **MTTR**: Mean Time To Resolution — the elapsed simulation steps from incident injection to verified system recovery.
- **Blast_Radius**: The set of services whose health metrics are degraded as a result of a remediation or cascade action.
- **GRPO**: Group Relative Policy Optimization — the training algorithm used via TRL GRPOTrainer.
- **Unsloth**: A training acceleration library providing 4x speed and 70% memory reduction for LoRA fine-tuning.
- **LoRA**: Low-Rank Adaptation — the parameter-efficient fine-tuning method applied to Llama-3-8B-Instruct.
- **Hypothesis_Tree**: A tree data structure maintained by HOLMES representing candidate root causes with Bayesian confidence scores.
- **SLA**: Service Level Agreement — the availability and latency thresholds that NexaStack services must meet.
- **Canary_Deploy**: A staged deployment strategy where a new version is rolled out to a small traffic percentage before full promotion.
- **Partial_Observability**: The condition where ARGUS receives incomplete telemetry — 0–80% of logs may be missing and some services are black boxes.
- **Red_Herring**: A spurious alert or signal present in the observation that does not correspond to the true root cause.
- **Episode**: A single incident scenario from injection to CloseIncident action or step limit exhaustion.
- **Trajectory**: The complete sequence of observations, actions, and rewards recorded during an episode.
- **ChromaDB**: The vector database used by ORACLE to store and retrieve incident trajectories and capability gap embeddings.
- **GRPOTrainer**: The TRL trainer class implementing GRPO used to fine-tune HOLMES and FORGE.
- **Gradio_Dashboard**: The live web UI that visualizes NexaStack health, agent actions, and training metrics.
- **FastAPI_Server**: The HTTP server exposing the OpenEnv-compliant API for SENTINEL.

---

## Requirements

### Requirement 1: OpenEnv-Compliant Environment Interface

**User Story:** As an RL researcher, I want SENTINEL to expose a standard OpenEnv gym.Env-compatible interface, so that I can integrate it with any OpenEnv-compatible training framework without custom adapters.

#### Acceptance Criteria

1. THE Sentinel_Env SHALL implement the `gym.Env` interface with `reset()`, `step()`, `render()`, and `close()` methods.
2. WHEN `reset()` is called, THE Sentinel_Env SHALL return an initial observation and an info dictionary, and SHALL sample a new incident from the Incident_Library.
3. WHEN `step(action)` is called, THE Sentinel_Env SHALL return a tuple of (observation, reward, terminated, truncated, info) conforming to the Gymnasium API.
4. THE Sentinel_Env SHALL expose `observation_space` and `action_space` attributes as valid Gymnasium Space objects.
5. THE FastAPI_Server SHALL expose HTTP endpoints `/reset`, `/step`, `/render`, and `/close` that proxy to the corresponding Sentinel_Env methods.
6. WHEN the FastAPI_Server receives a `/step` request with a malformed action payload, THE FastAPI_Server SHALL return an HTTP 422 response with a descriptive validation error.
7. THE Sentinel_Env SHALL be configurable via an `env_spec.yaml` file that specifies NexaStack topology, incident difficulty distribution, and reward weights.

---

### Requirement 2: NexaStack World State Simulation

**User Story:** As an RL researcher, I want a realistic simulated microservices environment, so that agents learn transferable incident response strategies applicable to real production systems.

#### Acceptance Criteria

1. THE NexaStackWorldState SHALL maintain ServiceMetrics (CPU, memory, latency, error_rate, saturation, availability) for all 30 NexaStack services at every simulation step.
2. THE NexaStackWorldState SHALL organize the 30 services across four layers: Frontend (3 services), Application (12 services), Data (8 services), and Infrastructure (7 services).
3. THE CDG SHALL represent service dependencies as a directed weighted graph where edge weights encode dependency strength between 0.0 and 1.0.
4. WHEN a service's health metric crosses a degradation threshold, THE NexaStackWorldState SHALL update the service's availability field to reflect the degraded state.
5. THE NexaStackWorldState SHALL expose a snapshot method that returns the current state as a serializable dictionary suitable for inclusion in an observation.
6. WHEN `reset()` is called on the Sentinel_Env, THE NexaStackWorldState SHALL restore all 30 services to their baseline healthy metric values.

---

### Requirement 3: Partial Observability and Noise

**User Story:** As an RL researcher, I want the environment to include partial observability, missing logs, and red herring alerts, so that agents must reason under uncertainty rather than exploit perfect information.

#### Acceptance Criteria

1. WHEN ARGUS constructs an observation, THE Observability_Layer SHALL randomly suppress between 0% and 80% of available log entries before including them in the observation.
2. THE Observability_Layer SHALL designate a configurable subset of NexaStack services as black boxes, for which no internal metrics are exposed in the observation.
3. WHEN an incident is active, THE Observability_Layer SHALL inject between 1 and 3 Red_Herring alerts into the active_alerts field of the observation.
4. THE Observability_Layer SHALL include the causal_graph_snapshot as a 30×30 adjacency matrix in every observation, with black-box service rows zeroed out.
5. WHEN the log suppression ratio for an episode is sampled, THE Observability_Layer SHALL sample it uniformly from the range [0.0, 0.8] and hold it constant for the duration of that episode.

---

### Requirement 4: Cascade Engine

**User Story:** As an RL researcher, I want failures to propagate realistically through the service dependency graph, so that agents must manage cascading incidents rather than isolated single-service failures.

#### Acceptance Criteria

1. WHEN a failure is injected into a root service, THE Cascade_Engine SHALL propagate the failure to dependent services using breadth-first search up to a maximum depth of 6.
2. THE Cascade_Engine SHALL apply a severity decay factor of 0.7 per hop depth, so that a failure at depth d has severity equal to initial_severity × (0.7^d).
3. THE Cascade_Engine SHALL support the following failure types: memory_leak, connection_pool_exhaustion, cpu_spike, bad_deployment, cache_miss_storm, and network_partition.
4. WHEN a remediation action resolves the root cause service, THE Cascade_Engine SHALL begin propagating recovery signals through the same dependency paths used during failure propagation.
5. WHEN a cascade reaches a service at depth 6, THE Cascade_Engine SHALL not propagate the failure further regardless of that service's downstream dependencies.
6. THE Cascade_Engine SHALL record the set of affected services and their degradation severities in the incident state for use by the reward function.

---

### Requirement 5: Incident Library and Scenario Management

**User Story:** As an RL researcher, I want a structured library of incident scenarios across difficulty tiers, so that agents can be trained progressively from simple to complex incidents.

#### Acceptance Criteria

1. THE Incident_Library SHALL contain at least 10 incident scenario templates organized into three difficulty tiers: Easy (E1, E2), Medium (M1, M2), and Hard (H1, H2), with additional scenarios generated by ORACLE.
2. EACH incident template in the Incident_Library SHALL specify: root_cause service, failure type, ground truth signals, Red_Herring signals, cascade_risk level, missing_log_ratio, and expected steps_to_resolve range.
3. THE Incident_Library SHALL be stored in a human-readable `incident_library.yaml` file that can be extended without modifying source code.
4. WHEN `reset()` is called, THE Incident_Generator SHALL sample an incident template from the Incident_Library according to the difficulty distribution specified in `env_spec.yaml`.
5. WHEN ORACLE generates a new incident template, THE Incident_Generator SHALL validate the template against the incident schema before adding it to the Incident_Library.
6. THE Incident_Library SHALL include the following named scenarios: E1 (Memory Leak in Cart Service), E2 (Failed Deployment Rollout), M1 (DB Connection Pool Exhaustion Cascading), M2 (Thundering Herd on Cache Miss), H1 (Race Condition Heisenbug), and H2 (Multi-Region Network Partition with Clock Skew).

---

### Requirement 6: Observation Space

**User Story:** As an RL researcher, I want a rich, structured observation that captures the full incident context, so that agents have sufficient information to reason about root causes and remediations.

#### Acceptance Criteria

1. THE Sentinel_Env SHALL construct each observation as a dictionary containing: metrics_snapshot, causal_graph_snapshot, active_alerts, recent_logs, active_traces, incident_context, and sla_state fields.
2. THE metrics_snapshot field SHALL contain ServiceMetrics for all 30 services, with black-box service metrics replaced by null values.
3. THE causal_graph_snapshot field SHALL be a 30×30 float matrix representing the CDG adjacency weights at the current simulation step.
4. THE active_alerts field SHALL contain all currently firing alerts including Red_Herring alerts, with no labeling to distinguish true from spurious alerts.
5. THE incident_context field SHALL contain the current incident timeline, the list of active hypotheses from HOLMES, the list of attempted remediations, and the current Blast_Radius.
6. THE sla_state field SHALL contain the current SLA compliance status for each of the 30 services, expressed as a boolean per service.
7. WHEN an observation is serialized for the FastAPI_Server response, THE Sentinel_Env SHALL encode it as a JSON-compatible dictionary with no non-serializable Python objects.

---

### Requirement 7: Action Space

**User Story:** As an RL researcher, I want a structured action space covering investigation, remediation, deployment, and meta actions, so that agents can express the full range of SRE behaviors.

#### Acceptance Criteria

1. THE Sentinel_Env SHALL define an action space containing four categories: investigative actions, remediation actions, deployment actions, and meta actions.
2. THE investigative actions SHALL include: QueryLogs(service, time_range), QueryTrace(trace_id), QueryMetrics(service, metric_name, time_range), and FormHypothesis(service, failure_type, confidence).
3. THE remediation actions SHALL include: RestartService(service), ScaleService(service, replicas), ModifyConfig(service, key, value), RollbackDeployment(service, version), DrainTraffic(service), and ModifyRateLimit(service, limit_rps).
4. THE deployment actions SHALL include: CanaryDeploy(service, version, traffic_percent), FullDeploy(service, version), and Rollback(service).
5. THE meta actions SHALL include: GenerateNewScenario(difficulty, target_gap), EscalateToHuman(reason), and CloseIncident(resolution_summary).
6. WHEN an agent submits a RestartService action for a service that is currently healthy, THE Sentinel_Env SHALL apply a blast_radius penalty to the reward for that step.
7. WHEN an agent submits a CloseIncident action, THE Sentinel_Env SHALL set terminated=True and compute the final episode reward.

---

### Requirement 8: ARGUS — Observer Agent

**User Story:** As an RL researcher, I want ARGUS to continuously monitor all 30 services and surface anomalies, so that other agents receive a pre-filtered alert stream rather than raw metrics.

#### Acceptance Criteria

1. THE ARGUS_Agent SHALL poll the NexaStackWorldState at every simulation step and produce an anomaly report for inclusion in the observation.
2. WHEN a service metric deviates from its baseline by more than a configurable threshold, THE ARGUS_Agent SHALL add an alert entry to the active_alerts field.
3. THE ARGUS_Agent SHALL operate under Partial_Observability constraints and SHALL NOT have access to black-box service internals.
4. WHEN ARGUS produces an anomaly report, THE ARGUS_Agent SHALL include a confidence score between 0.0 and 1.0 for each alert based on the strength of the metric deviation.
5. THE ARGUS_Agent SHALL not perform any remediation actions and SHALL only emit investigative and meta actions.

---

### Requirement 9: HOLMES — Root Cause Detective Agent

**User Story:** As an RL researcher, I want HOLMES to perform causal reasoning on the CDG and maintain a Hypothesis Tree, so that the system can identify root causes in complex cascading incidents.

#### Acceptance Criteria

1. THE HOLMES_Agent SHALL maintain a Hypothesis_Tree where each node represents a candidate root cause service and failure type, annotated with a Bayesian confidence score.
2. WHEN HOLMES receives a new observation, THE HOLMES_Agent SHALL update the confidence scores of all active hypotheses using the new metric and alert evidence.
3. WHEN HOLMES issues a FormHypothesis action, THE HOLMES_Agent SHALL add a new node to the Hypothesis_Tree with an initial confidence score derived from CDG edge weights and active alerts.
4. THE HOLMES_Agent SHALL be fine-tuned using GRPOTrainer on the Llama-3-8B-Instruct base model with LoRA (r=16, alpha=32).
5. WHEN the Hypothesis_Tree contains a hypothesis with confidence score above 0.85, THE HOLMES_Agent SHALL emit that hypothesis as the primary root cause candidate in the incident_context.
6. THE HOLMES_Agent SHALL only emit investigative actions and FormHypothesis actions; it SHALL NOT emit remediation or deployment actions.

---

### Requirement 10: FORGE — Solutions Architect Agent

**User Story:** As an RL researcher, I want FORGE to design and validate remediations before execution, so that the system minimizes blast radius when resolving incidents.

#### Acceptance Criteria

1. THE FORGE_Agent SHALL receive the current Hypothesis_Tree from HOLMES and the NexaStackWorldState snapshot, and SHALL produce a ranked list of candidate remediation plans.
2. WHEN FORGE proposes a remediation plan, THE FORGE_Agent SHALL estimate the Blast_Radius of that plan by simulating its effect on the CDG before submitting the action.
3. THE FORGE_Agent SHALL select the remediation plan with the lowest estimated Blast_Radius among plans that address the top-ranked hypothesis.
4. THE FORGE_Agent SHALL be fine-tuned using GRPOTrainer on the Llama-3-8B-Instruct base model with LoRA (r=16, alpha=32).
5. WHEN FORGE estimates that a remediation plan will expand the Blast_Radius beyond the current affected service count, THE FORGE_Agent SHALL escalate to HERMES with a warning flag rather than submitting the action directly.
6. THE FORGE_Agent SHALL only emit remediation actions; it SHALL NOT emit investigative, deployment, or meta actions.

---

### Requirement 11: HERMES — Deployment Agent

**User Story:** As an RL researcher, I want HERMES to execute deployments safely using canary strategies and automated rollback, so that remediations are applied without causing additional outages.

#### Acceptance Criteria

1. THE HERMES_Agent SHALL execute all deployment actions received from FORGE using a canary strategy, starting at a configurable initial traffic percentage.
2. WHEN a CanaryDeploy action is submitted, THE HERMES_Agent SHALL monitor the canary service's error_rate and latency for a configurable observation window before promoting to full deployment.
3. WHEN the canary service's error_rate exceeds the pre-canary baseline during the observation window, THE HERMES_Agent SHALL automatically issue a Rollback action without waiting for agent input.
4. WHEN a deployment completes successfully, THE HERMES_Agent SHALL validate that all affected services have returned to SLA-compliant metric values before marking the remediation as complete.
5. THE HERMES_Agent SHALL record a post-mortem entry in the incident timeline for every deployment action, including the action taken, outcome, and duration in simulation steps.
6. THE HERMES_Agent SHALL only emit deployment actions and the CloseIncident meta action; it SHALL NOT emit investigative or remediation actions.

---

### Requirement 12: ORACLE — Self-Improvement Agent

**User Story:** As an RL researcher, I want ORACLE to analyze completed incident trajectories and generate harder scenarios targeting identified capability gaps, so that the agent population continuously improves without human curriculum design.

#### Acceptance Criteria

1. WHEN an episode terminates, THE ORACLE_Agent SHALL analyze the complete Trajectory and identify the action category where the agent population performed worst (highest error rate or lowest reward contribution).
2. THE ORACLE_Agent SHALL store completed Trajectories and capability gap embeddings in ChromaDB for retrieval during scenario generation.
3. WHEN ORACLE identifies a capability gap, THE ORACLE_Agent SHALL generate a new incident template that specifically exercises the identified gap, using the GenerateNewScenario action.
4. WHEN ORACLE generates a new incident template, THE ORACLE_Agent SHALL assign it a difficulty tier one level above the incident that exposed the gap, capped at Hard.
5. THE ORACLE_Agent SHALL submit generated incident templates to the Incident_Generator for schema validation before they are added to the Incident_Library.
6. THE ORACLE_Agent SHALL implement the Hyperagent self-improvement principle from Meta FAIR arXiv:2603.19461, where the agent population's capability gaps drive the generation of the next training curriculum.
7. WHEN the Incident_Library contains more than 50 ORACLE-generated templates, THE ORACLE_Agent SHALL retire the oldest templates with below-median training utility scores to maintain library quality.

---

### Requirement 13: RLVR Reward Function

**User Story:** As an RL researcher, I want a 4-dimensional programmatically verifiable reward function, so that agent training is not subject to LLM-as-judge bias or inconsistency.

#### Acceptance Criteria

1. THE Reward_Function SHALL compute the episode reward as a weighted sum of four components: R1 (Root Cause Accuracy, weight 0.35), R2 (MTTR, weight 0.30), R3 (System Recovery Quality, weight 0.25), and R4 (Blast Radius Minimization, weight 0.10).
2. THE Reward_Function SHALL compute R1 as 1.0 when the agent's identified root cause exactly matches the ground truth root cause service and failure type, 0.5 when only the root cause service matches, and 0.0 otherwise.
3. THE Reward_Function SHALL compute R2 as a value between 0.0 and 1.0 inversely proportional to the MTTR in simulation steps, with a bonus of 0.1 added when resolution occurs before the SLA breach threshold.
4. THE Reward_Function SHALL compute R3 as the fraction of the 30 services whose health metrics have returned to within 5% of their baseline values at episode termination.
5. THE Reward_Function SHALL compute R4 as 1.0 minus the ratio of services in the final Blast_Radius to the total number of services affected at peak incident severity.
6. THE Reward_Function SHALL apply a hard penalty of -1.0 when any remediation action causes the Blast_Radius to expand beyond its value at the time the action was submitted.
7. THE Reward_Function SHALL apply a hard penalty of -0.5 when the incident resolution time exceeds 2× the SLA breach threshold for any affected service.
8. THE Reward_Function SHALL derive all reward signals exclusively from NexaStackWorldState and incident state fields, with no calls to external language models or human evaluators.

---

### Requirement 14: GRPO Training Pipeline

**User Story:** As an ML engineer, I want a complete GRPO training pipeline using TRL and Unsloth, so that HOLMES and FORGE can be fine-tuned efficiently on consumer-grade hardware.

#### Acceptance Criteria

1. THE Training_Pipeline SHALL use TRL GRPOTrainer to fine-tune the Llama-3-8B-Instruct base model for both HOLMES and FORGE agents.
2. THE Training_Pipeline SHALL apply Unsloth optimizations to achieve at least 4× training throughput improvement and at least 70% GPU memory reduction compared to standard full-parameter fine-tuning.
3. THE Training_Pipeline SHALL configure LoRA with rank r=16 and alpha=32 for all fine-tuned model parameters.
4. THE Training_Pipeline SHALL use the RLVR Reward_Function as the sole reward signal for GRPO updates, with no auxiliary reward terms.
5. THE Training_Pipeline SHALL log training metrics (episode reward, MTTR, R1–R4 components, loss) to a persistent log file and to the Gradio_Dashboard at the end of each episode.
6. THE Training_Pipeline SHALL support resuming training from a checkpoint without restarting the episode counter or losing logged metrics.
7. THE Training_Pipeline SHALL include an evaluation pipeline that runs 10 held-out episodes per difficulty tier and reports mean and standard deviation of all reward components.

---

### Requirement 15: Gradio Live Dashboard

**User Story:** As a hackathon judge, I want a live Gradio dashboard that visualizes NexaStack health, agent actions, and training progress, so that I can evaluate the system's behavior in real time without reading logs.

#### Acceptance Criteria

1. THE Gradio_Dashboard SHALL display a real-time health status panel showing the current metric values for all 30 NexaStack services, color-coded green (healthy) to red (critical).
2. THE Gradio_Dashboard SHALL display an agent action feed showing the most recent 20 actions taken by each of the 5 agents, with timestamps and action parameters.
3. THE Gradio_Dashboard SHALL display a training progress chart showing episode reward, MTTR, and R1–R4 component scores over the last 50 episodes.
4. THE Gradio_Dashboard SHALL provide an "Inject Incident" control that allows a user to manually trigger any named incident from the Incident_Library.
5. WHEN an incident is active, THE Gradio_Dashboard SHALL highlight the affected services in the health panel and display the current Blast_Radius.
6. THE Gradio_Dashboard SHALL display the ORACLE capability gap analysis and the most recently generated incident template after each episode completes.
7. THE Gradio_Dashboard SHALL refresh its displayed state at a minimum rate of 1 update per 2 simulation steps without requiring a page reload.

---

### Requirement 16: Docker Deployment

**User Story:** As a hackathon judge, I want to run SENTINEL with a single Docker command, so that I can reproduce the demo environment without installing Python dependencies manually.

#### Acceptance Criteria

1. THE Dockerfile SHALL produce a self-contained image that includes all Python dependencies, model weights download instructions, and the FastAPI server.
2. THE docker-compose.yml SHALL define services for the SENTINEL environment server, the Gradio dashboard, and ChromaDB, with correct inter-service networking.
3. WHEN `docker-compose up` is executed in the project root, THE Docker_Deployment SHALL start all services and expose the Gradio dashboard on port 7860 and the FastAPI server on port 8000 within 120 seconds.
4. THE Dockerfile SHALL use Python 3.11 as the base image and SHALL pin all dependency versions in requirements.txt.
5. WHEN any Docker service exits with a non-zero code, THE docker-compose.yml SHALL configure that service to restart automatically up to 3 times before marking it as failed.

---

### Requirement 17: HuggingFace Spaces Deployment

**User Story:** As a hackathon participant, I want to deploy the Gradio demo to HuggingFace Spaces, so that judges can access the live demo without running Docker locally.

#### Acceptance Criteria

1. THE Gradio_Dashboard SHALL be deployable to HuggingFace Spaces using the Gradio SDK without modification to the demo source code.
2. THE demo/app.py SHALL include a `demo.launch()` call compatible with HuggingFace Spaces auto-detection.
3. WHEN deployed to HuggingFace Spaces, THE Gradio_Dashboard SHALL connect to a pre-seeded NexaStack simulation state and SHALL not require a live training run to display meaningful data.

---

### Requirement 18: Parser and Serialization Round-Trip

**User Story:** As an ML engineer, I want all environment state serialization to be lossless and round-trippable, so that checkpointing, replay, and API transport do not corrupt training data.

#### Acceptance Criteria

1. THE Sentinel_Env SHALL serialize NexaStackWorldState to JSON and deserialize it back to an equivalent NexaStackWorldState object.
2. FOR ALL valid NexaStackWorldState objects, serializing then deserializing SHALL produce an object with identical ServiceMetrics values, CDG edge weights, and incident state fields (round-trip property).
3. THE Sentinel_Env SHALL serialize Trajectory objects (sequences of observations, actions, and rewards) to a persistent format and deserialize them without data loss.
4. FOR ALL valid Trajectory objects, serializing then deserializing SHALL produce a Trajectory with identical observation dictionaries, action parameters, and reward values (round-trip property).
5. WHEN the Incident_Generator parses an incident template from `incident_library.yaml`, THE Incident_Generator SHALL validate that the parsed template round-trips back to an equivalent YAML representation.
