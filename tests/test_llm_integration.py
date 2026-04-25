"""Integration smoke-test for the full LLM pipeline path."""
from sentinel.env import Sentinel_Env
from sentinel.training.prompt_builder import build_prompt, build_messages
from sentinel.training.action_parser import parse_llm_action, extract_think
from sentinel.training.llm_agent import make_grpo_reward_fn

env = Sentinel_Env(config_path="env_spec.yaml")
obs, _ = env.reset(seed=42)

# ── Test 1: prompt_builder ───────────────────────────────────────────────────
prompt = build_prompt(obs, agent_role="holmes", step_number=0)
messages = build_messages(obs, agent_role="forge", step_number=3)
print(f"[prompt_builder] prompt len={len(prompt)} chars | messages={len(messages)} turns")
print("  System snippet:", messages[0]["content"][:80].replace("\n", " "))
print("  User   snippet:", messages[1]["content"][:120].replace("\n", " "))

# ── Test 2: action_parser ────────────────────────────────────────────────────
test_cases = [
    (
        '<think>postgres is the root cause</think>\n```json\n'
        '{"agent":"holmes","category":"investigative","name":"QueryLogs",'
        '"params":{"service":"postgres-primary","time_range":[0,300]}}\n```',
        "code_block",
    ),
    (
        '<think>high saturation detected</think>\n'
        '{"agent":"forge","category":"remediation","name":"ScaleService",'
        '"params":{"service":"redis-cache","replicas":3}}',
        "raw_json",
    ),
    ("I think we should restart the service but I am not sure", "fallback"),
    (
        '{"agent":"holmes","category":"investigative","name":"querylogs","params":{}}',
        "case_fix",
    ),
    (
        '{"agent":"forge","category":"investigative","name":"ScaleService",'
        '"params":{"service":"api-gateway","replicas":2}}',
        "category_repair",
    ),
]

print("\n[action_parser] Test cases:")
all_ok = True
for raw, label in test_cases:
    action = parse_llm_action(raw, fallback_agent="holmes")
    think = extract_think(raw)
    think_repr = repr(think[:30]) if think else "(none)"
    print(
        f"  [{label:16s}] agent={action['agent']:6s} | "
        f"{action['name']:25s} | think={think_repr}"
    )
    # Validate required keys
    for key in ("agent", "category", "name", "params"):
        if key not in action:
            print(f"    ERROR: missing key '{key}'")
            all_ok = False

# ── Test 3: GRPO reward function ─────────────────────────────────────────────
print("\n[grpo_reward_fn] Testing reward computation:")
rf = make_grpo_reward_fn(env)
obs2, _ = env.reset(seed=7)

completions = [
    '{"agent":"forge","category":"remediation","name":"ScaleService",'
    '"params":{"service":"api-gateway","replicas":3}}',
    '{"agent":"holmes","category":"investigative","name":"QueryLogs",'
    '"params":{"service":"postgres-primary","time_range":[0,300]}}',
    "garbage that cannot be parsed",
]
rewards = rf(
    prompts=["<dummy prompt>"] * 3,
    completions=completions,
    agent_role="forge",
    obs=obs2,
)
print(f"  rewards = {[round(r, 3) for r in rewards]}")
assert len(rewards) == 3, "Expected 3 rewards"
assert all(isinstance(r, float) for r in rewards), "All rewards should be float"

# ── Test 4: sim-mode training (2 episodes) ───────────────────────────────────
print("\n[training_loop] 2-episode simulation run:")
from sentinel.training.pipeline import TrainingConfig, run_training_loop
cfg = TrainingConfig(
    agent="holmes",
    max_steps=2,
    batch_size=1,
    log_file="results/smoke_test_log.jsonl",
    checkpoint_dir="checkpoints/smoke_test",
)
metrics_list = run_training_loop(
    trainer=None,
    env=env,
    config=cfg,
    reward_fn=env.reward_function,
    llm_agent=None,
)
assert len(metrics_list) == 2
for m in metrics_list:
    print(
        f"  ep={m.episode} | total={m.total_reward:.3f} | "
        f"MTTR={m.mttr} | R1={m.r1:.2f} R2={m.r2:.2f} R3={m.r3:.2f} R4={m.r4:.2f}"
    )

print()
if all_ok:
    print("All integration smoke-tests PASSED.")
else:
    print("SOME TESTS FAILED — see above.")
    raise SystemExit(1)
