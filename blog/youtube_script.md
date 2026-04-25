# SENTINEL — YouTube Demo Script
# Duration: ~90 seconds (non-technical judges)
# Format: Screen recording of Colab notebook + voiceover

---

## [0:00–0:08] HOOK

**[VISUAL: Red alert dashboard, cascading error messages]**

> "It's 2 AM. Your payment service just went down.
> Five other services are failing. Your on-call engineer is asleep.
> What if AI could fix it — automatically?"

---

## [0:08–0:20] THE PROBLEM

**[VISUAL: Diagram of microservices with cascading red arrows]**

> "Modern cloud systems are a web of interconnected services.
> When one fails, the problem spreads — fast.
> Finding the root cause and fixing it before customers notice
> is one of the hardest problems in software engineering."

---

## [0:20–0:40] THE SOLUTION: SENTINEL

**[VISUAL: Colab notebook, env.reset() being called, render() output]**

> "SENTINEL is a reinforcement learning environment where five specialized AI agents
> collaborate to detect, diagnose, and fix cloud outages — automatically.
>
> Here's what it looks like in action.
> We inject a memory leak into the payment service.
> The cascade engine spreads the failure to five downstream systems.
> Now watch the agents work."

**[VISUAL: Step-by-step agent actions scrolling in the notebook]**

> "Holmes queries the logs. Argus spots the anomaly. Forge restarts the affected service.
> The blast radius shrinks. Hermes rolls back the bad deployment.
> Oracle closes the incident — 23 steps, well under the SLA."

---

## [0:40–1:05] HOW IT'S TRAINED

**[VISUAL: Training curve PNG — total reward increasing over episodes]**

> "We train this using GRPO — a reinforcement learning algorithm
> applied to Meta's Llama 3 8B model.
>
> The reward signal has four components:
> — Did the agent find the RIGHT root cause?
> — How FAST did it fix it?
> — How FULLY did the system recover?
> — How well did it CONTAIN the blast radius?
>
> After training, the agent goes from scoring 0 on root cause accuracy
> to identifying the correct failure 62% of the time on easy scenarios —
> rising to human-expert level with more training."

---

## [1:05–1:25] WHAT MAKES IT DIFFERENT

**[VISUAL: Split screen — incident_library.yaml + observability layer]**

> "Unlike simple chatbot environments, SENTINEL has:
> — 18 realistic failure scenarios with red herrings
> — Partial observability — two services are black boxes
> — Strict agent role constraints — just like real teams
> — An Oracle agent that generates NEW training scenarios
>   based on where the team is weakest
>
> This is not a toy. This is production-realistic RL."

---

## [1:25–1:40] CALL TO ACTION

**[VISUAL: GitHub repo, Colab notebook badge, HuggingFace post link]**

> "SENTINEL is fully open source, runs on a free Colab GPU,
> and ships with a Docker environment for local development.
>
> Try the Colab notebook — it runs top to bottom in under 5 minutes.
> Check the HuggingFace blog for the full technical writeup.
>
> We built SENTINEL because production reliability matters.
> And now, your AI team is always on call."

---

## [1:40–1:50] OUTRO

**[VISUAL: SENTINEL logo, repo link]**

> "SENTINEL — Multi-Agent RL for Cloud Incident Response.
> Meta PyTorch OpenEnv Hackathon 2026."

---

## PRODUCTION NOTES

- **Total runtime**: ~90 seconds
- **Screen recording**: Colab notebook running `sentinel_colab_demo.ipynb`
- **Key visual moments**:
  - Cell 3: `env.render()` output showing incident state
  - Cell 5: Agent step loop with printed actions
  - Cell 7: Training curve plot (`results/training_curves.png`)
  - Cell 8: Before/after evaluation table
- **Background music**: Lo-fi, neutral (no copyright issues)
- **Text overlays**: Reward formula at 0:50, GitHub link at 1:35
