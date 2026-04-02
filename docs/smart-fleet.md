# Smart Fleet Guide

How Tramontane's fleet intelligence saves money and improves quality.

## Why Routing Matters

| Task | Without routing | With routing | Savings |
|------|----------------|--------------|---------|
| Classify a lead | mistral-large EUR 6/1M | ministral-3b EUR 0.04/1M | 150x |
| Write CSS | mistral-large EUR 6/1M | devstral-small EUR 0.30/1M | 20x |
| Deep analysis | ministral-3b EUR 0.04/1M | mistral-small-4+high EUR 0.60/1M | Better quality |

## The Fleet (13 Models)

| Model | Tier | Best For | EUR/1M in/out | Reasoning | Vision |
|-------|------|----------|---------------|-----------|--------|
| ministral-3b | 0 | Classification, triage | 0.04/0.04 | | |
| ministral-7b | 1 | Bulk, extraction, tools | 0.10/0.10 | | |
| voxtral-mini | 1 | Transcription | 0.04/0.04 | | |
| voxtral-tts | 1 | Text-to-speech | 0.016/char | | |
| mistral-small | 2 | General, multilingual | 0.10/0.30 | | |
| mistral-small-4 | 2 | General+reasoning+vision | 0.15/0.60 | Yes | Yes |
| devstral-small | 2 | Code generation | 0.10/0.30 | | |
| magistral-small | 3 | Reasoning, planning | 0.50/1.50 | | |
| magistral-medium | 3 | Deep reasoning | 2.00/5.00 | | |
| devstral-2 | 4 | Complex SWE, monorepo | 0.50/1.50 | | |
| pixtral-large | 4 | Vision, multimodal, OCR | 2.00/6.00 | | |
| mistral-large | 4 | Frontier synthesis | 2.00/6.00 | | |
| mistral-large-3 | 4 | Frontier (Apache 2.0) | 2.00/6.00 | | |

## Router Decision Logic

```
has_vision?           -> pixtral-large
task=code?
  complexity >= 4     -> devstral-2
  else                -> devstral-small
task=classification?  -> ministral-3b
task=voice?           -> voxtral-mini
needs_reasoning?
  complexity >= 4     -> magistral-medium
  else                -> magistral-small
task=research+complex -> mistral-large-3
default               -> mistral-small-4 (with auto effort)
```

## Reasoning Effort

Mistral Small 4 supports configurable thinking depth:

```python
Agent(model="mistral-small-4", reasoning_effort="none")    # Fast, cheap
Agent(model="mistral-small-4", reasoning_effort="medium")  # Balanced
Agent(model="mistral-small-4", reasoning_effort="high")    # Deep thinking
```

The router auto-sets effort from task complexity when `reasoning_effort=None`.

## Progressive Reasoning

Start cheap, escalate only when validation fails:

```python
agent = Agent(
    model="mistral-small-4",
    reasoning_strategy="progressive",
    validate_output=lambda r: "conclusion" in r.output.lower(),
)
# Tries none -> medium -> high, stops at first success
# ~70% succeed on "none" -> 60% average cost reduction
```

## Model Cascading

Try affordable models first:

```python
agent = Agent(
    model="devstral-small",
    cascade=["devstral-2", "mistral-large-3"],
    validate_output=lambda r: len(r.output) > 1000,
)
```

Cascade + progressive combine: all effort levels per model before cascading.

## Fleet Profiles

```python
from tramontane import FleetProfile

Agent(fleet_profile=FleetProfile.BUDGET)    # Cheapest: ministral-3b + effort=none
Agent(fleet_profile=FleetProfile.BALANCED)  # Smart routing (default)
Agent(fleet_profile=FleetProfile.QUALITY)   # Best: devstral-2 for code, mistral-large-3 for research
Agent(fleet_profile=FleetProfile.UNIFIED)   # mistral-small-4 for everything
```

## Cost Simulation

Estimate before spending:

```python
from tramontane import simulate_pipeline

sim = simulate_pipeline(agents, prompt, budget_eur=0.05)
print(f"Estimated: EUR {sim.total_estimated_cost_eur:.4f}")
print(f"Status: {sim.budget_status}")
for a in sim.agents:
    print(f"  {a.role}: {a.model_predicted} EUR {a.estimated_cost_eur:.4f}")
```

CLI: `tramontane simulate pipeline.yaml --input "prompt"`

## Adaptive Budget Reallocation

Unspent budget flows to later agents:

```python
from tramontane import RunContext

ctx = RunContext(budget_eur=0.01, reallocation="adaptive")
await planner.run(input, run_context=ctx)   # Uses EUR 0.001 of 0.005
await builder.run(input, run_context=ctx)   # Gets EUR 0.005 + EUR 0.004 savings
```

## Self-Learning Router

```python
from tramontane import MistralRouter, FleetTelemetry

router = MistralRouter(telemetry=FleetTelemetry())
# Day 1: Routes by hand-crafted rules
# After 50+ outcomes: routes by YOUR production data
# Check: tramontane telemetry stats
```

## FleetTuner

Auto-discover optimal config per agent:

```python
from tramontane import FleetTuner

tuner = FleetTuner(models_to_test=["mistral-small-4", "devstral-small", "devstral-2"])
result = await tuner.tune(
    agent=builder,
    test_prompts=["Build a bakery page", "Build a SaaS dashboard", "Build a blog"],
    optimize_for="balanced",  # cost | quality | balanced | speed
)
optimized = result.apply(builder)
print(f"Optimal: {result.optimal_model}")
print(f"Savings: {result.savings_vs_default}")
print(f"Tuning cost: EUR {result.total_tuning_cost_eur:.4f}")
```

## RunContext

Shared cost tracking across agent chains:

```python
from tramontane import RunContext

ctx = RunContext(budget_eur=0.25)
await planner.run(input, run_context=ctx)
await designer.run(input, run_context=ctx)
await builder.run(input, run_context=ctx)
print(f"Total: EUR {ctx.spent_eur:.4f}")
print(f"Remaining: EUR {ctx.remaining_eur:.4f}")
print(f"Per agent: {ctx.agent_costs}")
```
