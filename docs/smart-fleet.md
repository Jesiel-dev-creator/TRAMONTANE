# Smart Fleet Guide

How Tramontane's fleet intelligence saves you money and improves quality.

## Why model routing matters

Without routing, you pick one model for everything. That means overpaying
for simple tasks or underperforming on complex ones.

| Task | Without routing | With routing | Savings |
|------|----------------|--------------|---------|
| Classify a lead | mistral-large (EUR 6/1M) | ministral-3b (EUR 0.04/1M) | 150x |
| Write CSS | mistral-large (EUR 6/1M) | devstral-small (EUR 0.30/1M) | 20x |
| Deep analysis | ministral-3b (EUR 0.04/1M) | mistral-small-4+high (EUR 0.60/1M) | Better quality |

## Reasoning effort

Mistral Small 4 supports configurable thinking depth:

```python
# Fast: classification, extraction, simple Q&A
Agent(model="mistral-small-4", reasoning_effort="none")

# Balanced: most tasks
Agent(model="mistral-small-4", reasoning_effort="medium")

# Deep: complex analysis, planning, architecture
Agent(model="mistral-small-4", reasoning_effort="high")
```

## Progressive reasoning

Start cheap, escalate only when needed:

```python
agent = Agent(
    model="mistral-small-4",
    reasoning_strategy="progressive",
    validate_output=lambda r: "conclusion" in r.output.lower(),
)
```

Flow: try `none` -> if validation fails, try `medium` -> try `high`.
In practice, 70% of prompts succeed on `none`. Average cost drops 60%.

## Model cascading

Try affordable models first, escalate on failure:

```python
agent = Agent(
    model="devstral-small",
    cascade=["devstral-2", "mistral-large-3"],
    validate_output=lambda r: len(r.output) > 1000,
)
```

Cascade + progressive combine: all effort levels per model before moving to the next.

## Fleet profiles

Pre-configured strategies:

```python
from tramontane import FleetProfile

Agent(fleet_profile=FleetProfile.BUDGET)    # Cheapest models
Agent(fleet_profile=FleetProfile.BALANCED)  # Smart routing (default)
Agent(fleet_profile=FleetProfile.QUALITY)   # Best models, deep reasoning
Agent(fleet_profile=FleetProfile.UNIFIED)   # mistral-small-4 for everything
```

## Cost simulation

Estimate before spending:

```python
from tramontane import simulate_pipeline

sim = simulate_pipeline(agents, prompt, budget_eur=0.05)
print(f"Estimated: EUR {sim.total_estimated_cost_eur:.4f}")
print(f"Status: {sim.budget_status}")  # "within_budget" or "over_budget"
```

Or from CLI: `tramontane simulate pipeline.yaml --input "prompt"`

## Adaptive budget reallocation

Unspent budget flows to later agents:

```python
from tramontane import RunContext

ctx = RunContext(budget_eur=0.01, reallocation="adaptive")
await planner.run(input, run_context=ctx)   # Uses EUR 0.001 of 0.005
await builder.run(input, run_context=ctx)   # Gets EUR 0.005 + EUR 0.004 savings
```

## Self-learning router

The router improves from your production data:

```python
from tramontane import MistralRouter, FleetTelemetry

router = MistralRouter(telemetry=FleetTelemetry())
# After 50+ calls, suggestions are data-driven
# Check: tramontane telemetry stats
```

## FleetTuner

Auto-discover optimal config per agent:

```python
from tramontane import FleetTuner

tuner = FleetTuner()
result = await tuner.tune(
    agent=builder,
    test_prompts=["Build a bakery page", "Build a SaaS dashboard"],
    optimize_for="balanced",  # cost | quality | balanced | speed
)
builder = result.apply(builder)
print(f"Optimal: {result.optimal_model}, savings: {result.savings_vs_default}")
```
