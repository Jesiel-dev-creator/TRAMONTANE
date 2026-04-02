# Quick Start

Get from zero to a working agent with memory and skills in 10 minutes.

## Install

```bash
pip install tramontane
export MISTRAL_API_KEY=your_key_here
```

## First Agent

```python
import asyncio
from tramontane import Agent, MistralRouter

agent = Agent(
    role="Analyst",
    goal="Analyze market trends",
    backstory="Senior market analyst with 10 years experience",
    model="auto",
    budget_eur=0.01,
)

async def main():
    router = MistralRouter()
    result = await agent.run("Analyze the EU AI market in 2026", router=router)
    print(f"Model: {result.model_used}")
    print(f"Cost:  EUR {result.cost_eur:.4f}")
    print(f"Tokens: {result.input_tokens} in / {result.output_tokens} out")
    print(result.output)

asyncio.run(main())
```

## First Agent with Memory

```python
from tramontane import Agent, TramontaneMemory

memory = TramontaneMemory(db_path="my_memory.db")

agent = Agent(
    role="Gerald",
    goal="Track client relationships",
    backstory="Business intelligence agent",
    tramontane_memory=memory,
    memory_tools=True,
    auto_extract_facts=True,
)

result = await agent.run("Acme Corp prefers React and has 50 employees in Paris")
# Agent auto-extracts: "Acme Corp prefers React", "Acme Corp has 50 employees"
# Next run: agent can recall_memory("What do we know about Acme?")
```

## First Skill

```python
from tramontane import Skill, SkillResult, SkillRegistry, track_skill

class GreetingSkill(Skill):
    name = "greeting"
    description = "Generate personalized greetings"
    triggers = ["greet", "hello", "welcome"]

    @track_skill
    async def execute(self, input_text, context=None):
        return SkillResult(output=f"Welcome! You said: {input_text}", success=True)

registry = SkillRegistry()
registry.register(GreetingSkill())
matches = registry.search("greet the new client")
result = await matches[0][0].execute("Alice from Acme Corp")
```

## First Pipeline (YAML)

Create `pipeline.yaml`:
```yaml
name: Market Report
budget_eur: 0.02
agents:
  researcher:
    role: Researcher
    goal: Research the topic
    backstory: Expert analyst
    model: mistral-small-4
    reasoning_effort: medium
  writer:
    role: Writer
    goal: Write a compelling report
    backstory: Senior writer
    model: devstral-small
    temperature: 0.7
flow: [researcher, writer]
```

Run it:
```bash
tramontane run pipeline.yaml --input "EU AI regulations 2026"
```

## CLI Walkthrough

```bash
tramontane models          # See all 13 models with pricing
tramontane doctor          # Check API key, connectivity, fleet
tramontane fleet           # Model stats from telemetry data
tramontane simulate pipeline.yaml --input "test"  # Cost estimate
tramontane knowledge ingest docs/   # Build RAG knowledge base
tramontane knowledge search "query" # Search knowledge base
tramontane telemetry stats          # Router learning metrics
```

## How the Router Works

When you set `model="auto"`, the router:
1. Classifies your prompt (code? reasoning? vision? bulk?)
2. Determines complexity (1-5)
3. Picks the optimal model from the 13-model fleet
4. Applies budget constraints (downgrades if needed, never below quality floor)
5. Sets reasoning effort based on complexity
6. Learns from outcomes via telemetry

## Cost Tracking

Every `AgentResult` includes real costs in EUR:
```python
result = await agent.run("prompt")
print(f"EUR {result.cost_eur:.6f}")  # Actual cost from API token counts
```

Shared budgets across agent chains:
```python
from tramontane import RunContext
ctx = RunContext(budget_eur=0.05, reallocation="adaptive")
await planner.run(input, run_context=ctx)
await builder.run(input, run_context=ctx)
print(f"Total: EUR {ctx.spent_eur:.4f}")
```

## Next Steps

- [Memory System](memory.md) — 3-tier memory deep dive
- [Skills System](skills.md) — typed, composable skills
- [Smart Fleet](smart-fleet.md) — model routing and optimization
- [Patterns](patterns.md) — real-world usage patterns
- [API Reference](api-reference.md) — full API documentation
