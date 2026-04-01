# Quick Start

Get from zero to a working agent in 5 minutes.

## Install

```bash
pip install tramontane
```

## Set your API key

```bash
export MISTRAL_API_KEY=your_key_here
```

## Your first agent (Python)

```python
import asyncio
from tramontane import Agent, MistralRouter

agent = Agent(
    role="Analyst",
    goal="Analyze market trends",
    backstory="Senior market analyst with 10 years experience",
    model="auto",        # Router picks the best model
    budget_eur=0.01,     # Hard cost ceiling
)

async def main():
    router = MistralRouter()
    result = await agent.run("Analyze the EU AI market in 2026", router=router)
    print(f"Model: {result.model_used}")
    print(f"Cost:  EUR {result.cost_eur:.4f}")
    print(result.output)

asyncio.run(main())
```

## Your first pipeline (YAML)

Create `pipeline.yaml`:

```yaml
name: Market Report
version: "1.0"
budget_eur: 0.02

agents:
  researcher:
    role: Researcher
    goal: Research the topic thoroughly
    backstory: Expert research analyst
    model: mistral-small-4
    reasoning_effort: medium

  writer:
    role: Writer
    goal: Write a compelling report
    backstory: Senior technical writer
    model: devstral-small
    temperature: 0.7

flow:
  - researcher
  - writer
```

Run it:

```bash
tramontane run pipeline.yaml --input "EU AI regulations 2026"
```

## What just happened?

1. The **router** classified your prompt as a research task
2. It picked `mistral-small-4` with medium reasoning effort
3. The researcher agent generated research output
4. That output was passed to the writer agent
5. The writer produced the final report
6. Total cost was tracked in EUR across both agents

## Next steps

- [Smart Fleet Guide](smart-fleet.md) — model routing, progressive reasoning, cascading
- [Patterns](patterns.md) — real-world examples from ArkhosAI and Gerald
- [API Reference](api-reference.md) — full API documentation
