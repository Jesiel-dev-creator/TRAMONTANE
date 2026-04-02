# Tramontane

> **The only agent framework with state-of-the-art memory, typed skills, and intelligent model routing.**

Mistral-native agent orchestration with 3-tier memory, composable skills, 4-channel retrieval, and GDPR compliance. Built in Orleans, France.

```bash
pip install tramontane
```

[![PyPI](https://img.shields.io/pypi/v/tramontane)](https://pypi.org/project/tramontane/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

## Why Tramontane?

| Feature | CrewAI | LangGraph | OpenClaw | Tramontane |
|---------|--------|-----------|----------|------------|
| Role-based agents | Yes | No | No | Yes |
| 3-tier memory (working+factual+experiential) | No | No | Basic | **Yes** |
| Agent-controlled memory tools | No | No | No | **Yes** |
| 4-channel retrieval + RRF fusion | No | No | No | **Yes** |
| Typed skills with profiling | No | No | .md only | **Yes** |
| Skill composition (pipelines) | No | No | Lobster | **Yes** |
| Tool calling (native functions) | Yes | Yes | Yes | Yes |
| Structured output (Pydantic) | Yes | No | No | **Yes** |
| Reasoning effort control | No | No | No | **Yes** |
| Progressive reasoning | No | No | No | **Yes** |
| Model cascading | No | No | No | **Yes** |
| Self-learning router | No | No | No | **Yes** |
| FleetTuner (auto-optimize) | No | No | No | **Yes** |
| Parallel execution | Yes | Yes | No | Yes |
| Knowledge bases (RAG) | Yes | Yes | No | Yes |
| Voice pipelines (TTS/STT) | No | No | No | **Yes** |
| Cost simulation (dry run) | No | No | No | **Yes** |
| EUR cost tracking | No | No | No | **Yes** |
| GDPR middleware | No | No | No | **Yes** |
| MCP tool export | No | No | No | **Yes** |

## Quick Start

```python
import asyncio
from tramontane import Agent, MistralRouter

agent = Agent(
    role="Analyst",
    goal="Analyze market trends",
    backstory="Senior market analyst",
    model="auto",
    budget_eur=0.01,
)

async def main():
    router = MistralRouter()
    result = await agent.run("Analyze the EU AI market", router=router)
    print(f"Model: {result.model_used}, Cost: EUR {result.cost_eur:.4f}")
    print(result.output)

asyncio.run(main())
```

## Memory

3-tier memory: working (always in context), factual (knowledge graph), experiential (self-improvement).

```python
from tramontane import Agent, TramontaneMemory

memory = TramontaneMemory(db_path="memory.db")

agent = Agent(
    role="Gerald",
    goal="Remember everything about clients",
    backstory="Autonomous business agent",
    tramontane_memory=memory,
    memory_tools=True,        # Gets retain/recall/reflect/forget/update tools
    auto_extract_facts=True,  # Auto-extracts facts after every run
    working_memory_blocks=["Goals", "User"],
)
```

The agent can call `retain_memory("Acme Corp prefers React")`, `recall_memory("What does Acme prefer?")`, `reflect_on_memory("What patterns have I seen?")`, `forget_memory(id, "GDPR request")`, and `update_memory(id, "new info")` during execution.

4-channel retrieval: semantic (cosine similarity on mistral-embed vectors) + keyword (FTS5 BM25) + entity (graph traversal) + temporal (recency + frequency). Results fused via Reciprocal Rank Fusion (k=60).

[Full memory docs](docs/memory.md)

## Skills

Typed, composable, learnable capabilities with profiling and security.

```python
from tramontane import Skill, SkillResult, SkillRegistry, track_skill

class LeadQualifier(Skill):
    name = "lead_qualifier"
    description = "Score B2B leads against ICP"
    triggers = ["qualify", "score lead"]
    preferred_model = "ministral-3b-latest"

    @track_skill  # Auto-logs timing, cost, success/failure
    async def execute(self, input_text, context=None):
        from tramontane import Agent
        agent = Agent(role="Qualifier", goal="Score leads", backstory="Sales expert",
                      model=self.preferred_model)
        result = await agent.run(input_text)
        return SkillResult(output=result.output, success=True, cost_eur=result.cost_eur)

registry = SkillRegistry()
registry.register(LeadQualifier())  # SHA-256 hash + security scan
matches = registry.search("qualify this lead")
```

Includes 5 built-in skills: TextAnalysis, CodeGeneration, EmailDraft, DataExtraction, WebSearch. Supports Python, YAML, and SKILL.md formats. Compose with `SkillPipeline`, `ConditionalSkill`, `ParallelSkills`.

[Full skills docs](docs/skills.md)

## Tool Calling

```python
async def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

agent = Agent(
    role="Researcher",
    goal="Find information",
    backstory="Expert researcher",
    tools=[search_web],
    tool_choice="auto",       # "auto" | "none" | "any" | "required"
    parallel_tool_calls=True,
    max_iter=5,
)
result = await agent.run("Research Mistral AI", router=router)
print(result.tool_calls)
```

## Structured Output

```python
from pydantic import BaseModel

class Analysis(BaseModel):
    summary: str
    score: int
    recommendations: list[str]

agent = Agent(role="Analyst", goal="Analyze", backstory="Expert",
              output_schema=Analysis)
result = await agent.run("Analyze this market")
analysis: Analysis = result.parsed_output  # Validated Pydantic model
```

## Smart Fleet

### Reasoning Effort
```python
agent = Agent(model="mistral-small-4", reasoning_effort="high")  # none | medium | high
```

### Progressive Reasoning
```python
agent = Agent(model="mistral-small-4", reasoning_strategy="progressive",
              validate_output=lambda r: "conclusion" in r.output)
# Tries none -> medium -> high, stops at first success
```

### Model Cascading
```python
agent = Agent(model="devstral-small",
              cascade=["devstral-2", "mistral-large-3"],
              validate_output=lambda r: len(r.output) > 1000)
```

### FleetTuner
```python
from tramontane import FleetTuner
tuner = FleetTuner()
result = await tuner.tune(agent, ["prompt1", "prompt2"], optimize_for="balanced")
optimized = result.apply(agent)
```

### Self-Learning Router
```python
from tramontane import MistralRouter, FleetTelemetry
router = MistralRouter(telemetry=FleetTelemetry())
# After 50+ decisions, routes by YOUR production data
```

### Fleet Profiles
```python
from tramontane import FleetProfile
agent = Agent(fleet_profile=FleetProfile.BUDGET)  # BUDGET | BALANCED | QUALITY | UNIFIED
```

[Full fleet docs](docs/smart-fleet.md)

## Parallel Execution

```python
from tramontane import ParallelGroup
group = ParallelGroup([designer, architect])
result = await group.run(input_text="Design a website")
print(result.get("Designer").output)
print(f"Total: EUR {result.total_cost_eur:.4f}")
```

## Knowledge Bases (RAG)

```python
from tramontane import KnowledgeBase
kb = KnowledgeBase(db_path="knowledge.db")
await kb.ingest(sources=["docs/*.md"])
agent = Agent(role="Support", goal="Help", backstory="Expert", knowledge=kb, knowledge_top_k=5)
```

## Pipeline YAML

```yaml
name: Lead Gen
budget_eur: 0.01
agents:
  researcher:
    role: Researcher
    model: mistral-small-4
  writer:
    role: Writer
    model: devstral-small
    temperature: 0.8
flow: [researcher, writer]
```
```bash
tramontane run pipeline.yaml --input "Research Scaleway"
```

## Voice Pipelines

```python
from tramontane import VoicePipeline
vpipe = VoicePipeline(agent=my_agent, enable_tts=True)
result = await vpipe.run(text_input="Brief me on today's leads")
# result.audio_bytes = spoken response via Voxtral TTS
```

## Streaming

```python
async for event in agent.run_stream("Generate a report",
    on_pattern={r"## (?P<section>.+)": on_section_found}):
    if event.type == "token":
        print(event.token, end="", flush=True)
    elif event.type == "tool_call":
        print(f"\n[Calling {event.tool_name}]")
```

## GDPR

```python
agent = Agent(role="Processor", goal="Process data", backstory="Expert",
              gdpr_level="strict", audit_actions=True)
# Built-in PII detection, Article 17 erasure, Article 30 reports
```

## The Mistral Fleet

| Model | Best For | EUR/1M in/out | Reasoning | Vision |
|-------|----------|---------------|-----------|--------|
| ministral-3b | Classification, triage | 0.04/0.04 | | |
| ministral-7b | Bulk, extraction | 0.10/0.10 | | |
| mistral-small | General, multilingual | 0.10/0.30 | | |
| mistral-small-4 | General+reasoning+vision | 0.15/0.60 | Yes | Yes |
| devstral-small | Code generation | 0.10/0.30 | | |
| devstral-2 | Complex SWE | 0.50/1.50 | | |
| magistral-small | Reasoning, planning | 0.50/1.50 | | |
| magistral-medium | Deep reasoning | 2.00/5.00 | | |
| mistral-large | Frontier synthesis | 2.00/6.00 | | |
| mistral-large-3 | Frontier (Apache 2.0) | 2.00/6.00 | | |
| pixtral-large | Vision, OCR | 2.00/6.00 | | |
| voxtral-mini | Transcription | 0.04/0.04 | | |
| voxtral-tts | Text-to-speech | 0.016/char | | |

## CLI

```bash
tramontane models                    # Fleet with pricing + capabilities
tramontane doctor                    # Health check + API connectivity
tramontane fleet                     # Fleet stats from telemetry
tramontane simulate pipeline.yaml    # Cost estimate without API calls
tramontane knowledge ingest docs/    # Build knowledge base
tramontane knowledge search "query"  # Search knowledge base
tramontane telemetry stats           # Router learning metrics
```

## Built With Tramontane

- **ArkhosAI** — EU answer to Lovable. 4-agent website generator, EUR 0.004/generation.
- **Gerald** — Autonomous business intelligence agent with memory + skills.

## Install

```bash
pip install tramontane                # Core
pip install tramontane[redis]         # Redis memory backend
pip install tramontane[postgres]      # PostgreSQL + pgvector
pip install tramontane[voice]         # Voice gateway
pip install tramontane[sandbox]       # E2B code sandbox
```

## Links

- [GitHub](https://github.com/Jesiel-dev-creator/TRAMONTANE)
- [PyPI](https://pypi.org/project/tramontane/)
- [Live Demo](https://hf.co/spaces/BleuCommerce-Apps/TRAMONTANE-demo)
- [Docs](https://github.com/Jesiel-dev-creator/TRAMONTANE/tree/main/docs)

## License

MIT — Bleucommerce SAS, Orleans, France
