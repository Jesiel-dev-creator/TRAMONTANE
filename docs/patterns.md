# Real-World Patterns

Production patterns from ArkhosAI, Gerald, and common architectures.

## Pattern 1: ArkhosAI — Website Generator

4-agent pipeline generating full websites. EUR 0.004/generation.

```yaml
name: ArkhosAI Builder
budget_eur: 0.02
agents:
  planner:
    role: Planner
    model: mistral-small-4
    reasoning_effort: medium
  designer:
    role: Designer
    model: mistral-small-4
    reasoning_effort: none
    routing_hint: text-only JSON output
  builder:
    role: Frontend Builder
    model: devstral-small
    max_tokens: 32000
  reviewer:
    role: Code Reviewer
    model: devstral-small
flow: [planner, designer, builder, reviewer]
```

### Streaming with File Extraction

```python
async for event in builder.run_stream(
    prompt,
    on_pattern={
        r'<file path="(?P<path>[^"]+)">\n(?P<content>.*?)</file>':
            on_file_complete,
    },
):
    if event.type == "token":
        emit_sse("token", event.token)
    elif event.type == "pattern_match":
        emit_sse("file_ready", event.pattern_id)
```

### Cascade for Reliability

```python
builder = Agent(
    model="devstral-small",
    cascade=["devstral-2", "mistral-large-3"],
    validate_output=lambda r: r.output.count("</file>") >= 5,
)
```

### Parallel Design + Architecture

```python
from tramontane import ParallelGroup

group = ParallelGroup([
    Agent(role="Designer", goal="Create design tokens"),
    Agent(role="Architect", goal="Plan component structure"),
])
result = await group.run(input_text="Design a bakery website")
design_tokens = result.get("Designer").output
architecture = result.get("Architect").output
```

## Pattern 2: Gerald — Business Intelligence Agent

Memory-powered business agent with skills and tool calling.

### Orchestrator with Skills

```python
from tramontane import SkillRegistry, TramontaneMemory

memory = TramontaneMemory(db_path="gerald.db")
registry = SkillRegistry()
registry.register(LeadResearchSkill())
registry.register(LeadQualifierSkill())
registry.register(EmailDraftSkill())

# Orchestrator finds best skill
matches = registry.search("research Scaleway")
skill = matches[0][0]
result = await skill.execute_with_memory("Research Scaleway", memory=memory)
```

### Tool Calling for Web Research

```python
async def search_linkedin(company: str) -> str:
    """Search LinkedIn for company info."""
    return await api.search(company)

researcher = Agent(
    role="Lead Researcher",
    tools=[search_linkedin, search_crunchbase],
    tramontane_memory=memory,
    memory_tools=True,
    auto_extract_facts=True,
)
```

### Conditional Pipeline

```python
from tramontane.skills.composition import ConditionalSkill, SkillPipeline

qualify_if_good = ConditionalSkill(
    skill=email_skill,
    condition=lambda prev: prev and "score: 8" in prev.output.lower(),
)
pipe = SkillPipeline([research_skill, qualify_skill, qualify_if_good])
```

## Pattern 3: Cost Optimization

### Step 1: Tune
```python
from tramontane import FleetTuner
result = await FleetTuner().tune(builder, test_prompts, optimize_for="balanced")
builder = result.apply(builder)
```

### Step 2: Progressive Reasoning
```python
builder = builder.model_copy(update={
    "reasoning_strategy": "progressive",
    "validate_output": lambda r: len(r.output) > 1000,
})
```

### Step 3: Cascade Safety Net
```python
builder = builder.model_copy(update={
    "cascade": ["devstral-2", "mistral-large-3"],
})
```

### Step 4: Monitor
```python
from tramontane import MistralRouter, FleetTelemetry
router = MistralRouter(telemetry=FleetTelemetry())
# tramontane telemetry stats
```

### Step 5: Adaptive Budget
```python
from tramontane import RunContext
ctx = RunContext(budget_eur=0.05, reallocation="adaptive")
```

## Pattern 4: RAG Support Agent

```python
from tramontane import Agent, KnowledgeBase, TramontaneMemory

kb = KnowledgeBase(db_path="support_kb.db")
await kb.ingest(sources=["docs/*.md", "faq/*.txt"])

memory = TramontaneMemory(db_path="support_memory.db")

support = Agent(
    role="Support Agent",
    goal="Answer customer questions accurately",
    backstory="Expert support engineer",
    knowledge=kb,
    knowledge_top_k=5,
    tramontane_memory=memory,
    memory_tools=True,
    auto_extract_facts=True,
)

result = await support.run("How do I configure SSO?")
# 1. Retrieves top-5 chunks from knowledge base
# 2. Generates answer grounded in documentation
# 3. Auto-extracts any new facts from the conversation
```

## Pattern 5: Multi-Skill Orchestrator

```python
from tramontane import SkillRegistry, SkillPipeline
from tramontane.skills.composition import ParallelSkills, SkillPersona

registry = SkillRegistry()
# Register domain skills
registry.register(ResearchSkill())
registry.register(AnalysisSkill())
registry.register(ReportSkill())

# Discovery
task = "Analyze Scaleway's competitive position"
matches = registry.search(task)

# Sequential execution
pipe = SkillPipeline(
    [matches[0][0], matches[1][0]],
    persona=SkillPersona(
        name="Gerald",
        description="EU business agent",
        instructions="Be data-driven. Think in EUR.",
    ),
)
results = await pipe.run(task)
```
