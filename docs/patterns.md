# Real-World Patterns

Production patterns from ArkhosAI and Gerald.

## Pattern 1: ArkhosAI — 4-Agent Website Generator

ArkhosAI generates full websites from a prompt. EUR 0.004 per generation.

### Pipeline structure

```yaml
name: ArkhosAI Builder
budget_eur: 0.02

agents:
  planner:
    role: Planner
    goal: Plan the website structure
    model: mistral-small-4
    reasoning_effort: medium
    budget_eur: 0.002

  designer:
    role: Designer
    goal: Create design tokens (colors, fonts, spacing)
    model: mistral-small-4
    reasoning_effort: none
    routing_hint: text-only JSON output
    budget_eur: 0.001

  builder:
    role: Frontend Builder
    goal: Generate React components
    model: devstral-small
    cascade: ["devstral-2"]
    max_tokens: 32000
    budget_eur: 0.012

  reviewer:
    role: Code Reviewer
    goal: Fix bugs and improve quality
    model: devstral-small
    budget_eur: 0.005

flow: [planner, designer, builder, reviewer]
```

### SSE streaming with file extraction

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

### Validation + cascade prevents truncation

```python
builder = Agent(
    model="devstral-small",
    cascade=["devstral-2", "mistral-large-3"],
    validate_output=lambda r: r.output.count("</file>") >= 5,
    max_validation_retries=1,
)
```

## Pattern 2: Gerald — Business Operations Agent

Gerald runs 3 pipelines: lead gen, social media, weekly briefing.

### Lead generation with tool calling

```python
def search_linkedin(company: str) -> str:
    """Search LinkedIn for company contacts."""
    return api.search(company)

researcher = Agent(
    role="Lead Researcher",
    tools=[search_linkedin, search_crunchbase],
    knowledge=company_kb,  # RAG with company data
    model="mistral-small-4",
)
```

### Knowledge base for company context

```python
from tramontane import KnowledgeBase

kb = KnowledgeBase(db_path="gerald_knowledge.db")
await kb.ingest(sources=["company_docs/*.md", "crm_exports/*.json"])

qualifier = Agent(
    role="Lead Qualifier",
    knowledge=kb,
    knowledge_top_k=3,
)
```

### YAML pipeline

```yaml
name: Gerald Lead Gen
budget_eur: 0.01

agents:
  researcher:
    role: Lead Researcher
    goal: Find and research target companies
    model: mistral-small-4

  qualifier:
    role: Lead Qualifier
    goal: Score leads against ICP
    model: ministral-3b
    budget_eur: 0.001

  writer:
    role: Email Writer
    goal: Draft personalized cold emails
    model: mistral-small-4
    temperature: 0.8

flow: [researcher, qualifier, writer]
```

## Pattern 3: Cost Optimization Workflow

### Step 1: Tune with FleetTuner

```python
tuner = FleetTuner()
result = await tuner.tune(
    agent=builder,
    test_prompts=load_test_prompts("prompts.txt"),
    optimize_for="balanced",
)
builder = result.apply(builder)
```

### Step 2: Add progressive reasoning

```python
builder = builder.model_copy(update={
    "reasoning_strategy": "progressive",
    "validate_output": lambda r: len(r.output) > 1000,
})
```

### Step 3: Add cascade as safety net

```python
builder = builder.model_copy(update={
    "cascade": ["devstral-2", "mistral-large-3"],
})
```

### Step 4: Monitor with telemetry

```python
router = MistralRouter(telemetry=FleetTelemetry())
# Run agents through the router
# Check: tramontane telemetry stats
```

### Step 5: Use parallel execution for independent work

```python
from tramontane import ParallelGroup

group = ParallelGroup([designer, architect])
result = await group.run(input_text="Design a bakery website")
# Both run concurrently — half the wall-clock time
```
