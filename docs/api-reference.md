# API Reference

Complete reference for Tramontane v0.2.2.

## Agent

```python
class Agent(BaseModel):
    # Identity
    role: str                    # Agent's role name
    goal: str                    # What the agent aims to achieve
    backstory: str               # Context and expertise description

    # Model routing
    model: str = "auto"          # Model alias or "auto" for router
    routing_hint: str | None     # Hint for classifier
    fleet_profile: FleetProfile | None  # Preset strategy

    # Tools
    tools: list[Any] = []        # Callable functions for tool calling
    tool_choice: str | None      # "auto", "none", "any", "required"
    parallel_tool_calls: bool = True

    # Execution
    max_iter: int = 20           # Max tool call iterations
    max_execution_time: int | None  # Timeout in seconds
    max_tokens: int | None       # Max output tokens
    max_retry_limit: int = 3     # API retry attempts
    temperature: float | None    # Sampling temperature (0.0-1.5)

    # Intelligence
    reasoning: bool = False      # Chain-of-thought prompting
    reasoning_effort: "none" | "medium" | "high" | None
    reasoning_strategy: "fixed" | "progressive" = "fixed"

    # Knowledge (RAG)
    knowledge: KnowledgeBase | None  # RAG knowledge base
    knowledge_top_k: int = 5

    # Budget
    budget_eur: float | None     # Per-agent cost ceiling

    # Validation
    validate_output: Callable | None  # Output validator
    max_validation_retries: int = 2
    cascade: list[str | dict] | None  # Model fallback chain

    # Structured output
    output_schema: type[BaseModel] | None  # Pydantic model for JSON mode

    # GDPR
    gdpr_level: "none" | "standard" | "strict" = "none"
```

### Methods

- `await agent.run(input_text, router=, run_context=, context=)` -> AgentResult
- `async for event in agent.run_stream(input_text, on_pattern=)` -> StreamEvent
- `agent.system_prompt()` -> str
- `Agent.estimate_cost(input_tokens, output_tokens, model)` -> float
- `Agent.from_yaml(path)` -> Agent

## AgentResult

```python
class AgentResult(BaseModel):
    output: str              # Generated text
    model_used: str          # Which model was used
    input_tokens: int        # Input token count
    output_tokens: int       # Output token count
    cost_eur: float          # Actual cost in EUR
    duration_seconds: float  # Wall-clock time
    tool_calls: list[dict]   # Tool calls made
    reasoning_used: bool     # Whether CoT was active
    parsed_output: Any       # Validated Pydantic model (if output_schema set)
```

## StreamEvent

```python
class StreamEvent(BaseModel):
    type: "start" | "token" | "complete" | "error"
         | "pattern_match" | "validation_retry"
         | "reasoning_escalation" | "cascade_escalation"
         | "tool_call"
    token: str               # Text chunk (for type="token")
    model_used: str
    result: AgentResult | None  # Full result (for type="complete")
    error: str               # Error message (for type="error")
    pattern_id: str          # Matched regex (for type="pattern_match")
    tool_name: str           # Tool name (for type="tool_call")
    tool_args: str           # Tool args JSON (for type="tool_call")
```

## RunContext

```python
@dataclass
class RunContext:
    budget_eur: float | None = None
    spent_eur: float = 0.0
    reallocation: "fixed" | "adaptive" = "fixed"

    remaining_eur: float | None  # property
    record(agent_role, cost_eur)  # Record cost
    get_effective_budget(role, budget)  # With adaptive reallocation
```

## MistralRouter

```python
router = MistralRouter(telemetry=FleetTelemetry())
decision = await router.route(prompt, agent_budget_eur=, locale=, context=)
# decision.primary_model, .reasoning_effort, .estimated_cost_eur
```

## KnowledgeBase

```python
kb = KnowledgeBase(db_path="kb.db", embedding_model="mistral-embed")
await kb.ingest(sources=["docs/*.md"], texts=[("raw text", "source")])
result = await kb.retrieve("query", top_k=5)
context = kb.format_context(result)
kb.chunk_count  # property
```

## FleetTuner

```python
tuner = FleetTuner(models_to_test=["mistral-small-4", "devstral-small"])
result = await tuner.tune(agent, test_prompts, optimize_for="balanced")
agent = result.apply(agent)
```

## ParallelGroup

```python
group = ParallelGroup([agent_a, agent_b])
result = await group.run(input_text="prompt", router=router)
result.get("RoleA")  # AgentResult by role
result.merge()       # Concatenated outputs
result.total_cost_eur
```

## Pipeline YAML

```yaml
name: Pipeline Name
version: "1.0"
budget_eur: 0.01
agents:
  agent_name:
    role: Role
    goal: Goal
    backstory: Backstory
    model: mistral-small-4
    temperature: 0.5
flow: [agent_name, ...]
```

```python
from tramontane import load_pipeline_spec, run_yaml_pipeline
spec = load_pipeline_spec("pipeline.yaml")
results = await run_yaml_pipeline("pipeline.yaml", "input text", router=router)
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `tramontane run <yaml>` | Run a YAML pipeline |
| `tramontane simulate <yaml>` | Estimate cost without API calls |
| `tramontane models` | Show model fleet |
| `tramontane fleet` | Fleet status with telemetry |
| `tramontane doctor` | Health check |
| `tramontane knowledge ingest <path>` | Ingest documents |
| `tramontane knowledge search <query>` | Search knowledge base |
| `tramontane telemetry stats` | Fleet performance stats |
| `tramontane init` | Initialize database |
| `tramontane audit` | View audit trails |
