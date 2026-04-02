# API Reference

Complete reference for all public exports from `tramontane`.

## Agent

```python
class Agent(BaseModel):
    # Identity
    role: str                          # Agent's role name (required)
    goal: str                          # What the agent aims to achieve (required)
    backstory: str                     # Context and expertise (required)

    # Model routing
    model: str = "auto"                # Model alias or "auto"
    function_calling_model: str = "auto"
    reasoning_model: str | None = None
    locale: str = "en"
    routing_hint: str | None = None    # Hint for classifier
    fleet_profile: FleetProfile | None = None

    # Tools
    tools: list[Any] = []              # Callable functions for tool calling
    tool_choice: str | None = None     # "auto" | "none" | "any" | "required"
    parallel_tool_calls: bool = True
    allow_code_execution: bool = False
    code_execution_mode: "safe" | "unsafe" = "safe"

    # Execution
    max_iter: int = 20                 # Max tool call iterations
    max_rpm: int | None = None
    max_execution_time: int | None = None  # Timeout (seconds)
    max_tokens: int | None = None      # Max output tokens
    max_retry_limit: int = 3
    respect_context_window: bool = True

    # Intelligence
    reasoning: bool = False
    max_reasoning_attempts: int | None = 3
    streaming: bool = True
    inject_date: bool = False
    allow_delegation: bool = False
    temperature: float | None = None
    reasoning_effort: "none" | "medium" | "high" | None = None
    reasoning_strategy: "fixed" | "progressive" = "fixed"

    # Memory
    memory: bool = True                # Agno-style history toggle
    add_history_to_context: bool = True
    learning: bool = False

    # TramontaneMemory (3-tier)
    tramontane_memory: Any | None = None
    memory_tools: bool = False
    auto_extract_facts: bool = False
    working_memory_blocks: list[str] = []

    # Knowledge (RAG)
    knowledge: Any | None = None       # KnowledgeBase instance
    knowledge_top_k: int = 5

    # Skills
    skills: list[Any] = []

    # Budget
    budget_eur: float | None = None

    # GDPR
    gdpr_level: "none" | "standard" | "strict" = "none"
    store_on_cloud: bool = True

    # Validation + Cascade
    validate_output: Callable | None = None
    max_validation_retries: int = 2
    cascade: list[str | dict] | None = None

    # Structured output
    output_schema: type[BaseModel] | None = None

    # Observability
    audit_actions: bool = True
    verbose: bool = False
    step_callback: Callable | None = None
```

### Methods

- `await agent.run(input_text, *, router=, run_context=, context=)` -> AgentResult
- `async for event in agent.run_stream(input_text, *, on_pattern=)` -> StreamEvent
- `agent.system_prompt()` -> str
- `Agent.estimate_cost(in_tokens, out_tokens, model)` -> float
- `Agent.from_yaml(path)` -> Agent

## AgentResult

```python
class AgentResult(BaseModel):
    output: str
    model_used: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_eur: float = 0.0
    duration_seconds: float = 0.0
    tool_calls: list[dict] = []
    reasoning_used: bool = False
    parsed_output: Any | None = None  # Set when output_schema used
```

## StreamEvent

```python
class StreamEvent(BaseModel):
    type: "start" | "token" | "complete" | "error" | "pattern_match"
         | "validation_retry" | "reasoning_escalation"
         | "cascade_escalation" | "tool_call"
    token: str = ""
    model_used: str = ""
    result: AgentResult | None = None
    error: str = ""
    pattern_id: str = ""
    tool_name: str = ""
    tool_args: str = ""
```

## RunContext

```python
@dataclass
class RunContext:
    budget_eur: float | None = None
    spent_eur: float = 0.0
    reallocation: "fixed" | "adaptive" = "fixed"
    remaining_eur: float | None  # property
    def record(agent_role: str, cost_eur: float): ...
    def get_effective_budget(role: str, budget: float | None): ...
```

## MistralRouter

```python
router = MistralRouter(telemetry=FleetTelemetry())
decision = await router.route(prompt, agent_budget_eur=, locale=, context=)
# decision.primary_model, .reasoning_effort, .estimated_cost_eur
```

## FleetTelemetry + RoutingOutcome

```python
telemetry = FleetTelemetry(db_path="telemetry.db")
telemetry.record(RoutingOutcome(task_type="code", complexity=3, ...))
telemetry.suggest_model("code", 3)  # Data-driven suggestion after 50+ outcomes
telemetry.get_model_stats()
telemetry.total_outcomes  # property
```

## FleetProfile

```python
class FleetProfile(str, Enum):
    BUDGET = "budget"      # Cheapest models
    BALANCED = "balanced"  # Smart routing (default)
    QUALITY = "quality"    # Best models
    UNIFIED = "unified"    # mistral-small-4 for everything
```

## FleetTuner + FleetTuneResult

```python
tuner = FleetTuner(models_to_test=["mistral-small-4", "devstral-small"])
result = await tuner.tune(agent, test_prompts, optimize_for="balanced")
optimized = result.apply(agent)
# result.optimal_model, .savings_vs_default, .total_tuning_cost_eur
```

## ParallelGroup + ParallelResult

```python
group = ParallelGroup([agent_a, agent_b])
result = await group.run(input_text="prompt")
result.get("RoleA")       # AgentResult by role
result.merge()             # Concatenated outputs
result.total_cost_eur
result.errors              # dict[str, str]
```

## KnowledgeBase + RetrievalResult

```python
kb = KnowledgeBase(db_path="kb.db", embedding_model="mistral-embed")
await kb.ingest(sources=["docs/*.md"], texts=[("raw text", "source")])
result = await kb.retrieve("query", top_k=5)
kb.format_context(result)
kb.chunk_count  # property
```

## TramontaneMemory

```python
memory = TramontaneMemory(db_path="mem.db")
await memory.retain(content, entity="", category="fact", source="")
await memory.recall(query, top_k=5, recency_weight=0.3)
await memory.reflect(question)
await memory.forget(memory_id, reason="")
await memory.update(memory_id, new_content)
await memory.extract_facts(text, source="")
await memory.record_experience(action_type, summary, outcome, score, ...)
memory.get_working_blocks(agent_id)
memory.set_working_block(agent_id, label, content)
memory.format_context(results, max_tokens=2000)
memory.stats()  # MemoryStats
memory.fact_count  # property
memory.experience_count  # property
```

## MemoryRetriever

```python
retriever = MemoryRetriever(conn, embedding_model="mistral-embed")
results = await retriever.retrieve(query, top_k=5, recency_weight=0.3)
```

## WorkingBlock

```python
@dataclass
class WorkingBlock:
    id: str
    agent_id: str
    label: str
    content: str
    max_tokens: int = 500
```

## Skill + SkillResult

```python
class Skill(ABC):
    name, description, version, triggers, preferred_model,
    preferred_temperature, max_tokens, budget_eur, tools,
    output_schema, memory_tags, author, tags

    async def execute(input_text, context=None) -> SkillResult
    async def execute_with_memory(input_text, memory=, context=)
    def validate(result) -> bool
    def matches(query) -> float  # 0-1
    def to_dict() -> dict
    def to_mcp_tool() -> dict

@dataclass
class SkillResult:
    output, parsed_output, cost_eur, model_used,
    success, validation_passed, error, metadata
```

## SkillRegistry

```python
registry = SkillRegistry(db_path="skills.db")
registry.register(skill, verify=True)
registry.unregister(name)
registry.get(name)
registry.search(query, top_k=5)
registry.list_all()
registry.get_by_tag(tag)
registry.record_execution(name, success, cost, duration, quality_score)
registry.verify_skill(skill)  # SHA-256 + dangerous pattern check
await registry.semantic_search(query, top_k=5)
```

## SkillLoader + MarkdownSkill

```python
SkillLoader.load_from_module("my.module")
SkillLoader.load_from_directory("skills/")
SkillLoader.load_from_skill_md("SKILL.md")
SkillLoader.load_from_yaml("skill.yaml")
```

## SkillPipeline + ConditionalSkill + ParallelSkills

```python
pipe = SkillPipeline([skill_a, skill_b], persona=SkillPersona(...))
results = await pipe.run(input_text, context={})

cond = ConditionalSkill(skill=s, condition=lambda prev: prev.success)
par = ParallelSkills([skill_a, skill_b])
results = await par.run(input_text)
```

## VoicePipeline + VoiceResult

```python
vpipe = VoicePipeline(agent, enable_tts=True)
result = await vpipe.run(text_input="Brief me")
# result.transcript, .agent_output, .audio_bytes, .cost_eur
```

## PipelineSpec + YAML Pipeline

```python
spec = load_pipeline_spec("pipeline.yaml")
results = await run_yaml_pipeline("pipeline.yaml", "input", router=router)
```

## Functions

```python
create_memory_tools(memory)  # Returns 5 callable memory tools
simulate_agent(agent, input_text, router=None)  # AgentSimulation
simulate_pipeline(agents, input_text, budget_eur=None)  # PipelineSimulation
track_skill  # Decorator for skill execute() profiling
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `tramontane models` | Fleet with pricing + capabilities |
| `tramontane doctor` | Health check + API connectivity |
| `tramontane fleet` | Fleet stats from telemetry |
| `tramontane simulate <yaml>` | Cost estimate without API |
| `tramontane knowledge ingest <path>` | Build RAG knowledge base |
| `tramontane knowledge search <query>` | Search knowledge base |
| `tramontane telemetry stats` | Router learning metrics |
| `tramontane init` | Initialize database |
| `tramontane audit` | View audit trails |
| `tramontane run <yaml>` | Run YAML pipeline |
| `tramontane serve` | Start API server |
