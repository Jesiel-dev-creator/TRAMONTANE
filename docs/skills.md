# Skills System

Typed, composable, learnable capabilities for Tramontane agents.

## What Are Skills?

| Concept | Scope | Example |
|---------|-------|---------|
| **Tool** | Single function | `search_web(query)` |
| **Agent** | Model + identity + tools | Agent(role="Researcher") |
| **Skill** | Agent + tools + prompts + memory + validation + profiling | LeadQualifier with ICP scoring |

Skills combine everything an agent needs for a specific capability into a reusable, discoverable, trackable package.

## Creating a Skill

```python
from tramontane import Skill, SkillResult, track_skill

class LeadQualifier(Skill):
    name = "lead_qualifier"
    description = "Score B2B leads against ideal customer profile"
    version = "1.0"
    triggers = ["qualify", "score lead", "ICP match"]
    preferred_model = "ministral-3b-latest"
    preferred_temperature = 0.1
    budget_eur = 0.001
    memory_tags = ["leads", "qualification"]
    tags = ["sales", "b2b"]

    @track_skill
    async def execute(self, input_text, context=None):
        from tramontane import Agent
        agent = Agent(
            role="Lead Qualifier",
            goal="Score leads 1-10 against ICP",
            backstory="B2B sales qualification expert",
            model=self.preferred_model,
            temperature=self.preferred_temperature,
        )
        result = await agent.run(input_text)
        return SkillResult(
            output=result.output,
            cost_eur=result.cost_eur,
            model_used=result.model_used,
            success=True,
        )
```

## @track_skill Decorator

Auto-profiles every execution:

```python
@track_skill
async def execute(self, input_text, context=None):
    ...
# Logs: "Skill 'lead_qualifier' completed: success=True cost=EUR 0.0004 dur=1.23s"
# Sets: result.metadata["duration_s"], result.metadata["skill_name"]
# On exception: returns SkillResult(success=False, error="...")
```

## Skill Registry

```python
from tramontane import SkillRegistry

registry = SkillRegistry(db_path="skills.db")
registry.register(LeadQualifier())  # Stores + SHA-256 security hash

# Search
matches = registry.search("qualify this lead", top_k=3)
best_skill, score = matches[0]

# Execute
result = await best_skill.execute("Qualify Acme Corp")

# Track performance
registry.record_execution("lead_qualifier", success=True, cost=0.001, duration=1.5)

# Filter
sales_skills = registry.get_by_tag("sales")
all_skills = registry.list_all()
```

### Security Verification

```python
check = registry.verify_skill(my_skill)
# {"verified": True, "hash": "a1b2c3...", "warnings": []}
# Checks: name+description present, no os.system/subprocess/eval/exec in source
```

## Loading Skills

### From Python Module
```python
from tramontane import SkillLoader
skills = SkillLoader.load_from_module("gerald.skills.sales")
```

### From Directory
```python
skills = SkillLoader.load_from_directory("skills/")
```

### From SKILL.md (OpenClaw compatible)
```markdown
---
name: email_reviewer
description: Review emails for tone and clarity
triggers: [review email, check email]
---
Review the email for professional tone, clarity, and grammar...
```
```python
skill = SkillLoader.load_from_skill_md("skills/email_reviewer/SKILL.md")
```

### From YAML
```yaml
name: lead_qualification
description: Score B2B leads against ICP
triggers: [qualify, score lead]
preferred_model: ministral-3b-latest
temperature: 0.1
budget_eur: 0.001
prompt: |
  Score this lead 1-10 against our ideal customer profile...
```
```python
skill = SkillLoader.load_from_yaml("skills/lead_qual.yaml")
```

## Composition

### Sequential Pipeline
```python
from tramontane import SkillPipeline
pipe = SkillPipeline([research_skill, qualify_skill, email_skill])
results = await pipe.run("Research Acme Corp")
# research output -> qualify input -> email input
```

### Conditional Skills
```python
from tramontane.skills.composition import ConditionalSkill
cond = ConditionalSkill(
    skill=deep_analysis_skill,
    condition=lambda prev: prev and prev.success and "complex" in prev.output,
)
pipe = SkillPipeline([research_skill, cond])
```

### Parallel Skills
```python
from tramontane import ParallelSkills
par = ParallelSkills([design_skill, architecture_skill])
results = await par.run("Plan a bakery website")
```

### Persona
```python
from tramontane.skills.composition import SkillPersona
persona = SkillPersona(
    name="Gerald",
    description="EU business intelligence agent",
    instructions="You are direct, data-driven, cost-conscious.",
)
pipe = SkillPipeline([research_skill, email_skill], persona=persona)
```

## Built-in Skills

| Skill | Triggers | Model |
|-------|----------|-------|
| TextAnalysisSkill | analyze, summarize, sentiment | mistral-small-4 |
| CodeGenerationSkill | code, implement, write function | devstral-small |
| EmailDraftSkill | email, draft email, compose | mistral-small-4 |
| DataExtractionSkill | extract, parse, structured data | ministral-3b |
| WebSearchSkill | search, find, look up, research | mistral-small-4 |

## Memory Integration

```python
result = await skill.execute_with_memory(
    "Qualify Acme Corp",
    memory=tramontane_memory,
)
# Before: recalls memories matching skill.memory_tags
# After: records experience (action + outcome) for learning
```

## MCP Tool Export

```python
mcp_def = skill.to_mcp_tool()
# {"name": "lead_qualifier", "description": "...", "inputSchema": {...}}
```

## Skill Base Class Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| name | str | "" | Unique skill identifier |
| description | str | "" | What the skill does |
| version | str | "1.0" | Semantic version |
| triggers | list[str] | [] | Keywords that activate this skill |
| preferred_model | str | "auto" | Mistral model to use |
| preferred_temperature | float/None | None | Sampling temperature |
| max_tokens | int/None | None | Output token limit |
| budget_eur | float/None | None | Cost ceiling |
| tools | list[Callable] | [] | Function tools |
| output_schema | type[BaseModel]/None | None | Pydantic validation |
| memory_tags | list[str] | [] | Tags for memory recall filtering |
| author | str | "" | Skill author |
| tags | list[str] | [] | Categorization tags |
