# Memory System

Tramontane's 3-tier agent-controlled memory with Mistral embeddings.

## Why Memory Matters

Without memory, every agent call starts from zero. The agent forgets the user's name, past decisions, what worked before. TramontaneMemory gives agents persistent, searchable, GDPR-compliant memory.

## Architecture

```
Tier 1: Working Memory   — Always in context, agent-editable labels
Tier 2: Factual Memory   — Knowledge graph + embeddings + FTS5
Tier 3: Experiential     — Outcomes, learnings, self-improvement
```

## Quick Start

```python
from tramontane import Agent, TramontaneMemory

memory = TramontaneMemory(db_path="memory.db")

agent = Agent(
    role="Gerald",
    goal="Remember everything about clients",
    backstory="Autonomous business agent",
    tramontane_memory=memory,
    memory_tools=True,
    auto_extract_facts=True,
    working_memory_blocks=["Goals", "User"],
)
```

## Tier 1: Working Memory

Small, labeled blocks always injected into the system prompt. The agent reads and edits them.

```python
memory.set_working_block("gerald", "Goals", "Close 3 deals this week")
memory.set_working_block("gerald", "User", "Name: Alice, Pref: warm colors")

blocks = memory.get_working_blocks("gerald")
# [{"label": "Goals", "content": "Close 3 deals..."}, ...]
```

Working memory is injected as:
```
## Working Memory
### Goals
Close 3 deals this week
### User
Name: Alice, Pref: warm colors
```

## Tier 2: Factual Memory

Stores atomic facts with embeddings for semantic search.

```python
# Store a fact
mem_id = await memory.retain(
    content="Acme Corp prefers React for frontend",
    entity="Acme Corp",
    category="preference",
    source="meeting_notes",
)

# Recall relevant facts
results = await memory.recall("What does Acme prefer?", top_k=5)

# Update a fact
await memory.update(mem_id, "Acme Corp now prefers Vue.js")

# Forget (GDPR Article 17)
await memory.forget(mem_id, reason="GDPR erasure request")
```

### Deduplication

When storing a fact, the system embeds it and compares against existing facts. If cosine similarity > 0.92, it updates the existing fact instead of creating a duplicate.

### Auto Fact Extraction

With `auto_extract_facts=True`, after every `agent.run()`, the output is sent to ministral-3b which extracts atomic facts:

```python
# Agent outputs: "Acme Corp is based in Paris. They have 50 employees."
# Auto-extracted:
# - "Acme Corp is based in Paris" (entity: Acme Corp, category: fact)
# - "Acme Corp has 50 employees" (entity: Acme Corp, category: fact)
```

Cost: EUR 0.04/1M tokens (ministral-3b).

## Tier 3: Experiential Memory

Records what the agent did, what happened, and what it learned.

```python
await memory.record_experience(
    action_type="lead_qualification",
    summary="Qualified Acme Corp as hot lead",
    outcome="Client converted within 2 weeks",
    score=0.95,
    agent_role="Qualifier",
    model="ministral-3b",
    cost=0.001,
)
```

## 4-Channel Retrieval

When you call `memory.recall(query)`, four channels run simultaneously:

1. **Semantic**: Embed query with mistral-embed, cosine similarity against all fact embeddings
2. **Keyword**: FTS5 BM25 full-text search on fact content
3. **Entity**: Extract entities from query (capitalized words, quoted phrases), traverse entity graph 1-2 hops
4. **Temporal**: Score by recency, access frequency, and confidence

Results are fused using **Reciprocal Rank Fusion** (RRF, k=60):
```
rrf_score(doc) = sum(1 / (60 + rank_in_channel)) for each channel
```

## Memory Tools

When `memory_tools=True`, the agent gets 5 callable tools:

| Tool | What it does |
|------|-------------|
| `retain_memory(content, entity, category)` | Store a new fact |
| `recall_memory(query, top_k)` | Search memory |
| `reflect_on_memory(question)` | Synthesize insights from multiple memories |
| `forget_memory(memory_id, reason)` | GDPR-safe deletion |
| `update_memory(memory_id, new_content)` | Correct existing fact |

The agent decides when to call these during execution.

## Reflection

`reflect()` uses mistral-small to synthesize insights from multiple recalled memories:

```python
answer = await memory.reflect("What patterns have I seen in successful deals?")
# Returns synthesized answer drawing from experiential + factual memory
```

## GDPR Compliance

- `forget()` soft-deletes (sets `erased_at`, removes from FTS5 index)
- Every erasure logged to `memory_erasure_log` table
- All operations logged to `memory_audit` table
- Stats available via `memory.stats()`

## API Reference

```python
class TramontaneMemory:
    async def retain(content, entity="", category="fact", source="") -> str
    async def recall(query, top_k=5, recency_weight=0.3) -> list[dict]
    async def reflect(question) -> str
    async def forget(memory_id, reason="") -> bool
    async def update(memory_id, new_content) -> bool
    async def extract_facts(text, source="") -> list[str]
    async def record_experience(action_type, summary, outcome, score, ...) -> str
    def get_working_blocks(agent_id) -> list[dict]
    def set_working_block(agent_id, label, content) -> None
    def format_context(results, max_tokens=2000) -> str
    def stats() -> MemoryStats
    fact_count: int  # property
    experience_count: int  # property
```

## Cost

| Operation | Model | Cost |
|-----------|-------|------|
| Embedding (retain/recall) | mistral-embed | EUR 0.10/1M tokens |
| Fact extraction | ministral-3b | EUR 0.04/1M tokens |
| Reflection | mistral-small | EUR 0.10/1M tokens |
| Keyword/entity/temporal search | SQLite | Free |
