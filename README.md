# TRAMONTANE

**Mistral-native agent orchestration.**
Route intelligently. Govern completely. Ship on EU infrastructure.

![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)
![EU Sovereign](https://img.shields.io/badge/EU-sovereign-blue)
![Mistral Native](https://img.shields.io/badge/Mistral-native-orange)

---

Tramontane is an open-source agent orchestration framework built exclusively on Mistral's model fleet. It routes every prompt to the optimal Mistral model based on task type, complexity, and budget -- then wraps execution with GDPR-native data governance, cost ceilings, and EU-sovereign infrastructure defaults. Unlike LangGraph or CrewAI, Tramontane is built *from* Mistral's primitives outward, not bolted on top.

## Quick Start

```bash
pip install tramontane
export MISTRAL_API_KEY=your_key_here
tramontane run pipelines/market_research.yaml --input "AI market in France 2026" --budget 0.05
```

## Code Example

```python
from tramontane.core.agent import Agent
from tramontane.core.pipeline import Pipeline

researcher = Agent(
    role="Market Researcher",
    goal="Find EU AI market data",
    backstory="Senior analyst with 10 years experience",
    model="auto",           # router picks the best Mistral model
    budget_eur=0.05,        # hard cost ceiling in EUR
    gdpr_level="standard",  # PII detection active
)

analyst = Agent(
    role="Data Analyst",
    goal="Structure findings into actionable insights",
    backstory="Quantitative analyst",
    model="auto",
    reasoning=True,         # routes to magistral for CoT
)

pipeline = Pipeline(
    name="market_research",
    agents=[researcher, analyst],
    handoffs=[("Market Researcher", "Data Analyst")],
    budget_eur=0.10,
    streaming=True,
)
```

## Why Tramontane

| Feature | Tramontane | LangGraph | CrewAI | Agno |
|---------|-----------|-----------|--------|------|
| Mistral-native | Yes | No | No (LiteLLM) | No (OpenAI) |
| Model router | Automatic | Manual | Manual | Manual |
| GDPR-native | Built-in | No | No | No |
| Budget ceiling (EUR) | Per-agent + pipeline | No | No | No |
| EU sovereign | Scaleway fr-par | No | No | No |
| Local mode (Ollama) | One flag | Complex | Complex | Complex |
| Pipeline hub | HuggingFace | LangSmith | CrewAI Hub | No |
| Cost tracking | Per-token EUR | No | No | No |

## The Mistral Fleet

Every prompt is routed to the optimal model automatically:

| Model | Tier | Best For | EUR/1M in | EUR/1M out |
|-------|------|----------|-----------|------------|
| `ministral-3b` | 0 | Classification, PII, triage | 0.04 | 0.04 |
| `ministral-7b` | 1 | Bulk, extraction, tool calls | 0.10 | 0.10 |
| `mistral-small` | 2 | General, multilingual | 0.10 | 0.30 |
| `devstral-small` | 2 | All code tasks, SWE | 0.10 | 0.30 |
| `magistral-small` | 3 | Reasoning, CoT, planning | 0.50 | 1.50 |
| `magistral-medium` | 3 | Deep reasoning | 2.00 | 5.00 |
| `devstral-2` | 4 | Complex SWE, monorepo | 0.50 | 1.50 |
| `pixtral-large` | 4 | Vision, multimodal, OCR | 2.00 | 6.00 |
| `mistral-large` | 4 | Frontier, synthesis | 2.00 | 6.00 |
| `voxtral-mini` | 1 | Voice transcription | 0.04 | 0.04 |

## Built-in Pipelines

```bash
tramontane run pipelines/market_research.yaml --input "..." --budget 0.15
tramontane run pipelines/lead_gen_fr.yaml --input "..." --budget 0.05 --gdpr strict
tramontane run pipelines/code_review.yaml --input "..." --budget 0.10
tramontane run pipelines/document_analysis.yaml --input "..." --budget 0.20 --gdpr strict
```

## GDPR

Tramontane is GDPR-native, not GDPR-bolted-on:

- **Three levels**: `none` (passthrough), `standard` (detect + log), `strict` (detect + redact + block)
- **PII detection**: Dual-mode -- regex offline, Ministral-3B online for contextual PII
- **French PII**: Email, phone (+33), IBAN (FR), NIR (securite sociale), passport
- **Right to erasure**: `memory.erase_user(user_id)` -- Article 17 compliant
- **Audit vault**: Append-only log, never deletes -- Article 30 report generation
- **EU residency**: Mistral AI (Paris) + Scaleway EU-west-1 (Paris) -- data never leaves EU
- **HTTP 451**: `GDPRViolationError` returns the correct status code

## CLI

```bash
tramontane run <pipeline.yaml> --input "..." --budget 0.05 --gdpr strict
tramontane models                  # display the Mistral fleet
tramontane init                    # create database + verify health
tramontane audit --run <run_id>    # view audit trail
tramontane hub search <query>      # browse community pipelines
tramontane serve --port 8080       # start the API server
```

## API Server

```bash
tramontane serve --port 8080
# or
uvicorn tramontane.server.app:create_app --factory --port 8080
```

All pipeline runs support SSE streaming via `POST /pipelines/run` with `stream: true`.

## Docker

```bash
docker compose up -d
# API available at http://localhost:8080
# Docs at http://localhost:8080/docs
```

## Contributing

Contributions welcome. Please follow the coding rules in `CLAUDE.md`.

## License

MIT (core) -- Open-core commercial (TramontaneOS Pro)

---

Built in Orleans, France by [Bleucommerce SAS](https://bleucommerce.fr)
