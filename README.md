# TRAMONTANE

**Mistral-native agent orchestration. Not bolted on top.**

[![PyPI](https://img.shields.io/pypi/v/tramontane)](https://pypi.org/project/tramontane/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-61%20passing-green.svg)]()

---

## Why Tramontane

- **Mistral-native** -- built FROM Mistral's primitives (Agents API, Conversations, Handoffs), not adapted from OpenAI patterns
- **Smart router** -- Ministral-3B classifies your task, then routes to the optimal model from a 10-model fleet. You set `model="auto"` and a EUR budget ceiling; the router handles the rest
- **EU-sovereign** -- GDPR built-in (PII detection, Article 30 reports, right to erasure), EUR cost tracking, Scaleway fr-par deployment

## Quick Start

```bash
pip install tramontane
export MISTRAL_API_KEY=your_key_here
```

```python
import asyncio
from tramontane.core.agent import Agent
from tramontane.router.router import MistralRouter

async def main():
    agent = Agent(
        role="Code Reviewer",
        goal="Review code for bugs and security issues",
        backstory="Senior engineer with 10 years experience",
        model="auto",       # router picks the optimal Mistral model
        budget_eur=0.01,    # hard cost ceiling in EUR
    )
    router = MistralRouter()
    result = await agent.run("def add(a, b): return a + b", router=router)

    print(f"Model: {result.model_used}")      # e.g. devstral-small
    print(f"Cost:  EUR {result.cost_eur:.4f}") # e.g. EUR 0.0003
    print(result.output)

asyncio.run(main())
```

Or run a multi-agent pipeline from YAML:

```bash
tramontane run pipelines/code_review.yaml --input "def add(a,b): return a+b" --budget 0.05
```

## How the Router Works

Every prompt goes through a Ministral-3B classifier that determines task type and complexity, then the router picks the cheapest capable model:

```
has_vision?              -> pixtral-large
task=code, complexity<4  -> devstral-small    (EUR 0.10/0.30 per 1M)
task=code, complexity>=4 -> devstral-2        (EUR 0.50/1.50 per 1M)
needs_reasoning, <4      -> magistral-small   (EUR 0.50/1.50 per 1M)
needs_reasoning, >=4     -> magistral-medium  (EUR 2.00/5.00 per 1M)
task=bulk                -> ministral-7b      (EUR 0.10/0.10 per 1M)
default                  -> mistral-small     (EUR 0.10/0.30 per 1M)
```

When budget forces a downgrade, **quality floors** prevent garbage output:

| Task Type | Floor Model | Never Below |
|-----------|-------------|-------------|
| Reasoning | magistral-small | Tier 3 |
| Code | devstral-small | Tier 2 |
| Vision | pixtral-large | Tier 4 |
| General | mistral-small | Tier 2 |

If budget can't afford the floor, `BudgetExceededError` is raised instead of silently degrading.

## Comparison

| Feature | Tramontane | LangGraph | CrewAI | Mistral Agents API |
|---------|-----------|-----------|--------|--------------------|
| Mistral-native fleet routing | Yes | No | No | Partial |
| EUR budget ceilings + quality floors | Yes | No | No | No |
| GDPR PII + Article 30 built-in | Yes | No | No | No |
| Agentic + deterministic modes | Yes | Yes | Partial | No |
| Checkpoint/resume | Yes | Yes | No | No |
| SSE streaming API | Yes | Complex | No | Partial |
| Voice input (Voxtral) | Yes | No | No | No |
| MCP client (stdio + SSE) | Yes | No | No | Native |
| 3-agent code review cost | EUR 0.002 | ~EUR 0.15* | ~EUR 0.15* | N/A |

*Estimated with GPT-4o equivalent pricing.

## The Mistral Fleet

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

| Pipeline | Agents | Typical Cost | GDPR |
|----------|--------|-------------|------|
| `code_review` | Reviewer, Security Auditor, Writer | EUR 0.002 | none |
| `lead_gen_fr` | Prospector, Qualifier, Copywriter | EUR 0.001 | strict |
| `market_research` | Researcher, Analyst, Writer | EUR 0.010 | standard |
| `document_analysis` | Extractor, Legal Analyst, Summarizer | EUR 0.018 | strict |

## GDPR

Three levels: `none` (passthrough), `standard` (detect + log), `strict` (detect + redact + block).

- **PII detection** -- dual-mode: regex offline, Ministral-3B online for contextual PII
- **French PII** -- email, phone (+33), IBAN (FR), NIR, passport
- **Right to erasure** -- `memory.erase_user(user_id)`, Article 17 compliant
- **Audit vault** -- append-only, never deletes, Article 30 report generation
- **HTTP 451** -- `GDPRViolationError` returns the correct status code

## CLI

```bash
tramontane --version                 # show version
tramontane models                    # display the Mistral fleet
tramontane init                      # create database + health check
tramontane run <pipeline.yaml> \
  --input "..." --budget 0.05        # run a pipeline
tramontane serve --port 8080         # start FastAPI server
tramontane audit --run <run_id>      # view audit trail
```

## API Server

```bash
tramontane serve --port 8080
# Docs at http://localhost:8080/docs
# SSE streaming: POST /pipelines/run with "stream": true
```

All responses include `X-Tramontane-Version` and `X-EU-Sovereign: true` headers.

## Links

- **PyPI**: https://pypi.org/project/tramontane/
- **GitHub**: https://github.com/Jesiel-dev-creator/TRAMONTANE
- **HF Demo**: https://huggingface.co/spaces/BleuCommerce-Apps/TRAMONTANE-demo
- **HF Pipelines**: https://hf.co/datasets/BleuCommerce-Apps/tramontane-pipelines

## Contributing

```bash
git clone https://github.com/Jesiel-dev-creator/TRAMONTANE.git
cd TRAMONTANE
uv sync
uv run pytest tests/ -q    # 61 tests
uv run ruff check .        # lint
uv run mypy .              # type check (strict)
```

## License

MIT -- Bleucommerce SAS, Orleans, France
