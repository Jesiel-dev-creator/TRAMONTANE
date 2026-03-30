# Changelog

## v0.1.3 (2026-03-30)

### Hardened
- **Mutable default fix** -- 7 Pydantic fields across 6 files changed from
  `= []` / `= {}` to `Field(default_factory=...)`. Prevents cross-instance
  state contamination.
- **Dead code removal** -- Removed unused PrivateAttrs (_run_count,
  _conversation_id, _mistral_agent_id) from Agent. Agent is now fully stateless.
- **Exception handling** -- Zero bare `except:` clauses. All catches use
  specific exceptions with logging.
- **Edge case validation** -- Agent.run() validates: empty input, negative
  budget, missing API key. All raise with clear error messages.
- **SQLite async safety** -- `check_same_thread=False` on all 4 SQLite
  connections (longterm, pipeline, workflow, audit).
- **Retry logic** -- Exponential backoff on all Mistral API calls
  (2^attempt, capped 30s, max_retry_limit attempts).
- **Logging** -- NullHandler on root logger (library best practice).
  Zero print() in library code.

### Added
- 5 new tests (61 total): empty input, whitespace, negative budget,
  missing API key, run_sync bridge.
- 3 working examples: quickstart.py, pipeline_code_review.py, budget_routing.py
- Full README rewrite with comparison table, working quick start,
  router explanation, model fleet.
- PyPI metadata: classifiers (Beta), keywords, project URLs, Bug Tracker.
- CLAUDE.md updated to match v0.1.2+ architecture.

## v0.1.2 (2026-03-30)

### Fixed
- **Agent.run() method** -- Agent now owns model resolution, Mistral API calls,
  timeout handling, and cost tracking. Pipeline delegates to Agent instead of
  calling Mistral directly. Returns AgentResult with output, model_used,
  input_tokens, output_tokens, cost_eur, duration_seconds.
- **Async-first architecture** -- Removed all asyncio.run() from library internals.
  New run_sync() bridge (core/_sync.py) handles both sync and async contexts via
  anyio. asyncio.run() only in CLI entry points.
- **Unified cost tracking** -- Pipeline is now single source of truth. Agent reports
  per-call cost in AgentResult, Pipeline accumulates. Removed duplicate _cost_tracker.
  Budget check accepts spent_eur from caller.
- **Accurate cost estimation** -- Pre-call budget check now estimates both input and
  output token costs with 1.4x reasoning overhead multiplier. Actual cost uses real
  token counts from Mistral API response with per-model EUR pricing.
- **Budget quality floors** -- Router budget downgrade now enforces minimum model
  quality per task type: reasoning->magistral-small, code->devstral-small,
  vision->pixtral-large. Raises BudgetExceededError if floor can't be met instead
  of silently degrading.

### Validated
- 56/56 unit tests passing (4 new for quality floors)
- 6/6 live API integration tests passing:
  - SSE streaming (22 events)
  - Pipeline E2E (3 agents, EUR 0.002)
  - Article 30 GDPR report (JSON export)
  - GDPR strict PII redaction (email/phone/name detected and redacted)
  - Hub publish (YAML validated locally)
  - Voice gateway (Voxtral-Mini accepted audio)
- ruff clean, mypy --strict clean
- Total live API spend: ~EUR 0.005

## v0.1.0 (2026-03-29)

Initial release. Mistral-native agent orchestration framework.

- Smart model router across full Mistral fleet (10 models)
- GDPR-native: PII detection, Article 30 reports, data residency
- Pipeline DSL: agentic + workflow modes with durable checkpoints
- 7 failure mode guards from academic research
- Voxtral-Mini voice gateway
- Real MCP client (stdio + SSE)
- EU Premium CLI with Rich
- FastAPI server with SSE streaming
- 4 built-in pipelines
- Docker + docker-compose production-ready
