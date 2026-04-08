# Changelog

## v0.2.4 (2026-04-07) — ArkhosAI Battle-Test Fixes

**Bugs Gerald and ArkhosAI found during real-world testing.**

### Added
- **`Agent.run(system_prompt=...)`** — optional override parameter on `run()` and
  `run_stream()`. When set, replaces the auto-built role/goal/backstory prompt
  for that call only. Gerald-style pattern works cleanly:
  `await agent.run(input_text, system_prompt=_QUALIFICATION_PROMPT)`.

### Fixed
- **Silent degradation on unknown model alias** — `Agent._run_once()` now raises
  `ModelNotAvailableError` with a clear message when an alias isn't in
  `MISTRAL_MODELS` (e.g. passing `mistral-small-latest` instead of `mistral-small`).
  `run_stream()` yields an equivalent error event. Previously these calls
  silently degraded to 1-file output with `cost_eur=0`.
- **Aggressive pre-call budget check** — tolerance bumped from 2x to 5x as
  a new class constant `_BUDGET_TOLERANCE`. The pre-call estimate is now a
  soft runaway guard; actual enforcement stays post-call via Pipeline /
  RunContext with real token counts.
- **Code-aware cost estimation** — `_estimate_call_cost()` detects code models
  (`devstral-*`, anything with `code`/`swe` strengths) and uses a 2.0x output
  multiplier instead of 0.8x. Output is capped at `max_output_tokens/2`.
  Prevents ArkhosAI's Builder from being blocked before it can generate.
- **Self-learning router telemetry** — `Agent.run()` now records a
  `RoutingOutcome` to the router's telemetry (when set) after every
  execution. Previously telemetry recorded 0 outcomes even with
  `router = MistralRouter(telemetry=FleetTelemetry())`.

### Validated
- 329 unit tests passing (323 → 329, +6 new)
- ruff clean, mypy clean

## v0.2.3 (2026-04-02) — Memory + Skills

**The only agent framework with state-of-the-art memory, typed skills, and intelligent model routing.**

### TramontaneMemory (3-tier agent-controlled memory)
- **Working Memory** — always-in-context labeled blocks, agent-editable
- **Factual Memory** — knowledge graph with mistral-embed vectors, FTS5, entity links
- **Experiential Memory** — outcome tracking for self-improvement
- **4-channel retrieval** — semantic + BM25 + entity graph + temporal, fused via RRF
- **5 memory tools** — retain, recall, reflect, forget (GDPR Article 17), update
- **Auto fact extraction** — ministral-3b extracts atomic facts after every run
- **Agent integration** — tramontane_memory=, memory_tools=True, auto_extract_facts=True

### TramontaneSkills (typed, composable, learnable)
- **Skill base class** — typed I/O, triggers, preferred model, budget, memory tags
- **@track_skill decorator** — auto-profiling (timing, cost, success/failure)
- **SkillRegistry** — SQLite-backed, keyword + semantic search, security verification
- **SkillLoader** — Python modules, SKILL.md (OpenClaw), YAML (NVIDIA-inspired)
- **Composition** — SkillPipeline, ConditionalSkill, ParallelSkills, SkillPersona
- **5 built-in skills** — TextAnalysis, CodeGeneration, EmailDraft, DataExtraction, WebSearch
- **Security** — SHA-256 hash, dangerous pattern detection (os.system, eval, exec)
- **MCP tool export** — skill.to_mcp_tool() for publishing

### Self-Learning Router Fix
- Agent.run() now records telemetry outcomes when router has FleetTelemetry
- Router learns from real production data after 50+ outcomes

### Documentation
- Complete README rewrite (319 lines, 16 code examples, comparison table)
- docs/memory.md — 3-tier memory deep dive
- docs/skills.md — typed skills system guide
- Updated quickstart, smart-fleet, patterns, api-reference

### Stats
- 324 tests, ruff clean, mypy clean
- 60 source files, 36 public exports, 84 code examples in docs

## v0.2.2 (2026-04-01) — The Evolution Release

**The only agent framework that gets smarter every time you use it.**

### New Capabilities
- Tool calling: agents call Python functions + external APIs natively
- Structured output: `output_schema=PydanticModel` for typed responses
- Knowledge bases: RAG with mistral-embed, grounded in your documents
- Parallel execution: run independent agents simultaneously
- Pipeline YAML: define pipelines in .yaml files, run from CLI
- Voice pipelines: speech -> agent -> speech via Voxtral TTS

### Fleet Intelligence
- FleetTuner: auto-discover optimal model+config per agent role
- Self-learning router evolves from rule-based to data-driven
- Progressive reasoning, model cascading, adaptive budget reallocation
- Cost simulation, FleetProfile presets (BUDGET/BALANCED/QUALITY/UNIFIED)

### CLI
- `tramontane simulate` — cost estimation without API calls
- `tramontane knowledge ingest/search` — manage knowledge bases
- `tramontane telemetry stats` — fleet performance insights

### Documentation
- docs/quickstart.md, docs/smart-fleet.md, docs/patterns.md, docs/api-reference.md

### Stats
- 234 tests, ruff clean, mypy clean
- Battle-tested on ArkhosAI and Gerald

## v0.2.1 (2026-04-01) — Smart Fleet + Self-Learning Router

**The conductor the Mistral fleet was waiting for.**

### New Models
- Mistral Small 4 (unified: reasoning + vision + code, configurable effort)
- Mistral Large 3 (675B MoE frontier, Apache 2.0)
- Voxtral TTS (text-to-speech, 9 languages, edge-deployable)

### Smart Fleet
- `reasoning_effort` parameter (none/medium/high) for Mistral Small 4
- Progressive reasoning: auto-escalate effort on validation failure
- Model cascading: try cheap models first, escalate on failure
- FleetProfile presets: BUDGET, BALANCED, QUALITY, UNIFIED
- Pipeline cost simulation: estimate cost before running
- Adaptive budget reallocation: unspent budget flows to later agents

### Self-Learning Router
- FleetTelemetry records every routing decision + outcome
- After 50+ outcomes, router uses data-driven suggestions
- Per-model success rate, cost, and latency tracking
- `tramontane doctor` and `tramontane fleet` CLI commands

### Router Improvements
- Mistral Small 4 as new default for general/reasoning/vision
- Simplified routing: code->devstral, classification->ministral, else->small-4
- Reasoning effort auto-determined from task complexity

### Validated
- 157 unit tests passing
- ruff clean, mypy clean

## v0.2.0 (2026-04-01) — Battle-tested on ArkhosAI

### Added
- **Agent.routing_hint** — guides router classification when model="auto"
  (e.g. "text-only JSON output" prevents vision misroute).
- **Agent.temperature** — sampling temperature (0.0-1.5), passed to Mistral API.
  `None` = model default. `0.0` valid (deterministic).
- **Agent.validate_output** — optional callback returns True/False. On False,
  agent auto-retries up to max_validation_retries (default 2).
- **Agent.run() context parameter** — dynamic per-call context appended to
  system prompt without recreating the Agent.
- **on_pattern callback in run_stream()** — dict of {regex: callback(match, text)}.
  Fires mid-stream when pattern matches accumulated output. Supports async callbacks.
  Yields "pattern_match" events.
- **RunContext** — shared cost tracker for multi-agent chains. Tracks spent_eur,
  remaining_eur, per-agent cost breakdown. Pass to run(run_context=ctx).
- **Per-model max_tokens defaults** — Agent auto-applies model's max_output_tokens
  when max_tokens=None (8192 for ministral/voxtral, 32768 for others).
- **StreamEvent.pattern_id** field + "pattern_match"/"validation_retry" event types.

### Fixed
- **Router: design → vision misclassification** — "Create a design system" no longer
  routes to pixtral-large (€2/6 per 1M). Design tasks map to "reasoning" or "code".
  Updated classifier system prompt with vision vs design distinction.
  Updated TASK_TYPE_ALIASES: design/ui_design/ux_design/color/layout → reasoning,
  styling → code.

### Validated
- 109 unit tests passing (85 → 109, +24 new)
- ruff clean, mypy clean

## v0.1.6 (2026-04-01)

### Added
- **Agent.max_tokens** field — controls maximum output tokens passed to
  Mistral API. `None` (default) uses model default. Set to 16000+ for
  long-form generation (e.g. ArkhosAI Builder generating full React projects).
- **MistralModel.max_output_tokens** — each model in the registry now
  declares its maximum output token limit (8192 for ministral/voxtral,
  32768 for all others).
- `max_tokens` passed to both `run()` and `run_stream()` API calls.

### Validated
- 85 unit tests passing (81 + 4 new)
- ruff clean, mypy clean

## v0.1.5 (2026-04-01)

### Added
- **Agent.run_stream()** — Token-by-token async streaming via Mistral SDK's
  `chat.stream_async()`. Yields `StreamEvent` objects (start/token/complete/error).
  Enables live preview in ArkhosAI and any SSE-based frontend.
- **StreamEvent** model — Pydantic model for streaming events, exported from
  `tramontane` package.

### Fixed
- **Router classifier validation** — Online classifier output is now validated
  and normalized before routing. Invalid task types (e.g. "design", "creative",
  "analysis") are remapped to valid types. Unknown types default to "general".
  Added `VALID_TASK_TYPES`, `TASK_TYPE_ALIASES`, `_validate_task_type()`.
- **Budget pre-estimation too aggressive** — `_estimate_call_cost()` multipliers
  reduced (output: 2.0x/1.5x to 1.2x/0.8x; overhead: 1.4x to 1.1x).
  `check_budget()` now uses 2x tolerance (soft pre-call guard). Prevents
  BudgetExceededError on affordable ministral-3b calls with tight budgets.
- **TaskType enum updated** — Removed "creative"/"analysis" (unmapped in router),
  added "classification"/"voice" with proper routing and quality floors.

### Validated
- 81 unit tests passing (74 + 7 new)
- ruff clean, mypy clean

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
