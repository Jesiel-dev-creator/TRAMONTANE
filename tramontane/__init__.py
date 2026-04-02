"""Tramontane — Mistral-native agent orchestration framework."""

from __future__ import annotations

import logging

__version__ = "0.2.3"
__author__ = "Bleucommerce SAS"
__license__ = "MIT"

# Library best practice: let the user configure logging.
logging.getLogger("tramontane").addHandler(logging.NullHandler())

# Public API — convenience imports (after logging setup intentionally)
from tramontane.core.agent import Agent, AgentResult, RunContext, StreamEvent  # noqa: E402
from tramontane.core.parallel import ParallelGroup, ParallelResult  # noqa: E402
from tramontane.core.pipeline import Pipeline  # noqa: E402
from tramontane.core.profiles import FleetProfile  # noqa: E402
from tramontane.core.simulate import (  # noqa: E402
    PipelineSimulation,
    simulate_agent,
    simulate_pipeline,
)
from tramontane.core.tuner import FleetTuner, FleetTuneResult  # noqa: E402
from tramontane.core.yaml_pipeline import (  # noqa: E402
    PipelineSpec,
    load_pipeline_spec,
    run_yaml_pipeline,
)
from tramontane.knowledge.base import KnowledgeBase, RetrievalResult  # noqa: E402
from tramontane.memory.engine import TramontaneMemory  # noqa: E402
from tramontane.memory.retrieval import MemoryRetriever  # noqa: E402
from tramontane.memory.tools import create_memory_tools  # noqa: E402
from tramontane.memory.working import WorkingBlock  # noqa: E402
from tramontane.router.router import MistralRouter  # noqa: E402
from tramontane.router.telemetry import FleetTelemetry, RoutingOutcome  # noqa: E402
from tramontane.skills.base import Skill, SkillResult  # noqa: E402
from tramontane.skills.composition import (  # noqa: E402
    ConditionalSkill,
    ParallelSkills,
    SkillPipeline,
)
from tramontane.skills.loader import MarkdownSkill, SkillLoader  # noqa: E402
from tramontane.skills.registry import SkillRegistry  # noqa: E402
from tramontane.voice.tts import VoicePipeline, VoiceResult  # noqa: E402

__all__ = [
    "Agent",
    "AgentResult",
    "FleetProfile",
    "FleetTelemetry",
    "FleetTuneResult",
    "FleetTuner",
    "KnowledgeBase",
    "MistralRouter",
    "ParallelGroup",
    "ParallelResult",
    "Pipeline",
    "PipelineSpec",
    "PipelineSimulation",
    "RetrievalResult",
    "RoutingOutcome",
    "RunContext",
    "StreamEvent",
    "load_pipeline_spec",
    "run_yaml_pipeline",
    "simulate_agent",
    "simulate_pipeline",
    "TramontaneMemory",
    "MemoryRetriever",
    "WorkingBlock",
    "create_memory_tools",
    "Skill",
    "SkillResult",
    "SkillRegistry",
    "SkillLoader",
    "SkillPipeline",
    "ConditionalSkill",
    "ParallelSkills",
    "MarkdownSkill",
    "VoicePipeline",
    "VoiceResult",
    "__version__",
]
