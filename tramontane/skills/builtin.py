"""Built-in skills that ship with Tramontane."""

from __future__ import annotations

from typing import Any

from tramontane.skills.base import Skill, SkillResult, track_skill


class TextAnalysisSkill(Skill):
    """Analyze text: sentiment, summary, key entities."""

    name = "text_analysis"
    description = "Analyze text for sentiment, summary, and key entities"
    version = "1.0"
    triggers = ["analyze", "summarize", "sentiment", "entities"]
    preferred_model = "mistral-small-4"
    memory_tags = ["analysis"]
    tags = ["analysis", "nlp"]

    @track_skill
    async def execute(
        self, input_text: str, context: dict[str, Any] | None = None,
    ) -> SkillResult:
        """Analyze text using an Agent."""
        from tramontane.core.agent import Agent

        agent = Agent(
            role="Text Analyst",
            goal="Analyze text and extract insights",
            backstory="Expert NLP analyst",
            model=self.preferred_model,
            reasoning_effort="none",
        )
        result = await agent.run(
            f"Analyze this text:\n\n{input_text}",
        )
        return SkillResult(
            output=result.output,
            cost_eur=result.cost_eur,
            model_used=result.model_used,
            success=True,
        )


class CodeGenerationSkill(Skill):
    """Generate code from a description."""

    name = "code_generation"
    description = "Generate code from a natural language description"
    version = "1.0"
    triggers = ["code", "implement", "write function", "generate code"]
    preferred_model = "devstral-small"
    memory_tags = ["code"]
    tags = ["code", "development"]

    @track_skill
    async def execute(
        self, input_text: str, context: dict[str, Any] | None = None,
    ) -> SkillResult:
        """Generate code using a code-specialized Agent."""
        from tramontane.core.agent import Agent

        agent = Agent(
            role="Code Generator",
            goal="Write clean, working code",
            backstory="Senior software engineer",
            model=self.preferred_model,
            max_tokens=16000,
        )
        result = await agent.run(input_text)
        return SkillResult(
            output=result.output,
            cost_eur=result.cost_eur,
            model_used=result.model_used,
            success=True,
        )


class EmailDraftSkill(Skill):
    """Draft professional emails."""

    name = "email_draft"
    description = "Draft professional emails for business communication"
    version = "1.0"
    triggers = ["email", "draft email", "write email", "compose email"]
    preferred_model = "mistral-small-4"
    preferred_temperature = 0.7
    memory_tags = ["communication", "email"]
    tags = ["communication", "writing"]

    @track_skill
    async def execute(
        self, input_text: str, context: dict[str, Any] | None = None,
    ) -> SkillResult:
        """Draft an email using an Agent."""
        from tramontane.core.agent import Agent

        agent = Agent(
            role="Email Writer",
            goal="Draft clear, professional emails",
            backstory="Business communication expert",
            model=self.preferred_model,
            temperature=self.preferred_temperature,
        )
        result = await agent.run(
            f"Draft a professional email:\n\n{input_text}",
        )
        return SkillResult(
            output=result.output,
            cost_eur=result.cost_eur,
            model_used=result.model_used,
            success=True,
        )


class DataExtractionSkill(Skill):
    """Extract structured data from text."""

    name = "data_extraction"
    description = "Extract structured data (names, dates, amounts) from text"
    version = "1.0"
    triggers = ["extract", "parse", "structured data", "pull data"]
    preferred_model = "ministral-3b"
    memory_tags = ["extraction"]
    tags = ["extraction", "data"]

    @track_skill
    async def execute(
        self, input_text: str, context: dict[str, Any] | None = None,
    ) -> SkillResult:
        """Extract structured data using a cheap model."""
        from tramontane.core.agent import Agent

        agent = Agent(
            role="Data Extractor",
            goal="Extract structured information accurately",
            backstory="Data extraction specialist",
            model=self.preferred_model,
            temperature=0.1,
        )
        result = await agent.run(
            f"Extract all key data points:\n\n{input_text}",
        )
        return SkillResult(
            output=result.output,
            cost_eur=result.cost_eur,
            model_used=result.model_used,
            success=True,
        )


class WebSearchSkill(Skill):
    """Search the web for information."""

    name = "web_search"
    description = "Search the web for current information"
    version = "1.0"
    triggers = ["search", "find", "look up", "research", "what is"]
    preferred_model = "mistral-small-4"
    memory_tags = ["research", "web"]
    tags = ["research", "web"]

    @track_skill
    async def execute(
        self, input_text: str, context: dict[str, Any] | None = None,
    ) -> SkillResult:
        """Search the web using an Agent with search tools."""
        from tramontane.core.agent import Agent

        agent = Agent(
            role="Web Researcher",
            goal="Find accurate, current information",
            backstory="Expert web researcher",
            model=self.preferred_model,
            max_iter=3,
        )
        result = await agent.run(
            f"Research this topic:\n\n{input_text}",
        )
        return SkillResult(
            output=result.output,
            cost_eur=result.cost_eur,
            model_used=result.model_used,
            success=True,
        )


ALL_BUILTIN_SKILLS: list[type[Skill]] = [
    TextAnalysisSkill,
    CodeGenerationSkill,
    EmailDraftSkill,
    DataExtractionSkill,
    WebSearchSkill,
]
