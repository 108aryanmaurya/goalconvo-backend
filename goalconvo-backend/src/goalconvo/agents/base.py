"""Abstract base and shared types for GoalConvo agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..config import Config
    from ..llm_client import LLMClient
    from ..memory.dialogue_memory import DialogueMemory


@dataclass
class AgentGenerationResult:
    """Unified return type for ``generate()`` across agents."""

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    ok: bool
    issues: List[str] = field(default_factory=list)


class BaseDialogueAgent(ABC):
    """Research-oriented agent contract: generate, validate, optional memory hooks."""

    def __init__(self, config: "Config", llm_client: "LLMClient"):
        self.config = config
        self.llm_client = llm_client

    @abstractmethod
    def generate(self, **kwargs: Any) -> AgentGenerationResult:
        raise NotImplementedError

    def validate(self, result: AgentGenerationResult, ctx: Optional[Dict[str, Any]] = None) -> ValidationResult:
        text = (result.text or "").strip()
        if not text:
            return ValidationResult(ok=False, issues=["empty_generation"])
        return ValidationResult(ok=True, issues=[])

    def update_memory(self, memory: "DialogueMemory", recent_turns: List[Dict[str, Any]], **ctx: Any) -> None:
        """Override when this agent owns memory updates; default is no-op."""
        return None
