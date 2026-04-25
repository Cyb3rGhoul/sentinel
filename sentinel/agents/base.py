"""Base agent interface for SENTINEL agents."""
from __future__ import annotations

from abc import ABC, abstractmethod

from sentinel.models import Action


class BaseAgent(ABC):
    """Abstract base class for all SENTINEL agents."""

    @abstractmethod
    def act(self, observation: dict) -> Action:
        """Produce an action given the current observation."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset agent state at the start of a new episode."""
        ...
