"""SENTINEL agent layer — exports all five agents and the base class."""
from sentinel.agents.base import BaseAgent
from sentinel.agents.argus import ARGUS
from sentinel.agents.holmes import HOLMES
from sentinel.agents.forge import FORGE
from sentinel.agents.hermes import HERMES
from sentinel.agents.oracle import ORACLE

__all__ = ["BaseAgent", "ARGUS", "HOLMES", "FORGE", "HERMES", "ORACLE"]
