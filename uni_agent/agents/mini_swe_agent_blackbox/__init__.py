"""Black-box mini-swe-agent adapter.

This package intentionally lives beside, rather than inside, the existing
``uni_agent.agents.mini_swe_agent`` package so it can be developed without
touching the current stub.
"""

from .agent import MiniSweAgentBlackboxAgent, MiniSweAgentBlackboxConfig

__all__ = ["MiniSweAgentBlackboxAgent", "MiniSweAgentBlackboxConfig"]
