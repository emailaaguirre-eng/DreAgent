"""
=============================================================================
HUMMINGBIRD-LEA - Agents
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
Agent management and access.
=============================================================================
"""

from typing import Dict, Optional

from .base import (
    BaseAgent,
    AgentResponse,
    ConversationContext,
    ConfidenceLevel,
    AmbiguityDetector,
)
from .lea import LeaAgent, get_lea
from .chiquis import ChiquisAgent, get_chiquis
from .grant import GrantAgent, get_grant


# =============================================================================
# Agent Registry
# =============================================================================

class AgentManager:
    """
    Manages all available agents and provides easy access.
    
    Usage:
        manager = AgentManager()
        lea = manager.get("lea")
        response = await lea.process("Hello!")
        
        # Or switch agents
        manager.set_active("chiquis")
        response = await manager.active.process("Write some code")
    """
    
    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {
            "lea": get_lea(),
            "chiquis": get_chiquis(),
            "grant": get_grant(),
        }
        self._active_agent: str = "lea"  # Default to Lea
    
    @property
    def available_agents(self) -> list:
        """List of available agent names"""
        return list(self._agents.keys())
    
    @property
    def active(self) -> BaseAgent:
        """Get the currently active agent"""
        return self._agents[self._active_agent]
    
    @property
    def active_name(self) -> str:
        """Get the name of the currently active agent"""
        return self._active_agent
    
    def get(self, name: str) -> Optional[BaseAgent]:
        """
        Get an agent by name.
        
        Args:
            name: Agent name (lea, chiquis, grant)
        
        Returns:
            The agent instance or None if not found
        """
        return self._agents.get(name.lower())
    
    def set_active(self, name: str) -> bool:
        """
        Set the active agent.
        
        Args:
            name: Agent name to activate
        
        Returns:
            True if successful, False if agent not found
        """
        name_lower = name.lower()
        if name_lower in self._agents:
            self._active_agent = name_lower
            return True
        return False
    
    def get_agent_info(self, name: str) -> Optional[dict]:
        """
        Get information about an agent.
        
        Args:
            name: Agent name
        
        Returns:
            Dict with agent info or None
        """
        agent = self.get(name)
        if agent:
            return {
                "name": agent.name,
                "role": agent.role,
                "capabilities": agent.capabilities,
            }
        return None
    
    def list_agents(self) -> list:
        """Get info for all agents"""
        return [
            self.get_agent_info(name)
            for name in self._agents.keys()
        ]


# =============================================================================
# Singleton Manager
# =============================================================================

_manager: Optional[AgentManager] = None


def get_agent_manager() -> AgentManager:
    """Get or create the agent manager singleton"""
    global _manager
    if _manager is None:
        _manager = AgentManager()
    return _manager


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Base
    "BaseAgent",
    "AgentResponse",
    "ConversationContext",
    "ConfidenceLevel",
    "AmbiguityDetector",
    # Agents
    "LeaAgent",
    "ChiquisAgent",
    "GrantAgent",
    "get_lea",
    "get_chiquis",
    "get_grant",
    # Manager
    "AgentManager",
    "get_agent_manager",
]
