"""
Grant Agent Integration for LEA
Routes incentives questions to Grant (EIAGUS)
"""

import sys
from pathlib import Path

# Add EIAGUS to path if it exists
eiagus_path = Path(r"C:\Users\email\Eiagus_Agent_Grant")
if eiagus_path.exists():
    sys.path.insert(0, str(eiagus_path))

try:
    from eiagus.agent.grant import GrantAgent
    from eiagus.agent.state import AgentState
    GRANT_AVAILABLE = True
except ImportError:
    GRANT_AVAILABLE = False
    GrantAgent = None
    AgentState = None


class GrantIntegration:
    """Integration wrapper for Grant agent in LEA"""
    
    def __init__(self):
        self.grant = None
        self.state = None
        if GRANT_AVAILABLE:
            try:
                self.grant = GrantAgent()
                self.state = AgentState()
            except Exception as e:
                print(f"Warning: Could not initialize Grant agent: {e}")
                self.grant = None
    
    def process(self, user_input: str) -> str:
        """
        Process user input through Grant agent.
        
        Returns:
            Grant's response, or fallback message if Grant unavailable
        """
        if not self.grant or not self.state:
            return (
                "⚠️ Grant (EIAGUS) is not available. "
                "Please ensure EIAGUS is installed at: C:\\Users\\email\\Eiagus_Agent_Grant\n\n"
                "I can still help with other tasks. What else can I assist with?"
            )
        
        try:
            # Process through Grant
            response = self.grant.process(user_input, self.state)
            return response
        except Exception as e:
            return (
                f"⚠️ Error processing with Grant: {e}\n\n"
                "I can still help with other tasks. What else can I assist with?"
            )


# Global instance
_grant_integration = None

def get_grant_integration():
    """Get or create Grant integration instance"""
    global _grant_integration
    if _grant_integration is None:
        _grant_integration = GrantIntegration()
    return _grant_integration

