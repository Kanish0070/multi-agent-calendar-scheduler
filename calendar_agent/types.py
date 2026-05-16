# types.py
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class AgentMessage:
    """Message protocol between agents and orchestrator"""
    sender: str
    recipient: str
    message_type: str  # 'request', 'response', 'error', 'status'
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AgentInfo:
    """Information about an agent"""
    agent_id: str
    agent_type: str
    status: str  # 'idle', 'busy', 'error', 'completed'
    created_at: datetime = field(default_factory=datetime.now)