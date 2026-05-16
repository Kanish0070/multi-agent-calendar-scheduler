"""Cloud Fallback Agent - Handles edge cases and failed attempts"""

import asyncio
from typing import Optional, Dict, Any
from ..tools import CalendarAPI
from ..types import AgentMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI


class CloudFallbackAgent:
    """
    Cloud Fallback Agent that handles edge cases and recovery from failed attempts.
    Uses context from previous failures to provide better responses.
    
    This agent is spawned on-demand when:
    - An operation fails or times out
    - Edge cases are encountered
    - Context from previous attempts is needed
    """
    
    def __init__(self, agent_id: str = "cloud-fallback-agent"):
        self.agent_id = agent_id
        self.mcp_tools = CalendarAPI()
        self.llm = ChatOpenAI(
            base_url="http://127.0.0.1:8082/v1", 
            api_key="not-needed",
            model="agentscope-ai_CoPaw-Flash-9B-Q4_K_M.gguf", 
            temperature=0.7,
        )
        
        # Context storage for failed attempts
        self.failure_context: Dict[str, Any] = {}
        
        # System prompt for fallback role
        self.system_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(
                content="""
You are a Cloud Fallback Agent. Your purpose is to:
1. Handle edge cases and unusual scenarios
2. Provide alternative solutions when primary operations fail
3. Use context from previous failed attempts to improve responses
4. Escalate only when human intervention is necessary

Key behaviors:
- Always analyze why a previous attempt failed (if available)
- Suggest alternative approaches or workarounds
- Keep responses concise and actionable
- Escalate to human only for truly complex issues
"""
            )
        ])
    
    async def handle_error(
        self,
        error_type: str,
        error_message: str,
        original_request: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle an error by analyzing it and providing a recovery response.
        
        Args:
            error_type: Type of error that occurred
            error_message: Error message/details
            original_request: The original user request that failed
            context: Additional context from previous attempts
        
        Returns:
            Recovery suggestion or alternative action
        """
        # Store failure context for future reference
        self.failure_context[error_type] = {
            "message": error_message,
            "request": original_request,
            "context": context or {}
        }
        
        # Build analysis prompt
        prompt_content = f"""
        === ERROR CONTEXT ===
        Error Type: {error_type}
        Error Message: {error_message}
        Original Request: {original_request}
        Previous Context: {context if context else 'None available'}
        
        === YOUR TASK ===
        Analyze what went wrong and provide:
        1. The root cause of the failure
        2. Alternative approaches or workarounds
        3. What information you need from the user
        4. Whether human intervention is required
        """
        
        response = await self.llm.ainvoke(
            self.system_prompt.format(prompt_content)
        )
        
        # Extract actionable suggestions
        suggestions = self._extract_suggestions(response.content)
        
        # Update failure context with analysis
        self.failure_context[error_type]["analysis"] = {
            "root_cause": suggestions.get("root_cause", "Unknown"),
            "suggestions": suggestions.get("suggestions", []),
            "needs_human": suggestions.get("needs_human", False)
        }
        
        return {
            "error_analyzed": True,
            "error_type": error_type,
            "analysis": response.content,
            "suggestions": suggestions,
            "needs_human": suggestions.get("needs_human", False)
        }
    
    async def provide_alternative(
        self,
        original_intent: str,
        failed_action: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Provide alternative approaches when the primary action fails.
        
        Args:
            original_intent: What the user was trying to achieve
            failed_action: The action that failed
            context: Context from previous attempts
        
        Returns:
            Alternative actions and suggestions
        """
        prompt_content = f"""
        User Intent: {original_intent}
        Failed Action: {failed_action}
        Available Tools: checkAvailability, createEvent, deleteEvent, listEvents
        Context: {context if context else 'None'}
        
        Provide:
        1. Alternative ways to achieve the user's goal
        2. Step-by-step alternative approach
        3. What the user needs to provide for success
        """
        
        response = await self.llm.ainvoke(
            self.system_prompt.format(prompt_content)
        )
        
        return {
            "alternatives": self._extract_alternatives(response.content),
            "next_steps": self._extract_next_steps(response.content)
        }
    
    def _extract_suggestions(self, text: str) -> Dict[str, Any]:
        """Extract structured suggestions from LLM response."""
        suggestions = []
        needs_human = False
        
        if "human" in text.lower() or "escalate" in text.lower():
            needs_human = True
        if "alternative" in text.lower() or "workaround" in text.lower():
            suggestions.append("provide_alternative")
        if "check" in text.lower():
            suggestions.append("verify_prerequisites")
        
        return {
            "root_cause": self._extract_root_cause(text),
            "suggestions": suggestions,
            "needs_human": needs_human
        }
    
    def _extract_root_cause(self, text: str) -> str:
        """Extract root cause from analysis text."""
        if "conflict" in text.lower():
            return "Calendar conflict or double booking"
        elif "unavailable" in text.lower() or "timeout" in text.lower():
            return "Service unavailable or timeout"
        elif "permission" in text.lower() or "access" in text.lower():
            return "Permission or access issue"
        elif "format" in text.lower() or "invalid" in text.lower():
            return "Invalid input or format"
        else:
            return "Unknown or complex issue"
    
    def _extract_alternatives(self, text: str) -> list:
        """Extract alternative actions from text."""
        alternatives = []
        # Simple extraction - can be improved
        lines = text.strip().split('\n')
        for line in lines[:5]:  # Take first 5 lines as alternatives
            if line.strip().startswith("- ") or line.strip().startswith("1."):
                alternatives.append(line.strip())
        return alternatives
    
    def _extract_next_steps(self, text: str) -> list:
        """Extract next steps from text."""
        steps = []
        lines = text.strip().split('\n')
        for i, line in enumerate(lines):
            if i >= 3:  # Start from line 4
                cleaned = line.strip()
                if cleaned and not cleaned.startswith("User"):  # Skip user input lines
                    steps.append(cleaned)
        return steps
    
    def get_failure_context(self, error_type: str) -> Optional[Dict[str, Any]]:
        """Retrieve stored context for a specific error type."""
        return self.failure_context.get(error_type)
    
    def clear_failure_context(self, error_type: str) -> None:
        """Clear stored context for an error type."""
        if error_type in self.failure_context:
            del self.failure_context[error_type]
    
    def get_agent_info(self) -> Dict[str, str]:
        """Return agent metadata."""
        return {
            "id": self.agent_id,
            "type": "cloud_fallback",
            "purpose": "Handle edge cases and recovery from failed attempts",
            "tools": ["checkAvailability", "createEvent", "listEvents"]
        }
    
    async def escalate_to_human(self, issue: str, context: Dict[str, Any]) -> str:
        """
        Escalate an issue to human intervention.
        
        Args:
            issue: Description of the issue
            context: Context information for the human
        
        Returns:
            Escalation confirmation
        """
        prompt = f"""
        === ESCALATION REQUIRED ===
        
        Issue: {issue}
        
        Context:
        {str(context)}
        
        Please provide guidance on how to proceed. This issue requires human review.
        """
        
        response = await self.llm.ainvoke(prompt)
        return response.content