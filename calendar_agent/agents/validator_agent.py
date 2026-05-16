"""Validator Agent - Checks calendar availability and validates operations"""

import asyncio
from typing import Optional
from ..tools import CalendarAPI
from ..types import AgentMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_openai import ChatOpenAI


class ValidatorAgent:
    """
    Validator Agent that checks calendar availability when asked.
    Spawns on-demand by the orchestrator supervisor.
    """
    
    def __init__(self, agent_id: str = "validator-agent"):
        self.agent_id = agent_id
        self.mcp_tools = CalendarAPI()
        
        # LLM for reasoning
        self.llm = ChatOpenAI(
            base_url="http://127.0.0.1:8082/v1", 
            api_key="not-needed",
            model="agentscope-ai_CoPaw-Flash-9B-Q4_K_M.gguf", 
            temperature=0.7,
        )
        
        # System prompt for validation role
        self.system_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(
                content="""
You are a Calendar Validator Agent. Your sole purpose is to:
1. Check calendar availability before scheduling events
2. Validate proposed schedules for conflicts
3. Verify event details match user requirements
4. Report availability issues clearly

You must always use the checkAvailability tool to verify time slots.
If any conflicts are found, report them with specific details.
"""
            )
        ])
    
    async def check_availability(
        self,
        start_time: str,
        end_time: str,
        event_title: Optional[str] = None
    ) -> dict:
        """
        Validate calendar availability for the given time range.
        
        Args:
            start_time: ISO format start time
            end_time: ISO format end time
            event_title: Optional event title for context
        
        Returns:
            Availability status and details
        """
        print(f"[ValidatorAgent] Checking availability: {start_time} - {end_time}")
        
        result: AvailabilityResult = await self.mcp_tools.check_availability(
            start_time=start_time,
            end_time=end_time
        )
        
        # Analyze results with LLM
        if result.get("available", False):
            # Confirm with LLM reasoning
            availability_message = f"The calendar shows availability for the requested time slot. Start: {start_time}, End: {end_time}"
            if event_title:
                availability_message += f". Event: {event_title}"
            
            response = await self.llm.ainvoke([
                SystemMessage(content=availability_message),
                HumanMessage(content="Please respond with a short confirmation based on the above information.")
            ])
            result["validation_analysis"] = response.content
            result["confidence"] = "high"
        else:
            response = await self.llm.ainvoke(
                [SystemMessage(content=availability_message)]
            )
            result["validation_analysis"] = response.content
            result["confidence"] = "high"
        
        print(f"[ValidatorAgent] Result: {result}")
        return result
    
    async def validate_event_creation(
        self,
        title: str,
        start_time: str,
        end_time: str,
        description: Optional[str] = None
    ) -> dict:
        """
        Validate all aspects before creating an event.
        
        Args:
            title: Event title
            start_time: ISO format start time
            end_time: ISO format end time
            description: Optional description
        
        Returns:
            Validation result with recommendations
        """
        # First check availability
        availability = await self.check_availability(start_time, end_time, title)
        
        # LLM validation analysis
        validation_prompt = f"""
        An event has these details:
        - Title: {title}
        - Start: {start_time}
        - End: {end_time}
        - Description: {description or 'N/A'}
        - Availability result: {availability.get('available', False)}
        {availability.get('validation_analysis', '')}
        
        Based on this information:
        1. Should this event be scheduled?
        2. What issues need attention?
        3. What recommendations do you have?
        """
        
        analysis = await self.llm.ainvoke(validation_prompt)
        
        return {
            "available": availability.get("available", False),
            "confidence": availability.get("confidence", "unknown"),
            "analysis": analysis.content,
            "recommendations": self._extract_recommendations(analysis.content)
        }
    
    def _extract_recommendations(self, text: str) -> list:
        """Extract action items from validation analysis."""
        recommendations = []
        # Simple extraction - can be improved with better NLP
        if "conflict" in text.lower():
            recommendations.append("RESOLVE_CONFLICT")
        if "suggest" in text.lower():
            recommendations.append("CONSIDER_ALTERNATIVE_TIME")
        if "approved" in text.lower() or "ready" in text.lower():
            recommendations.append("PROCEED")
        return recommendations
    
    def get_agent_info(self) -> dict:
        """Return agent metadata for the orchestrator."""
        return {
            "id": self.agent_id,
            "type": "validator",
            "purpose": "Check calendar availability and validate operations",
            "tools": ["checkAvailability"]
        }