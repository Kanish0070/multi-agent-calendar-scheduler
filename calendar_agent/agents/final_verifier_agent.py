"""Final Verifier Agent - Always runs before returning results"""

import asyncio
from typing import Optional, Dict, Any
from ..tools import CalendarAPI
from ..types  import AgentMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI


class FinalVerifierAgent:
    """
    Final Verifier Agent that always runs before returning results.
    Double-checks all operations to ensure correctness and safety.
    
    This agent is always spawned by the orchestrator to:
    - Validate operation outcomes
    - Check for side effects
    - Ensure data consistency
    - Provide final confidence assessment
    """
    
    def __init__(self, agent_id: str = "final-verifier-agent"):
        self.agent_id = agent_id
        self.mcp_tools = CalendarAPI()
        self.llm = ChatOpenAI(
            base_url="http://127.0.0.1:8082/v1", 
            api_key="not-needed",
            model="agentscope-ai_CoPaw-Flash-9B-Q4_K_M.gguf", 
            temperature=0.7,
        )
        
        # System prompt for verification role
        self.system_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(
                content="""
You are the Final Verifier Agent. Your role is to double-check ALL operations
before returning results to the user. You are the last line of defense.

Your responsibilities:
1. Verify operation outcomes (did it actually happen?)
2. Check for side effects or unintended consequences
3. Validate data consistency
4. Assess confidence levels
5. Flag any issues that need attention
6. Make final go/no-go decisions

You must ALWAYS run before returning results to the user.
No result should be sent without your verification.
"""
            )
        ])
    
    async def verify_operation(
        self,
        operation_type: str,
        operation_result: Dict[str, Any],
        original_request: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Verify an operation result before returning to user.
        
        Args:
            operation_type: Type of operation (create, delete, update, etc.)
            operation_result: Result from the operation
            original_request: Original user request
            context: Additional context
        
        Returns:
            Verification result with confidence assessment
        """
        # Build verification prompt
        prompt_content = f"""
        === OPERATION TO VERIFY ===
        Operation: {operation_type}
        Result: {str(operation_result)[:500]}
        Original Request: {str(original_request)[:500]}
        Context: {str(context) if context else 'None'}
        
        === VERIFICATION CHECKLIST ===
        1. Does the result match the expected outcome?
        2. Are there any side effects or issues?
        3. Is the data consistent?
        4. What is your confidence level?
        5. Any warnings or notes for the user?
        """
        
        response = await self.llm.ainvoke(
            self.system_prompt.format(prompt_content)
        )
        
        # Extract verification details
        verification = self._parse_verification_response(response.content)
        
        # Perform automated checks
        automated_checks = await self._run_automated_checks(
            operation_type, operation_result, original_request
        )
        
        # Combine LLM and automated verification
        final_verification = {
            "operation": operation_type,
            "verified": verification.get("verified", False),
            "confidence": verification.get("confidence", "unknown"),
            "automated_checks": automated_checks,
            "llm_analysis": verification.get("analysis", ""),
            "warnings": verification.get("warnings", []),
            "recommendations": verification.get("recommendations", []),
            "approved": self._should_approve(verification, automated_checks)
        }
        
        return final_verification
    
    async def verify_event_creation(
        self,
        event_id: str,
        event_details: Dict[str, Any],
        expected_details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Verify an event was created correctly.
        
        Args:
            event_id: The ID of the created event
            event_details: Details from creation response
            expected_details: Expected details for comparison
        
        Returns:
            Verification result
        """
        # Check if event exists by listing events
        listed_events = await self.mcp_tools.list_events(page_size=10)
        
        # Verify event is in the list
        event_found = any(e.get("id") == event_id for e in listed_events)
        
        # Compare details if expected provided
        details_match = True
        if expected_details:
            for key, value in expected_details.items():
                if event_details.get(key) != value:
                    details_match = False
                    break
        
        return {
            "verified": event_found and details_match,
            "event_id": event_id,
            "found_in_calendar": event_found,
            "details_match": details_match,
            "confidence": "high" if event_found and details_match else "low"
        }
    
    async def verify_event_deletion(
        self,
        event_id: str,
        original_event_details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Verify an event was deleted correctly.
        
        Args:
            event_id: The ID of the event to verify deletion
            original_event_details: Details of the original event
        
        Returns:
            Verification result
        """
        # List events and check if event is gone
        listed_events = await self.mcp_tools.list_events(page_size=10)
        
        event_still_exists = any(e.get("id") == event_id for e in listed_events)
        
        # Check for deleted events flag
        deleted_event = None
        for e in listed_events:
            if e.get("id") == event_id:
                deleted_event = e
                break
        
        return {
            "verified": not event_still_exists,
            "event_id": event_id,
            "event_still_exists": event_still_exists,
            "deleted_event": deleted_event,
            "confidence": "high" if not event_still_exists else "low"
        }
    
    def _parse_verification_response(self, text: str) -> Dict[str, Any]:
        """Parse LLM verification response."""
        verification = {
            "verified": False,
            "confidence": "unknown",
            "analysis": "",
            "warnings": [],
            "recommendations": []
        }
        
        # Simple parsing - can be improved with structured output
        lines = text.strip().split('\n')
        
        for line in lines:
            if "verified" in line.lower():
                if "yes" in line.lower() or "correct" in line.lower():
                    verification["verified"] = True
                elif "no" in line.lower() or "incorrect" in line.lower():
                    verification["verified"] = False
            if "confidence" in line.lower():
                parts = line.split()
                for i, part in enumerate(parts):
                    if "high" in part or "low" in part or "medium" in part:
                        verification["confidence"] = part
                        break
            if line.strip().startswith("Warning") or "warning" in line.lower():
                verification["warnings"].append(line.strip())
            if line.strip().startswith("Recommend") or "recommend" in line.lower():
                verification["recommendations"].append(line.strip())
        
        verification["analysis"] = text[:1000]
        return verification
    
    def _run_automated_checks(
        self,
        operation_type: str,
        operation_result: Dict[str, Any],
        original_request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run automated verification checks."""
        checks = {
            "performed": True,
            "results": [],
            "issues": []
        }
        
        # Check result structure
        if not operation_result:
            checks["issues"].append("No result returned")
        elif "error" in operation_result:
            checks["issues"].append(f"Error in result: {operation_result['error']}")
        
        # Operation-specific checks
        if operation_type == "createEvent":
            if "event_id" not in operation_result:
                checks["issues"].append("No event ID in creation response")
            elif not operation_result["event_id"]:
                checks["issues"].append("Empty event ID")
        
        elif operation_type == "deleteEvent":
            if "success" not in operation_result:
                checks["issues"].append("No success indicator in deletion response")
        
        checks["results"].append({
            "type": "structure_check",
            "passed": len(checks["issues"]) == 0
        })
        
        return checks
    
    def _should_approve(
        self,
        verification: Dict[str, Any],
        automated_checks: Dict[str, Any]
    ) -> bool:
        """Determine if operation should be approved based on verification."""
        # Auto-reject if there are issues
        if automated_checks.get("issues"):
            return False
        
        # Check LLM verification
        if not verification.get("verified", False):
            return False
        
        # Check confidence
        confidence = verification.get("confidence", "unknown")
        if confidence in ["low", "unknown"]:
            return False
        
        # Check for warnings
        warnings = verification.get("warnings", [])
        if any("critical" in w.lower() for w in warnings):
            return False
        
        return True
    
    def get_agent_info(self) -> Dict[str, str]:
        """Return agent metadata."""
        return {
            "id": self.agent_id,
            "type": "final_verifier",
            "purpose": "Double-check all operations before returning results",
            "always_run": True
        }
    
    async def flag_for_review(self, issue: str, context: Dict[str, Any]) -> str:
        """
        Flag an issue for human review.
        
        Args:
            issue: Description of the issue
            context: Context information
        
        Returns:
            Review request confirmation
        """
        prompt = f"""
        === FLAG FOR HUMAN REVIEW ===
        
        Issue: {issue}
        
        Context:
        {str(context)}
        
        Please provide a summary for human review and any recommendations.
        """
        
        response = await self.llm.ainvoke(prompt)
        return response.content