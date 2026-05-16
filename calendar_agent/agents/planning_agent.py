"""Planning Agent - Decomposes complex requests into step‑by‑step actions"""

import json
import re
from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


class PlanningAgent:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    async def create_plan(self, user_message: str) -> Optional[List[Dict[str, Any]]]:
        """Return a list of steps (JSON objects) if multiple actions; else None."""
        print("[Planner] Analysing user request...")
        prompt = f"""
You are a planning assistant for a calendar agent. Break the user request into a sequence of simple actions.
Return a JSON array of steps. Each step must be a JSON object with keys:
- action: one of "create_event", "delete_event", "check_availability", "list_events", "reschedule_event"
- title: event title (if applicable)
- start_time: ISO datetime with timezone (e.g., 2026-05-15T15:00:00+05:30)
- end_time: same format (default 1 hour after start)
- description, location (optional)
- event_id (for delete_event)

If only one action, return an empty list or null.
If multiple actions, return an array in execution order.

User message: {user_message}
Return ONLY valid JSON. No extra text.
"""
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        content = response.content.strip()
        json_match = re.search(r'```json\n?(.*?)\n?```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
        try:
            steps = json.loads(content)
            if isinstance(steps, list) and len(steps) > 1:
                print(f"[Planner] Created plan with {len(steps)} steps.")
                for i, step in enumerate(steps, 1):
                    action = step.get("action", "unknown")
                    title = step.get("title", "")
                    print(f"  {i}. {action} {('('+title+')' if title else '')}")
                return steps
        except:
            pass
        print("[Planner] Single action detected – no plan needed.")
        return None

    def step_to_message(self, step: Dict[str, Any]) -> str:
        """Convert a step dict into a simple natural language message."""
        action = step.get("action")
        if action == "create_event":
            title = step.get("title", "")
            start = step.get("start_time", "")
            end = step.get("end_time", "")
            return f"Create event '{title}' from {start} to {end}."
        elif action == "check_availability":
            start = step.get("start_time", "")
            end = step.get("end_time", "")
            return f"Check if I'm free from {start} to {end}."
        elif action == "delete_event":
            title = step.get("title", "")
            return f"Delete the event '{title}'."
        elif action == "list_events":
            return "List my upcoming events."
        elif action == "reschedule_event":
            title = step.get("title", "")
            new_start = step.get("start_time", "")
            new_end = step.get("end_time", "")
            return f"Reschedule '{title}' to {new_start} – {new_end}."
        else:
            return "Unknown action."

    def fill_placeholders(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Replace placeholders like {{free_slot_start}} with values from context."""
        new_step = step.copy()
        for key, value in new_step.items():
            if isinstance(value, str):
                for placeholder, replacement in context.items():
                    # Convert any non‑string replacement to string
                    value = value.replace(f"{{{{{placeholder}}}}}", str(replacement))
                new_step[key] = value
        return new_step

    async def update_context(self, step: Dict[str, Any], response: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to extract new information from the response and update context."""
        prompt = f"""
You are helping a calendar agent execute a plan. You just performed:
Step: {step}
Got the response: {response}
Current context (key‑value pairs from previous steps): {context}
Extract any new information (e.g., free time slots, event IDs, etc.) and return a JSON object with the updates.
If a step is a check_availability, its result will be a free time slot. A later create_event step should use the placeholders {{free_slot_start}} and {{free_slot_end}} in its start_time and end_time.
Return only the JSON object, no extra text.
"""
        llm_response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        try:
            updates = json.loads(llm_response.content)
            if isinstance(updates, dict):
                context.update(updates)
                print("[Planner] Extracted and added to context:", updates)
        except:
            # optional regex fallback for common patterns
            import re
            match = re.search(r"FREE from (.*?) to (.*?)\.", response)
            if match:
                context["free_slot_start"] = match.group(1)
                context["free_slot_end"] = match.group(2)
                print("[Planner] (regex fallback) Extracted free slot:", match.group(1))
        return context

    def combine_results(self, results: List[str]) -> str:
        if not results:
            return "No actions were performed."
        if len(results) == 1:
            return results[0]
        combined = "I performed the following actions:\n"
        for i, res in enumerate(results, 1):
            combined += f"{i}. {res}\n"
        return combined