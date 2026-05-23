"""Supervisor Agent - Spawns sub-agents on-demand"""

import asyncio
from typing import Optional, Dict, Any, List
from .types import AgentMessage, AgentInfo
from .agents import ValidatorAgent, CloudFallbackAgent, FinalVerifierAgent, WorkerAgent, PlanningAgent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
import re
from datetime import datetime, timedelta
import calendar
import dateparser
import pytz
import json

def format_datetime_12hr(dt_iso: str) -> str:
    dt = datetime.fromisoformat(dt_iso)
    return dt.strftime("%A, %B %d at %I:%M %p").lstrip("0").replace(" 0", " ")

def analyze_message_content(message: str) -> Dict[str, Any]:
    analysis = {
        "has_tool_calls": False,
        "mentions_availability": False,
        "mentions_conflicts": False,
        "indicates_completion": False,
        "user_done": False,
        "shows_uncertainty": False,
        "high_confidence": False,
        "recommended_action": None,
        "agents_to_spawn": []
    }
    lower_message = message.lower()
    tool_keywords = ["create event", "delete event", "schedule", "book",
                     "set meeting", "add appointment", "remove"]
    if any(kw in lower_message for kw in tool_keywords):
        analysis["has_tool_calls"] = True
    availability_keywords = ["available", "free", "conflict",
                             "schedule", "when can you", "can i", "do you have"]
    if any(kw in lower_message for kw in availability_keywords):
        analysis["mentions_availability"] = True
    conflict_keywords = ["conflict", "double book", "already have", "can't",
                         "not available", "busy", "overlap"]
    if any(kw in lower_message for kw in conflict_keywords):
        analysis["mentions_conflicts"] = True
    completion_keywords = ["created", "deleted", "scheduled", "done",
                           "finished", "completed", "set up", "arranged"]
    if any(kw in lower_message for kw in completion_keywords):
        analysis["indicates_completion"] = True
    done_keywords = ["done", "quit", "exit", "stop", "all set", "thank you",
                     "that's all", "i'm good", "no more"]
    if any(kw in lower_message for kw in done_keywords):
        analysis["user_done"] = True
    uncertainty_keywords = ["i'm not sure", "i think", "maybe", "perhaps",
                            "i don't know", "unclear", "confused", "help me"]
    if any(kw in lower_message for kw in uncertainty_keywords):
        analysis["shows_uncertainty"] = True
    routing_priority = [
        ("tools", analysis["has_tool_calls"]),
        ("tools_with_validator", analysis["mentions_availability"] or analysis["mentions_conflicts"]),
        ("output_parser", analysis["indicates_completion"]),
        ("output_parser", analysis["user_done"]),
        ("output_parser", analysis["high_confidence"]),
        ("reasoner", analysis["shows_uncertainty"]),
        ("tools", True)
    ]
    for action, condition in routing_priority:
        if condition:
            analysis["recommended_action"] = action
            break
    if not analysis["recommended_action"]:
        analysis["recommended_action"] = "tools"
    return analysis

class CalendarOrchestrator:
    """Supervisor Agent – coordinates sub‑agents and executes plans."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.llm = ChatOpenAI(
            base_url="http://127.0.0.1:8082/v1",
            api_key="not-needed",
            model="agentscope-ai_CoPaw-Flash-9B-Q4_K_M.gguf",
            temperature=0.7,
        )
        self.planner = PlanningAgent(self.llm)
        self.worker = WorkerAgent()
        self.agent_pool: Dict[str, Any] = {}
        self.conversation_history: List[BaseMessage] = []
        self.active_agents: List[str] = []
        self.system_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are the Supervisor Agent for a Calendar system...""")
        ])
        self.negotiation_event: Optional[Dict[str, Any]] = None
        self.last_free_slots: List[Dict] = []

        # Multi‑step plan pause state
        self.pending_plan: Optional[List[Dict]] = None
        self.pending_plan_context: Optional[Dict] = None
        self.pending_plan_results: Optional[List[str]] = None
        self.pending_plan_step_index: Optional[int] = None
        self.planning_paused: bool = False

    async def parse_event_details(self, message: str) -> dict:
        prompt = f"""
Extract calendar event details from the user message. Return a JSON object with:
- action: one of ["create_event", "delete_event", "check_availability", "list_events", "reschedule_event"]
- title: event title (string)
- start_time: ISO format datetime with timezone (e.g., 2026-05-15T15:00:00+05:30)
- end_time: same format (default 1 hour after start if not specified)
- description: optional
- location: optional
- event_id: (only for delete_event if provided)

User message: {message}
"""
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        json_str = re.sub(r'```json\n?(.*?)\n?```', r'\1', response.content, flags=re.DOTALL)
        return json.loads(json_str)

    # ---------- hybrid date parsing ----------
    async def _parse_availability_request(self, message: str):
        start, end = self._regex_parse_availability(message)
        if start and end:
            return start, end
        start, end = await self._llm_parse_availability(message)
        return start, end

    def _regex_parse_availability(self, message: str):
        after_match = re.search(r'after\s+([\d\w\s]+?)(?:\s+from\s+)?(\d{1,2}(?::\d{2})?\s*(?:am|pm))?\s*(?:-|to)\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm))?', message, re.IGNORECASE)
        if after_match:
            date_part = after_match.group(1).strip()
            start_time_str = after_match.group(2)
            end_time_str = after_match.group(3)
            base_date = dateparser.parse(date_part, settings={'PREFER_DATES_FROM': 'future'})
            if base_date:
                target_date = base_date + timedelta(days=1)
                if start_time_str and end_time_str:
                    start_dt = dateparser.parse(start_time_str, settings={'PREFER_DATES_FROM': 'future'})
                    end_dt = dateparser.parse(end_time_str, settings={'PREFER_DATES_FROM': 'future'})
                    if start_dt and end_dt:
                        start_combined = target_date.replace(hour=start_dt.hour, minute=start_dt.minute)
                        end_combined = target_date.replace(hour=end_dt.hour, minute=end_dt.minute)
                        if end_combined <= start_combined:
                            end_combined += timedelta(days=1)
                        return start_combined.isoformat(), end_combined.isoformat()
                else:
                    return target_date.replace(hour=0, minute=0).isoformat(), target_date.replace(hour=23, minute=59).isoformat()

        date_pattern = r'(?:on\s+)?(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(\w+)\s+(\d{4})'
        date_match = re.search(date_pattern, message, re.IGNORECASE)
        if date_match:
            day, month, year = date_match.groups()
            try:
                month_num = int(month)
            except ValueError:
                month_num = None
                for i, name in enumerate(calendar.month_abbr):
                    if name.lower() == month.lower()[:3]:
                        month_num = i
                        break
                if month_num is None:
                    return None, None
            try:
                dt = datetime(int(year), month_num, int(day))
                time_pattern = r'(?:from\s+)?(\d{1,2}(?::\d{2})?\s*(?:am|pm))?\s*(?:-|to)\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm))'
                time_match = re.search(time_pattern, message, re.IGNORECASE)
                if time_match and time_match.group(1) and time_match.group(2):
                    start_dt = dateparser.parse(time_match.group(1), settings={'PREFER_DATES_FROM': 'future'})
                    end_dt = dateparser.parse(time_match.group(2), settings={'PREFER_DATES_FROM': 'future'})
                    if start_dt and end_dt:
                        start_combined = dt.replace(hour=start_dt.hour, minute=start_dt.minute)
                        end_combined = dt.replace(hour=end_dt.hour, minute=end_dt.minute)
                        if end_combined <= start_combined:
                            end_combined += timedelta(days=1)
                        return start_combined.isoformat(), end_combined.isoformat()
                return dt.replace(hour=0, minute=0).isoformat(), dt.replace(hour=23, minute=59).isoformat()
            except:
                pass
        return None, None

    async def _llm_parse_availability(self, message: str):
        prompt = f"""
You are a date/time parser. From the user message, extract the intended start and end times for a calendar availability check.
Return ONLY a JSON object with keys "start_time" and "end_time" in ISO 8601 format WITHOUT timezone (e.g., "2026-05-22T14:00:00").
If only a single date is given (whole day), use 00:00 to 23:59.
If the user says "after X", add one day to X.
If no time range is specified, assume the whole day.
Message: {message}
"""
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        try:
            data = json.loads(response.content)
            start = data.get("start_time")
            end = data.get("end_time")
            if start and end:
                return start, end
        except:
            pass
        return None, None

    # ---------- sub‑agent spawning ----------
    def spawn_validator_agent(self) -> ValidatorAgent:
        agent = ValidatorAgent(agent_id=f"validator-{len(self.agent_pool)}")
        self.agent_pool[agent.agent_id] = agent
        self.active_agents.append(agent.agent_id)
        print(f"[Supervisor] Spawning Availability Check Agent: {agent.agent_id}")
        return agent

    def spawn_fallback_agent(self) -> CloudFallbackAgent:
        agent = CloudFallbackAgent(agent_id=f"fallback-{len(self.agent_pool)}")
        self.agent_pool[agent.agent_id] = agent
        self.active_agents.append(agent.agent_id)
        print(f"[Supervisor] Spawning Cloud Fallback Agent: {agent.agent_id}")
        return agent

    def spawn_verifier_agent(self) -> FinalVerifierAgent:
        agent = FinalVerifierAgent(agent_id=f"verifier-{len(self.agent_pool)}")
        self.agent_pool[agent.agent_id] = agent
        self.active_agents.append(agent.agent_id)
        print(f"[Supervisor] Spawning Task Verification Agent: {agent.agent_id}")
        return agent

    def cleanup_agent(self, agent_id: str) -> None:
        if agent_id in self.agent_pool:
            print(f"[Supervisor] Cleaning up agent: {agent_id}")
            del self.agent_pool[agent_id]
            if agent_id in self.active_agents:
                self.active_agents.remove(agent_id)

    async def _process_single_action(self, message: str) -> str:
        analysis = analyze_message_content(message)
        if analysis["mentions_availability"] or analysis["mentions_conflicts"]:
            self.spawn_validator_agent()
        self.spawn_verifier_agent()
        routing_decision = self._route_message(message, analysis)
        result = await self._execute_routing(message, routing_decision, analysis)
        for aid in self.active_agents.copy():
            self.cleanup_agent(aid)
        return result.get("response", "No response")

    async def process_message(self, user_message: str) -> Dict[str, Any]:
        print(f"[Supervisor] Received: {user_message}")

        # ----- 1. Handle ongoing conflict negotiation (single action) -----
        if self.negotiation_event is not None:
            lower_msg = user_message.lower().strip()
            if self.last_free_slots and lower_msg.isdigit():
                idx = int(lower_msg) - 1
                if 0 <= idx < len(self.last_free_slots):
                    chosen = self.last_free_slots[idx]
                    ev = self.negotiation_event
                    print(f"[Supervisor] User chose alternative slot {idx+1}")
                    avail = await self.worker.check_availability(chosen["start"], chosen["end"])
                    if not avail.get("available"):
                        response = "Sorry, that slot is no longer available. Please request alternatives again."
                        self._add_to_history(user_message, response)
                        self._cleanup_active_agents()
                        return {"response": response, "routing_action": "negotiation", "analysis": {}}
                    result = await self.worker.create_event(
                        title=ev["title"],
                        start_time=chosen["start"],
                        end_time=chosen["end"],
                        description=ev.get("description", "")
                    )
                    self.negotiation_event = None
                    self.last_free_slots = []
                    response = f"✅ Event '{ev['title']}' rescheduled to {format_datetime_12hr(chosen['start'])} – {format_datetime_12hr(chosen['end'])}. (Event ID: {result.get('event_id', 'unknown')})"
                    self._add_to_history(user_message, response)
                    self._cleanup_active_agents()
                    # Resume pending plan if any
                    if self.planning_paused and self.pending_plan is not None:
                        return await self._resume_plan(user_message, response)
                    return {"response": response, "routing_action": "pending_handled", "analysis": {}}
                else:
                    response = f"Invalid choice. Please reply with a number from 1 to {len(self.last_free_slots)}."
                    self._add_to_history(user_message, response)
                    self._cleanup_active_agents()
                    return {"response": response, "routing_action": "negotiation", "analysis": {}}
            if lower_msg in ["9", "force", "yes", "y"]:
                ev = self.negotiation_event
                result = await self.worker.create_event(
                    title=ev["title"],
                    start_time=ev["start_time"],
                    end_time=ev["end_time"],
                    description=ev.get("description", "")
                )
                self.negotiation_event = None
                self.last_free_slots = []
                response = f"✅ Event '{ev['title']}' created despite conflict. (Event ID: {result.get('event_id', 'unknown')})"
                self._add_to_history(user_message, response)
                self._cleanup_active_agents()
                if self.planning_paused and self.pending_plan is not None:
                    return await self._resume_plan(user_message, response)
                return {"response": response, "routing_action": "pending_handled", "analysis": {}}
            elif lower_msg in ["2", "alternatives", "suggest", "alternative times"]:
                ev = self.negotiation_event
                start_dt = datetime.fromisoformat(ev["start_time"])
                end_dt = datetime.fromisoformat(ev["end_time"])
                duration = int((end_dt - start_dt).total_seconds() / 60)
                search_start = start_dt - timedelta(days=2)
                search_end = end_dt + timedelta(days=2)
                free_slots = await self.worker.find_free_slots(search_start, search_end, duration)
                if not free_slots:
                    response = "❌ No free slots found near that time. You can force create (reply '9') or cancel (reply '3')."
                    self.last_free_slots = []
                else:
                    self.last_free_slots = free_slots
                    lines = []
                    for i, slot in enumerate(free_slots, 1):
                        start_str = format_datetime_12hr(slot["start"])
                        end_str = format_datetime_12hr(slot["end"])
                        lines.append(f"{i}. {start_str} – {end_str}")
                    response = "Here are some free slots near your requested time:\n" + "\n".join(lines) + "\n\nReply with the number to pick, or '9' to force create anyway, or '3' to cancel."
                self._add_to_history(user_message, response)
                self._cleanup_active_agents()
                return {"response": response, "routing_action": "negotiation", "analysis": {}}
            elif lower_msg in ["3", "cancel", "no", "n"]:
                self.negotiation_event = None
                self.last_free_slots = []
                response = "❌ Event creation cancelled."
                self._add_to_history(user_message, response)
                self._cleanup_active_agents()
                if self.planning_paused and self.pending_plan is not None:
                    # Cancel the entire plan
                    self.pending_plan = None
                    self.pending_plan_context = None
                    self.pending_plan_results = None
                    self.pending_plan_step_index = None
                    self.planning_paused = False
                    final_response = "The plan was cancelled due to conflict."
                    self.conversation_history.append(AIMessage(content=final_response))
                    return {"response": final_response, "routing_action": "plan_cancelled", "analysis": {}}
                return {"response": response, "routing_action": "pending_handled", "analysis": {}}
            else:
                response = "I didn't understand. Please reply with:\n- '9' to create anyway\n- '2' to see alternative times\n- '3' to cancel"
                self._add_to_history(user_message, response)
                self._cleanup_active_agents()
                return {"response": response, "routing_action": "negotiation", "analysis": {}}

        # ----- 2. Handle paused multi‑step plan (waiting for user response to conflict) -----
        if self.planning_paused and self.pending_plan is not None:
            lower_msg = user_message.lower().strip()
            if lower_msg in ["9", "force", "yes", "y"]:
                ev = self.negotiation_event
                if ev:
                    result = await self.worker.create_event(
                        title=ev["title"],
                        start_time=ev["start_time"],
                        end_time=ev["end_time"],
                        description=ev.get("description", "")
                    )
                    self.negotiation_event = None
                    self.last_free_slots = []
                    step_response = f"✅ Event '{ev['title']}' created despite conflict. (Event ID: {result.get('event_id', 'unknown')})"
                    # Replace the conflicted step's result, not append
                    self.pending_plan_results[self.pending_plan_step_index] = step_response
                    # Update context
                    context_update = await self.planner.update_context(self.pending_plan[self.pending_plan_step_index], step_response, self.pending_plan_context)
                    self.pending_plan_context.update(context_update)
                    # Move to next step
                    self.pending_plan_step_index += 1
                    if self.pending_plan_step_index >= len(self.pending_plan):
                        final_response = self.planner.combine_results(self.pending_plan_results)
                        self.conversation_history.append(AIMessage(content=final_response))
                        self._cleanup_active_agents()
                        self.pending_plan = None
                        self.pending_plan_context = None
                        self.pending_plan_results = None
                        self.pending_plan_step_index = None
                        self.planning_paused = False
                        return {"response": final_response, "routing_action": "multi_step_completed", "analysis": {}}
                    else:
                        return await self._resume_plan(user_message, step_response)
                else:
                    return await self._resume_plan(user_message, "Force created (no conflict data)")
            elif lower_msg in ["2", "alternatives", "suggest", "alternative times"]:
                ev = self.negotiation_event
                if ev:
                    start_dt = datetime.fromisoformat(ev["start_time"])
                    end_dt = datetime.fromisoformat(ev["end_time"])
                    duration = int((end_dt - start_dt).total_seconds() / 60)
                    search_start = start_dt - timedelta(days=2)
                    search_end = end_dt + timedelta(days=2)
                    free_slots = await self.worker.find_free_slots(search_start, search_end, duration)
                    if not free_slots:
                        response = "❌ No free slots found near that time. You can force create (reply '9') or cancel (reply '3')."
                        self.last_free_slots = []
                    else:
                        self.last_free_slots = free_slots
                        lines = []
                        for i, slot in enumerate(free_slots, 1):
                            start_str = format_datetime_12hr(slot["start"])
                            end_str = format_datetime_12hr(slot["end"])
                            lines.append(f"{i}. {start_str} – {end_str}")
                        response = "Here are some free slots near your requested time:\n" + "\n".join(lines) + "\n\nReply with the number to pick, or '9' to force create anyway, or '3' to cancel."
                    self._add_to_history(user_message, response)
                    self._cleanup_active_agents()
                    return {"response": response, "routing_action": "negotiation", "analysis": {}}
                else:
                    return await self._resume_plan(user_message, "No conflict data")
            elif lower_msg in ["3", "cancel", "no", "n"]:
                self.negotiation_event = None
                self.last_free_slots = []
                response = "❌ Event creation cancelled. The plan will stop."
                self._add_to_history(user_message, response)
                self._cleanup_active_agents()
                self.pending_plan = None
                self.pending_plan_context = None
                self.pending_plan_results = None
                self.pending_plan_step_index = None
                self.planning_paused = False
                return {"response": response, "routing_action": "plan_cancelled", "analysis": {}}
            else:
                if lower_msg.isdigit() and self.last_free_slots:
                    idx = int(lower_msg) - 1
                    if 0 <= idx < len(self.last_free_slots):
                        chosen = self.last_free_slots[idx]
                        ev = self.negotiation_event
                        if ev:
                            print(f"[Supervisor] User chose alternative slot {idx+1} for plan step")
                            avail = await self.worker.check_availability(chosen["start"], chosen["end"])
                            if not avail.get("available"):
                                response = "Sorry, that slot is no longer available. Please request alternatives again."
                                self._add_to_history(user_message, response)
                                self._cleanup_active_agents()
                                return {"response": response, "routing_action": "negotiation", "analysis": {}}
                            result = await self.worker.create_event(
                                title=ev["title"],
                                start_time=chosen["start"],
                                end_time=chosen["end"],
                                description=ev.get("description", "")
                            )
                            self.negotiation_event = None
                            self.last_free_slots = []
                            step_response = f"✅ Event '{ev['title']}' rescheduled to {format_datetime_12hr(chosen['start'])} – {format_datetime_12hr(chosen['end'])}. (Event ID: {result.get('event_id', 'unknown')})"
                            # Replace the conflicted step's result, not append
                            self.pending_plan_results[self.pending_plan_step_index] = step_response
                            context_update = await self.planner.update_context(self.pending_plan[self.pending_plan_step_index], step_response, self.pending_plan_context)
                            self.pending_plan_context.update(context_update)
                            self.pending_plan_step_index += 1
                            if self.pending_plan_step_index >= len(self.pending_plan):
                                final_response = self.planner.combine_results(self.pending_plan_results)
                                self.conversation_history.append(AIMessage(content=final_response))
                                self._cleanup_active_agents()
                                self.pending_plan = None
                                self.pending_plan_context = None
                                self.pending_plan_results = None
                                self.pending_plan_step_index = None
                                self.planning_paused = False
                                return {"response": final_response, "routing_action": "multi_step_completed", "analysis": {}}
                            else:
                                return await self._resume_plan(user_message, step_response)
                        else:
                            return await self._resume_plan(user_message, "Slot chosen but no conflict event")
                    else:
                        response = f"Invalid choice. Please reply with a number from 1 to {len(self.last_free_slots)}."
                        self._add_to_history(user_message, response)
                        self._cleanup_active_agents()
                        return {"response": response, "routing_action": "negotiation", "analysis": {}}
                else:
                    response = "I didn't understand. Please reply with:\n- '9' to create anyway\n- '2' to see alternative times\n- '3' to cancel"
                    self._add_to_history(user_message, response)
                    self._cleanup_active_agents()
                    return {"response": response, "routing_action": "negotiation", "analysis": {}}

        # ----- 3. Multi‑step planning (new request) -----
        plan = await self.planner.create_plan(user_message)
        if plan and len(plan) > 1:
            print(f"[Supervisor] Multi‑step request detected. Executing plan with {len(plan)} steps.")
            self.conversation_history.append(HumanMessage(content=user_message))
            results = []
            context = {}
            conflict_occurred = False
            for i, step in enumerate(plan, 1):
                print(f"[Supervisor] Step {i}/{len(plan)}: {step.get('action')}")
                resolved_step = self.planner.fill_placeholders(step, context)
                step_message = self.planner.step_to_message(resolved_step)
                step_response = await self._process_single_action(step_message)
                results.append(step_response)
                if step_response.startswith("⚠️ Conflict detected"):
                    print(f"[Supervisor] Conflict detected in step {i}, pausing plan.")
                    self.pending_plan = plan
                    self.pending_plan_context = context
                    self.pending_plan_results = results
                    self.pending_plan_step_index = i - 1  # store 0‑based index
                    self.planning_paused = True
                    conflict_occurred = True
                    break
                context = await self.planner.update_context(step, step_response, context)
                print(f"[Supervisor] Step {i} completed.")
            if conflict_occurred:
                self._add_to_history(user_message, step_response)
                self._cleanup_active_agents()
                return {"response": step_response, "routing_action": "multi_step_conflict", "analysis": {}}
            final_response = self.planner.combine_results(results)
            self.conversation_history.append(AIMessage(content=final_response))
            self._cleanup_active_agents()
            return {"response": final_response, "routing_action": "multi_step", "analysis": {}}

        # ----- 4. Single action -----
        print("[Supervisor] Single action detected. Proceeding normally.")
        self.conversation_history.append(HumanMessage(content=user_message))
        analysis = analyze_message_content(user_message)
        if analysis["mentions_availability"] or analysis["mentions_conflicts"]:
            self.spawn_validator_agent()
        self.spawn_verifier_agent()
        routing_decision = self._route_message(user_message, analysis)
        result = await self._execute_routing(user_message, routing_decision, analysis)
        self.conversation_history.append(AIMessage(content=result.get("response", "")))
        self._cleanup_active_agents()
        return result

    async def _resume_plan(self, user_message: str, last_step_response: str) -> Dict[str, Any]:
        print(f"[Supervisor] Resuming plan from step {self.pending_plan_step_index+1} / {len(self.pending_plan)}")
        results = self.pending_plan_results
        context = self.pending_plan_context
        plan = self.pending_plan
        i = self.pending_plan_step_index
        while i < len(plan):
            step = plan[i]
            i_show = i + 1
            print(f"[Supervisor] Resuming step {i_show}/{len(plan)}: {step.get('action')}")
            resolved_step = self.planner.fill_placeholders(step, context)
            step_message = self.planner.step_to_message(resolved_step)
            step_response = await self._process_single_action(step_message)
            results.append(step_response)
            if step_response.startswith("⚠️ Conflict detected"):
                print(f"[Supervisor] Conflict detected again in step {i_show}, pausing plan again.")
                self.pending_plan_results = results
                self.pending_plan_context = context
                self.pending_plan_step_index = i  # store 0‑based index
                self.planning_paused = True
                return {"response": step_response, "routing_action": "multi_step_conflict", "analysis": {}}
            context = await self.planner.update_context(step, step_response, context)
            print(f"[Supervisor] Step {i_show} completed.")
            i += 1
        final_response = self.planner.combine_results(results)
        self.conversation_history.append(AIMessage(content=final_response))
        self._cleanup_active_agents()
        self.pending_plan = None
        self.pending_plan_context = None
        self.pending_plan_results = None
        self.pending_plan_step_index = None
        self.planning_paused = False
        return {"response": final_response, "routing_action": "multi_step_completed", "analysis": {}}

    # ---------- helper methods (unchanged) ----------
    def _add_to_history(self, user_msg: str, agent_msg: str):
        self.conversation_history.append(HumanMessage(content=user_msg))
        self.conversation_history.append(AIMessage(content=agent_msg))

    def _cleanup_active_agents(self):
        for agent_id in self.active_agents.copy():
            self.cleanup_agent(agent_id)

    def _route_message(self, message: str, analysis: Dict[str, Any]) -> str:
        if analysis["has_tool_calls"]:
            return "tools"
        elif analysis["mentions_availability"] or analysis["mentions_conflicts"]:
            return "tools_with_validator"
        elif analysis["indicates_completion"]:
            return "output_parser"
        elif analysis["user_done"]:
            return "output_parser"
        elif analysis["shows_uncertainty"]:
            return "reasoner"
        else:
            return "tools"

    async def _execute_routing(self, message: str, routing_action: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        result = {"routing_action": routing_action, "analysis": analysis, "response": ""}
        if routing_action == "tools":
            result["response"] = await self._delegate_to_worker(message)
        elif routing_action == "tools_with_validator":
            validator = self.spawn_validator_agent()
            start_time, end_time = await self._parse_availability_request(message)
            if start_time and end_time:
                # --- TIMEZONE NORMALIZATION PATCH ---
                ist_offset = "+05:30"
                if "+" not in start_time and "Z" not in start_time:
                    start_time = start_time.strip() + ist_offset
                if "+" not in end_time and "Z" not in end_time:
                    end_time = end_time.strip() + ist_offset
                # ------------------------------------

                availability_result = await validator.check_availability(start_time, end_time)
                result["response"] = f"Availability: {availability_result}"
            else:
                result["response"] = "I couldn't understand the date. Please specify a clear date (e.g., '16th of May 2026')."
        elif routing_action == "output_parser":
            result["response"] = await self._output_parser(message)
        elif routing_action == "reasoner":
            response = await self.llm.ainvoke(self.system_prompt.format(message))
            result["response"] = response.content
        return result

    async def _delegate_to_worker(self, message: str) -> str:
        try:
            intent = await self.parse_event_details(message)
            print(f"[Worker] Action: {intent.get('action')} | Title: {intent.get('title', '')}")
            if not intent.get("action"):
                raise ValueError("No action in LLM response")
        except Exception as e:
            print(f"[Worker] LLM parsing failed: {e}, falling back to keyword extraction")
            intent = await self._extract_intent(message)

        action = intent.get("action")
        if action == "create_event":
            start = intent.get("start_time", "")
            end = intent.get("end_time", "")
            title = intent.get("title", "Untitled")
            avail = await self.worker.check_availability(start, end)
            if avail.get("available"):
                result = await self.worker.create_event(
                    title=title,
                    start_time=start,
                    end_time=end,
                    description=intent.get("description", "")
                )
                return f"✅ Event '{title}' created successfully. (Event ID: {result.get('event_id', 'unknown')})"
            else:
                conflicts = avail.get("conflicts", [])
                conflict_summary = ", ".join([f"'{c.get('summary', 'Unknown')}' at {c.get('start', {}).get('dateTime', '?')}" for c in conflicts[:3]])
                self.negotiation_event = {
                    "title": title,
                    "start_time": start,
                    "end_time": end,
                    "description": intent.get("description", "")
                }
                self.last_free_slots = []
                return f"⚠️ Conflict detected with {conflict_summary}. Would you like to:\n9. Force create anyway\n2. See alternative times\n3. Cancel\n(Reply with number or keyword)"

        elif action == "reschedule_event":
            title = intent.get("title")
            new_start = intent.get("start_time")
            new_end = intent.get("end_time")
            if not title or not new_start or not new_end:
                return "❌ Missing title or new time for rescheduling."
            matches = await self.worker.find_events_by_title(title, days_ahead=30)
            if not matches:
                return f"❌ No event found with title containing '{title}'."
            event_id = matches[0]["id"]
            result = await self.worker.update_event(event_id, new_start, new_end)
            if result.get("success"):
                return f"✅ Event '{title}' rescheduled to {format_datetime_12hr(new_start)} – {format_datetime_12hr(new_end)}."
            else:
                return f"❌ Failed to reschedule event: {result.get('error', 'Unknown error')}"

        elif action == "delete_event":
            event_id = intent.get("event_id")
            title = intent.get("title")
            if event_id and len(event_id) > 10:
                await self.worker.delete_event(event_id)
                return f"🗑️ Event deleted successfully. (ID: {event_id})"
            elif title:
                res = await self.worker.delete_event_by_title(title)
                if res.get("success"):
                    deleted = res["deleted_event"]
                    return f"🗑️ Deleted event: '{deleted.get('summary', 'Untitled')}' (ID: {deleted.get('id')})"
                else:
                    return f"❌ {res.get('message')}"
            else:
                return "❌ Cannot delete: no event ID or title provided."

        elif action == "check_availability":
            start = intent.get("start_time", "")
            end = intent.get("end_time", "")
            result = await self.worker.check_availability(start, end)
            if result.get("available"):
                return f"✅ The calendar shows you are FREE from {start} to {end}."
            else:
                return f"⚠️ The calendar shows you are BUSY during that time. Conflicts: {result.get('conflicts', [])}"

        elif action == "list_events":
            events = await self.worker.list_events()
            if not events:
                return "📅 No upcoming events found."
            lines = ["📅 **Upcoming events:**"]
            for ev in events[:10]:
                start_iso = ev.get("start", {}).get("dateTime", "")
                start_str = format_datetime_12hr(start_iso) if start_iso else "unknown time"
                summary = ev.get("summary", "Untitled")
                lines.append(f"- {summary} at {start_str}")
            return "\n".join(lines)

        else:
            return f"❌ No matching tool found for action: {action}"

    # ---------- keyword extraction fallback ----------
    async def _extract_intent(self, message: str) -> Dict[str, Any]:
        intent = {
            "action": None,
            "title": "",
            "start_time": "",
            "end_time": "",
            "event_id": "",
            "description": ""
        }
        lower = message.lower()
        if any(kw in lower for kw in ["shift", "reschedule", "move", "change time", "change the time"]):
            intent["action"] = "reschedule_event"
            match = re.search(r"(?:shift|reschedule|move|change\s+time)\s+(?:the\s+)?(?:event\s+)?(?:meeting\s+)?(.*?)\s+(?:to|at)\s+(.*)", message, re.IGNORECASE)
            if match:
                intent["title"] = match.group(1).strip()
                time_str = match.group(2).strip()
                dates = dateparser.search.search_dates(time_str)
                if dates:
                    start_dt = dates[0][1]
                    if start_dt.tzinfo is None:
                        ist = pytz.timezone("Asia/Kolkata")
                        start_dt = ist.localize(start_dt)
                    intent["start_time"] = start_dt.isoformat()
                    intent["end_time"] = (start_dt + timedelta(hours=1)).isoformat()
            return intent
        if any(kw in lower for kw in ["create event", "schedule", "book"]):
            intent["action"] = "create_event"
            match = re.search(r"(?:schedule|create event|book)\s+(.+?)(?:\s+(?:tomorrow|today|at|on)|$)", message, re.IGNORECASE)
            if match:
                intent["title"] = match.group(1).strip()
            dates = dateparser.search.search_dates(message)
            if dates:
                start_dt = dates[0][1]
                if start_dt.tzinfo is None:
                    ist = pytz.timezone("Asia/Kolkata")
                    start_dt = ist.localize(start_dt)
                intent["start_time"] = start_dt.isoformat()
                intent["end_time"] = (start_dt + timedelta(hours=1)).isoformat()
            return intent
        if any(kw in lower for kw in ["delete", "remove", "cancel"]):
            intent["action"] = "delete_event"
            match = re.search(r"(?:delete|remove|cancel)\s+(?:the\s+)?(?:event\s+)?(?:meeting\s+)?(.*)", message, re.IGNORECASE)
            if match:
                intent["title"] = match.group(1).strip()
            return intent
        if any(kw in lower for kw in ["check availability", "available", "free"]):
            intent["action"] = "check_availability"
            start, end = await self._parse_availability_request(message)
            if start:
                intent["start_time"] = start
                intent["end_time"] = end
            return intent
        if any(kw in lower for kw in ["list", "show events"]):
            intent["action"] = "list_events"
            return intent
        return intent

    async def _output_parser(self, message: str) -> str:
        recent_history = self.conversation_history[-3:]
        prompt = f"""
        User message: {message}
        Conversation history: {str(recent_history)}
        Summarize the result and provide a user-friendly response.
        """
        response = await self.llm.ainvoke(prompt)
        return response.content

    # ---------- getters ----------
    def get_conversation_history(self) -> List[BaseMessage]:
        return self.conversation_history.copy()

    def clear_history(self) -> None:
        self.conversation_history.clear()

    def get_active_agents(self) -> List[str]:
        return self.active_agents.copy()

    def get_agent_pool_info(self) -> Dict[str, str]:
        info = {}
        for agent_id, agent in self.agent_pool.items():
            if hasattr(agent, 'get_agent_info'):
                info[agent_id] = agent.get_agent_info()
            else:
                info[agent_id] = {"type": "unknown", "purpose": "Unknown purpose"}
        return info
