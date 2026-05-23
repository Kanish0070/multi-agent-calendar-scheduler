# Calendar Agent – Multi‑Agent Scheduling System

A hierarchical agent system that orchestrates specialized sub‑agents to handle calendar operations, multi‑step planning, conflict resolution, and resilient fallback strategies. Built with Python, LangChain, and Google Calendar API.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📌 Overview

The Calendar Agent is not a monolithic script. It is a **supervisor‑orchestrated multi‑agent system** where the main orchestrator dynamically spawns specialised agents on‑demand. This design allows complex requests to be decomposed, executed step‑by‑step, and recovered from failures gracefully.

## 🧠 Architecture

The system consists of a central **Orchestrator (Supervisor)** that maintains conversation state, routes messages, and manages the lifecycle of several sub‑agents:

- **Planning Agent** – decomposes complex user requests into executable plans.
- **Worker Agent** – directly interacts with Google Calendar API.
- **Validator Agent** – checks availability before event creation (spawned on‑demand).
- **Final Verifier Agent** – always runs after any operation to double‑check results.
- **Cloud Fallback Agent** – handles edge cases, failures, and escalations.

The orchestrator also supports **multi‑step plan execution** with context sharing and conflict detection, as well as **single‑action** handling.

*(Architecture image can be generated separately.)*

## 🧩 Key Agentic AI Aspects

### 🔹 State Management

- **Conversation History** – The orchestrator maintains a full list of `HumanMessage` and `AIMessage` objects, used for context when routing or parsing.
- **Agent‑Local State** – Each spawned agent keeps its own short‑lived context (e.g., `failure_context` in `CloudFallbackAgent`).
- **Plan Execution State** – For multi‑step plans, the orchestrator stores:
  - `pending_plan`: the list of steps.
  - `pending_plan_context`: a key‑value dictionary that accumulates data across steps (e.g., free slot start/end times, event IDs).
  - `pending_plan_results`: the responses from completed steps.
  - `pending_plan_step_index`: the current step (0‑based) when a plan is paused.
- **Negotiation State** – When a conflict occurs, the orchestrator enters a negotiation state (`self.negotiation_event` and `self.last_free_slots`) and waits for user input.

### 🔹 Planning & Decomposition (with Step Tracking)

The **Planning Agent** is responsible for breaking a natural language request into a sequence of atomic actions.

- **Input** – User message (e.g., “Check my Friday afternoon and then schedule a meeting with John in the first free slot”).
- **Process** – The LLM is prompted to return a JSON array of steps. Each step contains:
  - `action`: one of `create_event`, `delete_event`, `check_availability`, `list_events`, `reschedule_event`.
  - `title`, `start_time`, `end_time`, `description`, `location`, `event_id` (as needed).
- **Placeholders** – Steps may include placeholders like `{{free_slot_start}}` and `{{free_slot_end}}`. These are replaced at runtime using the shared `pending_plan_context`.
- **Step Tracking** – The orchestrator executes steps sequentially. After each step, it calls `planning_agent.update_context()` to extract new information (e.g., free time slots, event IDs) and update the context dictionary. This ensures that later steps automatically receive the output of earlier steps.
- **Pause & Resume** – If a step fails (e.g., a time conflict), the orchestrator pauses the plan, stores the entire execution state, and enters a negotiation sub‑dialogue. After the user resolves the conflict, the plan resumes from the exact step where it stopped.

### 🔹 Orchestration & Control Flow (Collision Detection)

The orchestrator’s routing logic distinguishes between **single‑action** and **multi‑step** requests.

#### Single‑action flow
1. **Analyse** – `analyze_message_content()` determines if the message mentions tools, availability, conflicts, completion, etc.
2. **Route** – Based on analysis, the orchestrator decides:
   - `"tools"` → directly delegate to the worker.
   - `"tools_with_validator"` → spawn a `ValidatorAgent` to check availability first.
   - `"output_parser"` → summarise recent history.
   - `"reasoner"` → use the LLM for uncertainty handling.
3. **Execute** – The selected routing action produces a response.

#### Multi‑step flow (with collision detection)
1. **Plan** – `planning_agent.create_plan()` returns a list of steps.
2. **Loop** – For each step:
   - Replace placeholders using current context.
   - Convert step to a natural language message.
   - Call `_process_single_action()` on that message.
   - If the step is a `create_event` and a conflict is detected, the worker returns a special response starting with `"⚠️ Conflict detected"`.
3. **On conflict** – The orchestrator stores all pending plan state (`pending_plan`, `pending_plan_context`, `pending_plan_results`, `pending_plan_step_index`) and enters a **negotiation loop**. The user can:
   - Force create despite conflict (reply `9`).
   - See alternative free slots (reply `2`).
   - Cancel the plan (reply `3`).
4. **Resume** – After the user resolves the conflict, the orchestrator resumes executing from the exact step index, re‑running availability checks if necessary.
5. **Completion** – All steps are combined into a final user‑friendly message via `planning_agent.combine_results()`.

#### Collision detection in detail
- **Single step** – When `create_event` detects a busy time slot, the worker returns a conflict message and stores the event details in `self.negotiation_event`. The orchestrator returns this message to the user and waits for a response.
- **Multi‑step** – The same conflict mechanism triggers the plan pause. The orchestrator does **not** commit any later steps until the conflict is resolved or the plan is cancelled.

### 🔹 Guardrails & Schema Validation

- **Input parsing** – The orchestrator uses both regex and LLM‑based extraction (`parse_event_details`, `_parse_availability_request`) to validate that dates, times, and actions conform to expected schemas. If parsing fails, it falls back to keyword extraction.
- **Agent spawning limits** – Agents are spawned on‑demand and immediately cleaned up after use to prevent resource leaks. The `cleanup_agent()` method removes them from the agent pool and active list.
- **Verification agent** – The `FinalVerifierAgent` is **always** spawned before returning any result. It runs automated checks (e.g., verifying that an event ID exists) and LLM‑based verification, and only approves the operation if confidence is high and no critical warnings exist.
- **Fallback agent** – The `CloudFallbackAgent` provides structured recovery suggestions and escalates to human intervention when the system cannot resolve an issue.

### 🔹 Compute Tiering & Resiliency

- **On‑demand spawning** – Agents are not pre‑loaded. The validator, fallback, and verifier agents are instantiated only when needed, reducing memory and compute overhead.
- **Fallback on LLM failure** – If the LLM fails to parse an intent, the worker falls back to a keyword‑based extraction method (`_extract_intent`) that uses regex and `dateparser` to salvage the request.
- **Conflict recovery** – The system never blindly creates conflicting events. It offers alternatives (free slots found via `find_free_slots`) and lets the user decide, preventing irreversible mistakes.
- **Stateless API interactions** – The worker’s calendar API calls are asynchronous and stateless. If a call fails (e.g., network error), the `CloudFallbackAgent` can be spawned to analyse the failure and suggest retries or workarounds.
- **Human escalation** – When the fallback agent determines that an issue is too complex (e.g., ambiguous user intent or API permission errors), it raises a flag for human
