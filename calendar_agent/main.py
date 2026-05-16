"""
Calendar Agent - Hierarchical Agent Supervisor System

Main entry point for the calendar agent application.
"""
from dotenv import load_dotenv
load_dotenv(dotenv_path="D:\\Calender agent\\.env")
import asyncio
import sys
from typing import Optional


from .orchestrator import CalendarOrchestrator, analyze_message_content
from .agents import ValidatorAgent, CloudFallbackAgent, FinalVerifierAgent


async def main():
    """Main entry point for the calendar agent."""
    
    # Initialize orchestrator with MCP config if provided
    config_path = None
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    orchestrator = CalendarOrchestrator(config_path=config_path)
    
    print("=== Calendar Agent System Initialized ===")
    print("Available tools:")
    print("  - createEvent: Create a new calendar event")
    print("  - deleteEvent: Delete an existing calendar event")
    print("  - checkAvailability: Check calendar availability")
    print("  - listEvents: List calendar events")
    print("  - exit: to exit the agent")
    print("")
    print("Active agents:", orchestrator.get_active_agents())
    print("")
    
    try:
        # Interactive mode - process messages until quit
        while True:
            print("\n--- Calendar Agent ---")
            user_input = input("You: ").strip()
            
            if not user_input:
                continue

            elif user_input  == "exit":
                break
            
            # Process the message
            result = await orchestrator.process_message(user_input)
            
            # Print response
            print(f"\nAgent: {result['response']}")
            
            # Update active agents display
            print("Active agents:", orchestrator.get_active_agents())
            
    except KeyboardInterrupt:
        print("\n\nShutting down gracefully...")
        orchestrator.clear_history()
        # Clean up all agents
        for agent_id in orchestrator.get_active_agents():
            orchestrator.cleanup_agent(agent_id)
        print("All agents cleaned up.")
    
    return orchestrator


def demonstrate_routing():
    """
    Demonstrate the routing logic with sample messages.
    """
    sample_messages = [
        "Create a meeting with John tomorrow at 3 PM",
        "Check if I'm available on Friday",
        "I have a conflict at 4 PM", 
        "I've scheduled the meeting, thanks!",
        "Can you help me schedule something?",
        "I think I need to reschedule",
        "List my upcoming events"
    ]
    
    print("=== Routing Logic Demonstration ===")
    print()
    
    for message in sample_messages:
        analysis = analyze_message_content(message)
        print(f"Message: {message[:50]}...")
        print(f"  -> Routing: {analysis['recommended_action']}")
        print(f"  -> Agents needed: {analysis['agents_to_spawn']}")
        print()


if __name__ == "__main__":
    # Run demonstration or main application
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demonstrate_routing()
    else:
        asyncio.run(main())