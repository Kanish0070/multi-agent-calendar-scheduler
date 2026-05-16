"""Worker Agent - Executes calendar tool calls on behalf of the orchestrator"""

from typing import Dict, Any, List, Optional
from ..tools import CalendarAPI
from datetime import datetime, timedelta


class WorkerAgent:
    """Responsible for all direct Google Calendar API operations."""

    def __init__(self):
        self.calendar_api = CalendarAPI()

    async def create_event(self, title: str, start_time: str, end_time: str,
                           description: str = "", location: str = "", attendees: list = None) -> Dict[str, Any]:
        return await self.calendar_api.create_event(title, start_time, end_time, description, location, attendees)

    async def delete_event(self, event_id: str) -> Dict[str, Any]:
        return await self.calendar_api.delete_event(event_id)

    async def delete_event_by_title(self, title: str, days_ahead: int = 7) -> Dict[str, Any]:
        return await self.calendar_api.delete_event_by_title(title, days_ahead)

    async def check_availability(self, start_time: str, end_time: str) -> Dict[str, Any]:
        return await self.calendar_api.check_availability(start_time, end_time)

    async def list_events(self, page_size: int = 10, include_deleted: bool = False) -> List[Dict[str, Any]]:
        return await self.calendar_api.list_events(page_size, include_deleted)

    async def update_event(self, event_id: str, start_time: str, end_time: str) -> Dict[str, Any]:
        # You need to add update_event to CalendarAPI first
        return await self.calendar_api.update_event(event_id, start_time, end_time)

    async def find_free_slots(self, start_dt: datetime, end_dt: datetime, duration_minutes: int = 60) -> List[Dict]:
        return await self.calendar_api.find_free_slots(start_dt, end_dt, duration_minutes)

    async def find_events_by_title(self, title: str, days_ahead: int = 7) -> List[Dict]:
        return await self.calendar_api.find_events_by_title(title, days_ahead)