import os
from typing import Optional, Dict, Any, List
from google.oauth2 import service_account
from googleapiclient.discovery import build
from datetime import datetime, timedelta


class CalendarAPI:
    def __init__(self, credentials_path: Optional[str] = None, calendar_id: Optional[str] = None):
        if credentials_path is None:
            credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if not credentials_path:
                raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
        self.credentials_path = credentials_path

        if calendar_id is None:
            calendar_id = os.getenv("GOOGLE_CALENDAR_ID", "primary")
        self.calendar_id = calendar_id

        self.service = self._build_service()

    def _build_service(self):
        creds = service_account.Credentials.from_service_account_file(
            self.credentials_path,
            scopes=["https://www.googleapis.com/auth/calendar"]
        )
        return build("calendar", "v3", credentials=creds)

    async def create_event(self, title: str, start_time: str, end_time: str,
                           description: str = "", location: str = "", attendees: list = None) -> Dict[str, Any]:
        event = {
            "summary": title,
            "description": description,
            "location": location,
            "start": {"dateTime": start_time},
            "end": {"dateTime": end_time},
        }
        if attendees:
            event["attendees"] = [{"email": email} for email in attendees]

        print(f"DEBUG: Event body being sent: {event}")
        try:
            created = self.service.events().insert(calendarId=self.calendar_id, body=event).execute()
            return {"event_id": created["id"], "html_link": created["htmlLink"]}
        except Exception as e:
            print(f"ERROR: {e}")
            if hasattr(e, 'resp') and hasattr(e, 'content'):
                print(f"Response content: {e.content}")
            raise

    async def delete_event(self, event_id: str) -> Dict[str, Any]:
        self.service.events().delete(calendarId=self.calendar_id, eventId=event_id).execute()
        return {"success": True}

    async def find_events_by_title(self, title: str, days_ahead: int = 7) -> List[Dict]:
        now = datetime.now().astimezone()
        time_min = now.isoformat()
        time_max = (now + timedelta(days=days_ahead)).isoformat()
        events_result = self.service.events().list(
            calendarId=self.calendar_id,
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy="startTime"
        ).execute()
        items = events_result.get("items", [])
        title_lower = title.lower()
        matches = [e for e in items if title_lower in e.get("summary", "").lower()]
        return matches

    async def delete_event_by_title(self, title: str, days_ahead: int = 7) -> Dict[str, Any]:
        matches = await self.find_events_by_title(title, days_ahead)
        if not matches:
            return {"success": False, "message": f"No event found with title containing '{title}'"}
        event = matches[0]
        event_id = event["id"]
        self.service.events().delete(calendarId=self.calendar_id, eventId=event_id).execute()
        return {"success": True, "deleted_event": event}

    async def update_event(self, event_id: str, start_time: str, end_time: str) -> Dict[str, Any]:
        """
        Update an existing event's start and end times.
        """
        event = {
            "start": {"dateTime": start_time},
            "end": {"dateTime": end_time},
        }
        updated = self.service.events().patch(
            calendarId=self.calendar_id,
            eventId=event_id,
            body=event
        ).execute()
        return {"success": True, "event_id": updated["id"], "html_link": updated["htmlLink"]}

    async def check_availability(self, start_time: str, end_time: str) -> Dict[str, Any]:
        start_time = self._ensure_tz(start_time);
        end_time = self._ensure_tz(end_time);        
        events = self.service.events().list(
            calendarId=self.calendar_id,
            timeMin=start_time,
            timeMax=end_time,
            singleEvents=True,
            orderBy="startTime"
        ).execute()
        busy = len(events.get("items", [])) > 0
        return {"available": not busy, "conflicts": events.get("items", [])}

    async def find_free_slots(self, start_dt: datetime, end_dt: datetime, duration_minutes: int = 60) -> List[Dict]:
        """
        Find free time slots of exactly `duration_minutes` minutes within the date range.
        Returns list of dicts with 'start' and 'end' (ISO format with timezone).
        """
        # Fetch all events in the range
        events = self.service.events().list(
            calendarId=self.calendar_id,
            timeMin=start_dt.isoformat(),
            timeMax=end_dt.isoformat(),
            singleEvents=True,
            orderBy="startTime"
        ).execute()
        busy = []
        for e in events.get("items", []):
            start = e["start"].get("dateTime")
            end = e["end"].get("dateTime")
            if start and end:
                busy.append((datetime.fromisoformat(start), datetime.fromisoformat(end)))
        busy.sort(key=lambda x: x[0])

        current = start_dt
        slots = []
        # We want at most 3 slots
        for b_start, b_end in busy:
            if b_start > current:
                # Free interval from current to b_start
                free_start = current
                free_end = b_start
                # Slide a window of duration_minutes across this interval
                window_start = free_start
                while window_start + timedelta(minutes=duration_minutes) <= free_end and len(slots) < 3:
                    window_end = window_start + timedelta(minutes=duration_minutes)
                    slots.append({
                        "start": window_start.isoformat(),
                        "end": window_end.isoformat()
                    })
                    window_start += timedelta(minutes=duration_minutes)
            current = max(current, b_end)
        # After the last busy event
        if current < end_dt:
            window_start = current
            while window_start + timedelta(minutes=duration_minutes) <= end_dt and len(slots) < 3:
                window_end = window_start + timedelta(minutes=duration_minutes)
                slots.append({
                    "start": window_start.isoformat(),
                    "end": window_end.isoformat()
                })
                window_start += timedelta(minutes=duration_minutes)
        # Filter out zero‑duration slots (shouldn't happen, but safe)
        slots = [s for s in slots if s["start"] != s["end"]]
        return slots[:3]

    async def list_events(self, page_size: int = 10, include_deleted: bool = False) -> List[Dict[str, Any]]:
        events = self.service.events().list(
            calendarId=self.calendar_id,
            maxResults=page_size,
            orderBy="startTime",
            singleEvents=True,
            showDeleted=include_deleted
        ).execute()
        return events.get("items", [])
    
    async def update_event(self, event_id: str, start_time: str, end_time: str) -> Dict[str, Any]:
        event = {
        "start": {"dateTime": start_time},
        "end": {"dateTime": end_time},
    }
        updated = self.service.events().patch(
        calendarId=self.calendar_id,
        eventId=event_id,
        body=event
    ).execute()
        return {"success": True, "event_id": updated["id"], "html_link": updated["htmlLink"]}
    
    def _ensure_tz(self, dt_str: str) -> str:
        if dt_str and not any(c in dt_str for c in ('Z', '+', '-')):
            dt_str += "+05:30"
        return dt_str