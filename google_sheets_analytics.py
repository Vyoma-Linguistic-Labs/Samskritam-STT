"""
Google Sheets integration for analytics logging and reporting.
Handles writing events to Google Sheets and retrieving data for analysis.
"""

import json
import streamlit as st
from datetime import datetime
from typing import Dict, List, Optional, Any
try:
    import gspread
    from google.oauth2.service_account import Credentials
except ImportError:
    pass

class GoogleSheetsAnalytics:
    """Manages analytics logging to Google Sheets"""
    
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 
              'https://www.googleapis.com/auth/drive']
    
    def __init__(self, spreadsheet_name: str = "Samskritam-STT Analytics"):
        """
        Initialize Google Sheets analytics client.
        
        Args:
            spreadsheet_name: Name of the Google Sheet to create/use
        """
        self.spreadsheet_name = spreadsheet_name
        self.client = self._authenticate()
        self.spreadsheet = self._get_or_create_spreadsheet()
        self.worksheet = self._get_or_create_worksheet("events")
        self.session_id = self._generate_session_id()
    
    @staticmethod
    def _generate_session_id() -> str:
        """Generate unique session ID"""
        import time
        return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(time.time())) % 10000}"
    
    def _authenticate(self) -> gspread.Client:
        """
        Authenticate with Google Sheets API using Streamlit secrets.
        
        Returns:
            gspread.Client: Authenticated client
            
        Raises:
            ValueError: If GOOGLE_SHEETS_CREDENTIALS not in secrets
        """
        try:
            # Get credentials from Streamlit secrets
            credentials_dict = st.secrets["GOOGLE_SHEETS_CREDENTIALS"]
            credentials = Credentials.from_service_account_info(
                credentials_dict,
                scopes=self.SCOPES
            )
            return gspread.authorize(credentials)
        except KeyError:
            raise ValueError(
                "Missing GOOGLE_SHEETS_CREDENTIALS in Streamlit secrets. "
                "Please add your Google service account credentials to .streamlit/secrets.toml"
            )
    
    def _get_or_create_spreadsheet(self) -> gspread.Spreadsheet:
        """Get existing spreadsheet or create new one"""
        try:
            # Try to find existing spreadsheet
            spreadsheet = self.client.open(self.spreadsheet_name)
            return spreadsheet
        except gspread.SpreadsheetNotFound:
            # Create new spreadsheet
            spreadsheet = self.client.create(self.spreadsheet_name)
            # Share with service account email (optional: add more permissions here)
            return spreadsheet
    
    def _get_or_create_worksheet(self, title: str) -> gspread.Worksheet:
        """Get existing worksheet or create new one"""
        try:
            worksheet = self.spreadsheet.worksheet(title)
            return worksheet
        except gspread.WorksheetNotFound:
            # Create new worksheet with headers
            worksheet = self.spreadsheet.add_worksheet(title=title, rows=10000, cols=20)
            self._add_headers(worksheet)
            return worksheet
    
    def _add_headers(self, worksheet: gspread.Worksheet) -> None:
        """Add headers to worksheet if empty"""
        if worksheet.cell(1, 1).value is None:
            headers = [
                "timestamp",
                "session_id",
                "event_type",
                "input_mode",
                "filename",
                "file_size_bytes",
                "file_extension",
                "audio_duration",
                "transcription_time",
                "transcription_length",
                "word_count",
                "sample_rate",
                "error_type",
                "error_message",
                "user_id",
                "metadata"
            ]
            worksheet.insert_row(headers, 1)
    
    def log_event(self, event_type: str, user_id: Optional[str] = None, **kwargs) -> None:
        """
        Log an event to Google Sheets.
        
        Args:
            event_type: Type of event (e.g., "file_uploaded", "transcription_complete")
            user_id: Optional user identifier
            **kwargs: Additional event data
        """
        try:
            row_data = [
                datetime.now().isoformat(),
                self.session_id,
                event_type,
                kwargs.get("input_mode", ""),
                kwargs.get("filename", ""),
                kwargs.get("file_size_bytes", ""),
                kwargs.get("file_extension", ""),
                kwargs.get("audio_duration", ""),
                kwargs.get("transcription_time", ""),
                kwargs.get("transcription_length", ""),
                kwargs.get("word_count", ""),
                kwargs.get("sample_rate", ""),
                kwargs.get("error_type", ""),
                kwargs.get("error_message", ""),
                user_id or "",
                json.dumps(kwargs)  # Store all data as backup
            ]
            
            self.worksheet.append_row(row_data)
        except Exception as e:
            # Fallback: print error but don't crash the app
            print(f"Error logging to Google Sheets: {str(e)}")
    
    def get_monthly_data(self, year: int, month: int) -> List[Dict]:
        """
        Retrieve all events for a specific month.
        
        Args:
            year: Year (e.g., 2026)
            month: Month (1-12)
            
        Returns:
            List of dictionaries containing event data
        """
        try:
            records = self.worksheet.get_all_records()
            
            monthly_events = []
            for record in records:
                try:
                    timestamp = datetime.fromisoformat(record.get("timestamp", ""))
                    if timestamp.year == year and timestamp.month == month:
                        monthly_events.append(record)
                except (ValueError, KeyError):
                    continue
            
            return monthly_events
        except Exception as e:
            print(f"Error retrieving monthly data: {str(e)}")
            return []
    
    def get_all_data(self) -> List[Dict]:
        """Retrieve all analytics data from Google Sheets"""
        try:
            return self.worksheet.get_all_records()
        except Exception as e:
            print(f"Error retrieving data: {str(e)}")
            return []


class AnalyticsReporter:
    """Generate reports from Google Sheets analytics data"""
    
    def __init__(self, analytics: GoogleSheetsAnalytics):
        """
        Initialize reporter.
        
        Args:
            analytics: GoogleSheetsAnalytics instance
        """
        self.analytics = analytics
    
    def generate_monthly_report(self, year: int, month: int) -> Dict[str, Any]:
        """
        Generate comprehensive monthly report.
        
        Args:
            year: Year
            month: Month (1-12)
            
        Returns:
            Dictionary with monthly statistics
        """
        events = self.analytics.get_monthly_data(year, month)
        
        if not events:
            return {"message": f"No data for {year}-{month:02d}"}
        
        # Calculate statistics
        report: Dict[str, Any] = {
            "year": year,
            "month": month,
            "total_events": len(events),
            "unique_sessions": len(set(e.get("session_id", "") for e in events)),
            "unique_users": len(set(e.get("user_id", "") for e in events if e.get("user_id"))),
        }
        
        # Event type breakdown
        event_types: Dict[str, int] = {}
        for event in events:
            event_type = event.get("event_type", "unknown")
            event_types[event_type] = event_types.get(event_type, 0) + 1
        report["event_types"] = event_types
        
        # Transcription metrics
        transcriptions = [e for e in events if e.get("event_type") == "transcription_complete"]
        if transcriptions:
            total_duration = sum(
                float(e.get("audio_duration", 0) or 0) for e in transcriptions
            )
            total_time = sum(
                float(e.get("transcription_time", 0) or 0) for e in transcriptions
            )
            report["transcriptions"] = {
                "count": len(transcriptions),
                "total_audio_duration": round(total_duration, 2),
                "total_processing_time": round(total_time, 2),
                "avg_audio_duration": round(total_duration / len(transcriptions), 2),
                "avg_processing_time": round(total_time / len(transcriptions), 2),
            }
        
        # Error tracking
        errors = [e for e in events if "error" in e.get("event_type", "").lower()]
        if errors:
            error_types = {}
            for error in errors:
                error_type = error.get("error_type", "unknown")
                error_types[error_type] = error_types.get(error_type, 0) + 1
            report["errors"] = {
                "total_errors": len(errors),
                "error_types": error_types
            }
        
        # Upload vs Recording breakdown
        uploads = [e for e in events if e.get("input_mode") == "upload"]
        recordings = [e for e in events if e.get("input_mode") == "recording"]
        report["input_modes"] = {
            "uploads": len(uploads),
            "recordings": len(recordings)
        }
        
        return report
    
    def generate_user_report(self, user_id: str, year: Optional[int] = None, month: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate report for specific user.
        
        Args:
            user_id: User identifier
            year: Optional year filter
            month: Optional month filter
            
        Returns:
            Dictionary with user statistics
        """
        all_data = self.analytics.get_all_data()
        user_events = [e for e in all_data if e.get("user_id") == user_id]
        
        if year and month:
            user_events = [
                e for e in user_events
                if datetime.fromisoformat(e.get("timestamp", "")).year == year
                and datetime.fromisoformat(e.get("timestamp", "")).month == month
            ]
        
        if not user_events:
            return {"message": f"No data for user {user_id}"}
        
        transcriptions = [e for e in user_events if e.get("event_type") == "transcription_complete"]
        
        report: Dict[str, Any] = {
            "user_id": user_id,
            "total_events": len(user_events),
            "transcriptions": len(transcriptions),
            "errors": len([e for e in user_events if "error" in e.get("event_type", "").lower()])
        }
        
        if transcriptions:
            total_duration = sum(
                float(e.get("audio_duration", 0) or 0) for e in transcriptions
            )
            report["total_audio_duration"] = round(total_duration, 2)
        
        return report


# Example usage for standalone script (not in Streamlit)
if __name__ == "__main__":
    # For testing without Streamlit
    print("This module is designed to be used within Streamlit or with proper Google API setup.")
    print("See SETUP.md for configuration instructions.")
