#!/usr/bin/env python3
"""
Generate monthly analytics reports from Google Sheets.
Run this script at the end of each month to aggregate analytics across all users.

Usage:
    python generate_monthly_report.py --year 2026 --month 5
    python generate_monthly_report.py  # Uses current year/month
"""

import argparse
import json
import csv
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any, List

# This script can run standalone with credentials file or via environment
try:
    from google_sheets_analytics import GoogleSheetsAnalytics, AnalyticsReporter
    GOOGLE_SHEETS_AVAILABLE = True
except ImportError:
    GOOGLE_SHEETS_AVAILABLE = False
    print("Warning: google_sheets_analytics not available. Install required dependencies.")


def save_report_json(report: Dict[str, Any], year: int, month: int, output_dir: str = "reports"):
    """Save report as JSON file"""
    Path(output_dir).mkdir(exist_ok=True)
    filename = f"{output_dir}/analytics_report_{year:04d}_{month:02d}.json"
    with open(filename, "w") as f:
        json.dump(report, f, indent=2)
    return filename


def save_report_csv(report: Dict[str, Any], year: int, month: int, output_dir: str = "reports"):
    """Save report as CSV for Excel import"""
    Path(output_dir).mkdir(exist_ok=True)
    filename = f"{output_dir}/analytics_report_{year:04d}_{month:02d}.csv"
    
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        
        # Flatten report into rows
        for key, value in report.items():
            if isinstance(value, dict):
                writer.writerow([f"{key}:"])
                for k, v in value.items():
                    if isinstance(v, dict):
                        writer.writerow([f"  {k}:"])
                        for k2, v2 in v.items():
                            writer.writerow([f"    {k2}", v2])
                    else:
                        writer.writerow([f"  {k}", v])
            else:
                writer.writerow([key, value])
    
    return filename


def generate_html_report(report: Dict[str, Any], year: int, month: int, output_dir: str = "reports") -> str:
    """Generate a nice HTML report"""
    Path(output_dir).mkdir(exist_ok=True)
    filename = f"{output_dir}/analytics_report_{year:04d}_{month:02d}.html"
    
    month_name = datetime(year, month, 1).strftime("%B")
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Analytics Report - {month_name} {year}</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 20px;
                color: #333;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1000px;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #34495e;
                margin-top: 30px;
                border-left: 4px solid #3498db;
                padding-left: 10px;
            }}
            .metric {{
                display: inline-block;
                background: #ecf0f1;
                padding: 15px 20px;
                margin: 10px 10px 10px 0;
                border-radius: 4px;
                border-left: 4px solid #3498db;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
            }}
            .metric-label {{
                font-size: 12px;
                color: #7f8c8d;
                text-transform: uppercase;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th {{
                background-color: #3498db;
                color: white;
                padding: 12px;
                text-align: left;
            }}
            td {{
                padding: 10px 12px;
                border-bottom: 1px solid #ecf0f1;
            }}
            tr:hover {{
                background-color: #f9f9f9;
            }}
            .error {{
                color: #e74c3c;
            }}
            .success {{
                color: #27ae60;
            }}
            .footer {{
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #ecf0f1;
                font-size: 12px;
                color: #7f8c8d;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Samskritam-STT Analytics Report</h1>
            <p><strong>{month_name} {year}</strong></p>
            
            <h2>Overview</h2>
            <div class="metric">
                <div class="metric-label">Total Events</div>
                <div class="metric-value">{report.get('total_events', 0)}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Unique Sessions</div>
                <div class="metric-value">{report.get('unique_sessions', 0)}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Unique Users</div>
                <div class="metric-value">{report.get('unique_users', 0)}</div>
            </div>
    """
    
    # Event types section
    if "event_types" in report:
        html += "<h2>Event Types</h2><table><tr><th>Event Type</th><th>Count</th></tr>"
        for event_type, count in report["event_types"].items():
            html += f"<tr><td>{event_type}</td><td>{count}</td></tr>"
        html += "</table>"
    
    # Transcription metrics
    if "transcriptions" in report:
        trans = report["transcriptions"]
        html += f"""
        <h2>Transcription Metrics</h2>
        <div class="metric">
            <div class="metric-label">Transcriptions</div>
            <div class="metric-value">{trans.get('count', 0)}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Total Audio Duration</div>
            <div class="metric-value">{trans.get('total_audio_duration', 0)}s</div>
        </div>
        <div class="metric">
            <div class="metric-label">Avg Processing Time</div>
            <div class="metric-value">{trans.get('avg_processing_time', 0):.2f}s</div>
        </div>
        <div class="metric">
            <div class="metric-label">Avg Audio Duration</div>
            <div class="metric-value">{trans.get('avg_audio_duration', 0):.2f}s</div>
        </div>
        """
    
    # Error tracking
    if "errors" in report and report["errors"]["total_errors"] > 0:
        errors = report["errors"]
        html += f"""
        <h2 class="error">⚠️ Error Summary</h2>
        <div class="metric error">
            <div class="metric-label">Total Errors</div>
            <div class="metric-value">{errors.get('total_errors', 0)}</div>
        </div>
        <h3>Error Types</h3>
        <table><tr><th>Error Type</th><th>Count</th></tr>
        """
        for error_type, count in errors.get("error_types", {}).items():
            html += f"<tr><td>{error_type}</td><td>{count}</td></tr>"
        html += "</table>"
    
    # Input modes
    if "input_modes" in report:
        modes = report["input_modes"]
        html += f"""
        <h2>Input Modes</h2>
        <div class="metric">
            <div class="metric-label">File Uploads</div>
            <div class="metric-value">{modes.get('uploads', 0)}</div>
        </div>
        <div class="metric">
            <div class="metric-label">Recordings</div>
            <div class="metric-value">{modes.get('recordings', 0)}</div>
        </div>
        """
    
    html += f"""
        <div class="footer">
            <p>Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Samskritam-STT Analytics | Vyoma Linguistic Labs</p>
        </div>
        </div>
    </body>
    </html>
    """
    
    with open(filename, "w") as f:
        f.write(html)
    
    return filename


def main():
    parser = argparse.ArgumentParser(description="Generate monthly analytics reports")
    parser.add_argument(
        "--year", 
        type=int, 
        default=date.today().year,
        help="Year (default: current year)"
    )
    parser.add_argument(
        "--month", 
        type=int, 
        default=date.today().month,
        help="Month 1-12 (default: current month)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Output directory for reports (default: reports/)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "csv", "html", "all"],
        default="html",
        help="Report format(s) (default: html)"
    )
    parser.add_argument(
        "--credentials",
        type=str,
        help="Path to Google service account JSON credentials"
    )
    
    args = parser.parse_args()
    
    if not GOOGLE_SHEETS_AVAILABLE:
        print("Error: google_sheets_analytics module not found")
        print("Make sure you have the google_sheets_analytics.py file in the same directory")
        return 1
    
    print(f"Generating analytics report for {args.year:04d}-{args.month:02d}...")
    
    try:
        # Initialize analytics client
        analytics = GoogleSheetsAnalytics()
        reporter = AnalyticsReporter(analytics)
        
        # Generate report
        report = reporter.generate_monthly_report(args.year, args.month)
        
        if "message" in report:
            print(f"⚠️  {report['message']}")
            return 0
        
        # Save in requested formats
        formats = [args.format] if args.format != "all" else ["json", "csv", "html"]
        saved_files = []
        
        for fmt in formats:
            if fmt == "json":
                filename = save_report_json(report, args.year, args.month, args.output_dir)
                saved_files.append(filename)
                print(f"✅ JSON report: {filename}")
            elif fmt == "csv":
                filename = save_report_csv(report, args.year, args.month, args.output_dir)
                saved_files.append(filename)
                print(f"✅ CSV report: {filename}")
            elif fmt == "html":
                filename = generate_html_report(report, args.year, args.month, args.output_dir)
                saved_files.append(filename)
                print(f"✅ HTML report: {filename}")
        
        # Print summary
        print("\n" + "="*50)
        print("📊 MONTHLY ANALYTICS SUMMARY")
        print("="*50)
        print(f"Total Events: {report.get('total_events', 0)}")
        print(f"Unique Sessions: {report.get('unique_sessions', 0)}")
        print(f"Unique Users: {report.get('unique_users', 0)}")
        
        if "transcriptions" in report:
            trans = report["transcriptions"]
            print(f"\n🎙️  Transcriptions: {trans.get('count', 0)}")
            print(f"   Total Audio Duration: {trans.get('total_audio_duration', 0):.2f}s")
            print(f"   Avg Processing Time: {trans.get('avg_processing_time', 0):.2f}s")
        
        if "errors" in report:
            print(f"\n⚠️  Total Errors: {report['errors'].get('total_errors', 0)}")
        
        print("="*50)
        
        return 0
        
    except Exception as e:
        print(f"❌ Error generating report: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
