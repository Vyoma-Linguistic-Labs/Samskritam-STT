# Quick Start: Google Sheets Analytics

## TL;DR Setup (5 minutes)

### 1. Create Google Cloud Service Account
```
1. Go to https://console.cloud.google.com
2. Create new project → Name: "Samskritam-STT"
3. Enable APIs:
   - Google Sheets API
   - Google Drive API
4. Create Service Account:
   - Name: samskritam-stt-analytics
   - Role: Editor
5. Create Key (JSON) → Download and save securely
```

### 2. Add to Streamlit Cloud
```
1. Go to your app on Streamlit Cloud
2. Settings → Secrets
3. Paste the entire JSON file content as:

[GOOGLE_SHEETS_CREDENTIALS]
type = "service_account"
project_id = "..."
... (rest of JSON fields)
```

### 3. Install Packages
```bash
pip install -r requirements.txt
```

### 4. Deploy
Push to GitHub → Streamlit Cloud automatically deploys

## Monthly Reports

```bash
# Generate current month report
python generate_monthly_report.py

# Specific month
python generate_monthly_report.py --year 2026 --month 5

# All formats (HTML + JSON + CSV)
python generate_monthly_report.py --format all

# Output location
# Reports are saved in ./reports/ directory
```

## Key Features

✅ **Automatic** - App logs to Google Sheets automatically  
✅ **Persistent** - Data survives app restarts  
✅ **No Database** - Simple Google Sheets integration  
✅ **Monthly Reports** - One command generates dashboard  
✅ **Per-user Tracking** - Track analytics by user  
✅ **Real-time** - View data in Google Sheets anytime  

## File Structure

```
├── ASR_Streamlit.py              # Main app (updated to use Google Sheets)
├── google_sheets_analytics.py    # Google Sheets integration
├── generate_monthly_report.py    # Monthly report generator
├── GOOGLE_SHEETS_SETUP.md        # Full setup instructions
├── QUICK_START.md                # This file
├── requirements.txt              # Dependencies (updated)
└── reports/                      # Monthly reports (auto-created)
    ├── analytics_report_2026_05.html
    ├── analytics_report_2026_05.json
    └── analytics_report_2026_05.csv
```

## What Gets Tracked

📊 **Session Data**
- Session start/end
- User ID
- Session duration

🎙️ **Transcriptions**
- Audio duration
- Processing time
- Transcription length
- Word count
- Input mode (upload/recording)

⚠️ **Errors**
- Error types
- Error messages
- Timestamps

📁 **Files**
- Filename
- File size
- File format

## Fallback

If Google Sheets connection fails, the app automatically falls back to local `analytics.jsonl` file. Perfect for local development!

## Data Privacy

- Service account credentials stored securely in Streamlit Secrets
- Never committed to GitHub
- All data in your own Google Drive
- You control who has access

## Support

For detailed setup: See `GOOGLE_SHEETS_SETUP.md`

Common issues:
- Missing dependencies → `pip install -r requirements.txt`
- No credentials → Add to Streamlit Cloud Secrets
- Permission error → Check service account has Editor role
- Sheet not found → First event will create it automatically

## Next Month

At end of month, run:
```bash
python generate_monthly_report.py
```

Get beautiful HTML dashboard + JSON + CSV files. Share with your team!
