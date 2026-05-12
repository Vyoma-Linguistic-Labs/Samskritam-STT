# Google Sheets Integration for Samskritam-STT Analytics

This guide explains how to set up Google Sheets integration for persistent analytics storage on Streamlit Cloud.

## Why Google Sheets?

- ✅ **Persistent**: Data survives app restarts
- ✅ **Cloud-based**: Works on Streamlit Cloud
- ✅ **Real-time**: Accessible from any device
- ✅ **Easy**: No database setup required
- ✅ **Shareable**: Easy to share reports with team
- ✅ **Queryable**: Can use Google Sheets formulas for analysis

## Step 1: Create a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project (or use existing):
   - Click the project dropdown at top
   - Click "NEW PROJECT"
   - Name it "Samskritam-STT"
   - Click CREATE

3. Enable required APIs:
   - Go to **Enabled APIs & services**
   - Click **+ ENABLE APIS AND SERVICES**
   - Search for "Google Sheets API" → Click it → Click ENABLE
   - Search for "Google Drive API" → Click it → Click ENABLE

## Step 2: Create a Service Account

1. Go to **Credentials** (in Cloud Console left sidebar)
2. Click **+ CREATE CREDENTIALS** → Select **Service Account**
3. Fill in:
   - Service account name: `samskritam-stt-analytics`
   - Click CREATE AND CONTINUE
4. Grant roles:
   - Click the dropdown under "Grant this service account access to project"
   - Search for and select: `Editor`
   - Click CONTINUE then DONE

## Step 3: Create and Download Service Account Key

1. In Cloud Console, go to **Credentials**
2. Under "Service Accounts", click the email you just created
3. Go to the **KEYS** tab
4. Click **Add Key** → **Create new key**
5. Choose **JSON** → Click CREATE
6. A JSON file downloads automatically - **save this securely**

**⚠️ IMPORTANT**: This file contains credentials. Never commit it to GitHub or share publicly!

## Step 4: Configure Streamlit Cloud Secrets

1. Go to your app on [Streamlit Cloud](https://share.streamlit.io)
2. Click the **menu** (⋮) → **Settings**
3. Go to **Secrets** tab
4. Copy the entire JSON file content you downloaded
5. Paste it in the secrets editor with this format:

```toml
[GOOGLE_SHEETS_CREDENTIALS]
type = "service_account"
project_id = "your-project-id"
private_key_id = "your-key-id"
private_key = "-----BEGIN PRIVATE KEY-----\nYOUR_KEY_HERE\n-----END PRIVATE KEY-----\n"
client_email = "your-service-account@your-project.iam.gserviceaccount.com"
client_id = "your-client-id"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "your-cert-url"
```

6. Click SAVE

## Step 5: Local Development Setup

For local testing, create `.streamlit/secrets.toml` in your project:

```toml
[GOOGLE_SHEETS_CREDENTIALS]
type = "service_account"
project_id = "your-project-id"
private_key_id = "your-key-id"
private_key = "-----BEGIN PRIVATE KEY-----\nYOUR_KEY_HERE\n-----END PRIVATE KEY-----\n"
client_email = "your-service-account@your-project.iam.gserviceaccount.com"
client_id = "your-client-id"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "your-cert-url"
```

⚠️ **Add `.streamlit/secrets.toml` to `.gitignore`** to prevent accidental commits.

## Step 6: Install Required Dependencies

```bash
pip install gspread google-auth-oauthlib google-auth-httplib2
```

Update your `requirements.txt`:

```
streamlit
torch
gspread
google-auth-oauthlib
google-auth-httplib2
google-auth>=2.0.0
gspread>=5.0.0
```

## Step 7: Share the Google Sheet (Optional)

After the first run, a Google Sheet named "Samskritam-STT Analytics" will be created. You can:

1. Share it with team members
2. View it at https://sheets.google.com
3. Create custom dashboards with formulas
4. Export data as needed

## Usage

### In Your Streamlit App

The app will automatically log to Google Sheets if credentials are configured:

```python
from google_sheets_analytics import GoogleSheetsAnalytics

# Already handled in ASR_Streamlit.py
# Just make sure to set user_id if you want per-user tracking:
# st.session_state.user_id = "user@example.com"  # Or any unique identifier
```

### Generate Monthly Reports

```bash
# Generate HTML report for current month
python generate_monthly_report.py

# Generate report for specific month
python generate_monthly_report.py --year 2026 --month 5

# Generate in multiple formats
python generate_monthly_report.py --format all --output-dir ./reports

# Generate for specific month with all formats
python generate_monthly_report.py --year 2026 --month 5 --format all
```

**Supported formats:**
- `html` - Beautiful HTML dashboard (default)
- `json` - Raw JSON data
- `csv` - Excel-compatible format

## User Identification

To track individual users, set the user_id in your app:

```python
import streamlit as st
import os

# Get user ID from environment or session
user_id = os.getenv("USER_ID", st.session_state.get("user_id", "anonymous"))
st.session_state.user_id = user_id
```

On Streamlit Cloud, add to **Secrets**:
```toml
USER_ID = "user@example.com"
```

Or pass via URL:
```
https://your-app.streamlit.app/?user_id=user@example.com
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'gspread'"
```bash
pip install gspread google-auth-oauthlib
```

### "Credentials not found"
- Verify `GOOGLE_SHEETS_CREDENTIALS` is in `.streamlit/secrets.toml` (local)
- Verify secrets are set in Streamlit Cloud Settings → Secrets

### "Permission denied" on Google Sheets
- Ensure the service account has "Editor" role on the project
- Check that the sheet exists and is accessible

### "SpreadsheetNotFound"
- The sheet is created automatically on first event
- Check that the service account can create files in Google Drive

## API Limits

Google Sheets API has free tier limits:
- 500 requests per 100 seconds per project
- 500 requests per 100 seconds per user

For most use cases, this is plenty. If you hit limits, consider:
- Batch writing events (currently writes one at a time)
- Using Google BigQuery for high-volume analytics

## Security Notes

1. **Never commit credentials to GitHub**
2. **Use `.gitignore` for `.streamlit/secrets.toml`**
3. **Rotate service account keys periodically**
4. **Use environment variables for sensitive data**
5. **Consider IP whitelisting for sensitive deployments**

## What Gets Logged

Each event includes:
- Timestamp
- Session ID
- User ID
- Event type (upload, transcription, error, etc.)
- Event-specific data (audio duration, processing time, etc.)

### Event Types

- `session_start` - User starts the app
- `file_uploaded` - User uploads an audio file
- `recording_captured` - User records audio
- `transcription_complete` - Transcription finished
- `preprocessing_error` - Audio preprocessing failed
- `transcription_error` - Transcription failed
- `recording_processing_error` - Recording processing failed

## Example Custom Queries in Google Sheets

Once data is in Google Sheets, you can create powerful dashboards:

```
=COUNTIF(B:B, "session_start")  // Total sessions

=COUNTIFS(D:D, "transcription_complete", E:E, "upload")  // Upload transcriptions

=AVERAGEIF(D:D, "transcription_complete", H:H)  // Avg audio duration

=FILTER(G:G, D:D = "transcription_complete")  // All transcription times
```

## Next Steps

1. ✅ Set up Google Cloud project
2. ✅ Create service account and download credentials
3. ✅ Add credentials to Streamlit Cloud secrets
4. ✅ Install required packages
5. ✅ Deploy app - analytics will start logging
6. ✅ Generate monthly reports with `generate_monthly_report.py`

For questions or issues, check the Streamlit documentation or Google Sheets API docs.
