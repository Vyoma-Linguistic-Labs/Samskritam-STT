import streamlit as st
import torch
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path
import tempfile
import torchaudio
import soundfile as sf
from inference import SpeechRecognizer
import time

# Try to import Google Sheets analytics, fall back to local file if not available
try:
    from google_sheets_analytics import GoogleSheetsAnalytics
    USE_GOOGLE_SHEETS = True
except ImportError:
    USE_GOOGLE_SHEETS = False

# ==================== ANALYTICS SETUP ====================

class StreamlitAnalytics:
    """Handles all analytics collection and logging (Google Sheets or local file)"""
    
    def __init__(self, use_google_sheets: bool = True, analytics_file: str = "analytics.jsonl"):
        self.use_google_sheets = use_google_sheets and USE_GOOGLE_SHEETS
        self.analytics_file = analytics_file
        self.session_id = st.session_state.get("session_id", self._generate_session_id())
        st.session_state.session_id = self.session_id
        
        # Get user_id from session or environment
        self.user_id = st.session_state.get("user_id", os.getenv("USER_ID", "anonymous"))
        
        # Initialize Google Sheets client if enabled
        if self.use_google_sheets:
            try:
                if USE_GOOGLE_SHEETS:
                    self.gs_analytics = GoogleSheetsAnalytics()
            except Exception as e:
                st.warning(f"Could not connect to Google Sheets: {str(e)}. Using local file instead.")
                self.use_google_sheets = False
    
    @staticmethod
    def _generate_session_id() -> str:
        """Generate unique session ID"""
        return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(time.time())) % 10000}"
    
    def log_event(self, event_type: str, **kwargs) -> None:
        """Log an analytics event to Google Sheets or local file"""
        if self.use_google_sheets:
            try:
                self.gs_analytics.log_event(event_type, user_id=self.user_id, **kwargs)
            except Exception as e:
                # Fallback to file if Google Sheets fails
                print(f"Google Sheets logging failed: {str(e)}, falling back to file")
                self._log_to_file(event_type, **kwargs)
        else:
            self._log_to_file(event_type, **kwargs)
    
    def _log_to_file(self, event_type: str, **kwargs) -> None:
        """Log event to local JSONL file (fallback)"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "user_id": self.user_id,
            "event_type": event_type,
            **kwargs
        }
        with open(self.analytics_file, "a") as f:
            f.write(json.dumps(event) + "\n")
    
    def get_session_stats(self) -> dict:
        """Get stats for current session"""
        if not hasattr(st.session_state, "session_stats"):
            st.session_state.session_stats = {
                "session_start": datetime.now(),
                "upload_count": 0,
                "record_count": 0,
                "transcription_count": 0,
                "total_audio_duration": 0.0,
                "errors": 0
            }
        return st.session_state.session_stats

# Initialize Analytics
analytics = StreamlitAnalytics(use_google_sheets=USE_GOOGLE_SHEETS)
stats = analytics.get_session_stats()

# Log session start
if "session_logged" not in st.session_state:
    analytics.log_event(
        "session_start",
        user_agent=st.session_state.get("user_agent", "unknown"),
        page_title="ASR Streamlit Demo"
    )
    st.session_state.session_logged = True

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="ASR Streamlit Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load and cache the recognizer for faster performance
@st.cache_resource
def load_recognizer(model_path: str = "./model_200_fixed.pth") -> SpeechRecognizer:
    analytics.log_event("model_load", model_path=model_path)
    return SpeechRecognizer(model_path)

# Initialize recognizer
recognizer = load_recognizer()

# Header
st.title("🎙️ Automatic Speech Recognition Demo")

# Instructions
st.markdown(
    """
    **Welcome!**
    - **Upload** an audio file (WAV, MP3, FLAC, OGG, M4A)
    - **Or** record a short voice note below
    - Click **Transcribe** to convert speech to text
    """
)

# Sidebar for mode selection
st.sidebar.header("Input Mode")
mode = st.sidebar.radio("Choose input:", ["Upload File", "Record Audio"])

# Analytics Dashboard
with st.sidebar.expander("📊 Analytics (Session)"):
    session_duration = (datetime.now() - stats["session_start"]).total_seconds()
    st.metric("Session Duration (sec)", f"{session_duration:.1f}")
    st.metric("Uploads", stats["upload_count"])
    st.metric("Recordings", stats["record_count"])
    st.metric("Transcriptions", stats["transcription_count"])
    st.metric("Errors", stats["errors"])
    if stats["total_audio_duration"] > 0:
        st.metric("Total Audio Duration (sec)", f"{stats['total_audio_duration']:.2f}")

tmp_path = None

# ==================== UPLOAD FILE MODE ====================

if mode == "Upload File":
    st.subheader("📁 Upload Audio File")
    uploaded_file = st.file_uploader(
        "Select an audio file:",
        type=["wav", "mp3", "flac", "ogg", "m4a"]
    )
    if uploaded_file is not None:
        # Log file upload
        file_size = len(uploaded_file.getvalue())
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        analytics.log_event(
            "file_uploaded",
            filename=uploaded_file.name,
            file_size_bytes=file_size,
            file_extension=file_extension
        )
        stats["upload_count"] += 1
        
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        
        st.audio(tmp_path, format=f"audio/{suffix.replace('.', '')}")
        
        if st.button("Transcribe Upload"):
            transcription_start = time.time()
            try:
                # Load the audio file using soundfile
                data, sr = sf.read(tmp_path, dtype='float32')
                audio_duration = len(data) / sr
                
                # Convert to torch tensor and ensure correct shape [channels, samples]
                if len(data.shape) == 1:
                    waveform = torch.from_numpy(data).unsqueeze(0)
                else:
                    waveform = torch.from_numpy(data.T)

                # Resample if needed
                if sr != 22050:
                    resampler = torchaudio.transforms.Resample(sr, 22050)
                    waveform = resampler(waveform)
                    sr = 22050
                
                waveform = waveform.to(dtype=torch.float64)

                st.write(f"🔍 Loaded upload: sample_rate={sr}, waveform shape={waveform.shape}")
                
                spec = recognizer._preprocess_audio(tmp_path)
                if spec is None:
                    st.error("Error: Preprocessing returned None for upload.")
                    stats["errors"] += 1
                    analytics.log_event(
                        "preprocessing_error",
                        input_mode="upload",
                        audio_duration=audio_duration
                    )
                else:
                    st.write(f"🔍 Spectrogram shape: {spec.shape}")
                    with st.spinner("Transcribing uploaded file..."):
                        transcription = recognizer.transcribe(tmp_path)
                    
                    transcription_time = time.time() - transcription_start
                    
                    st.success("✅ Transcription complete")
                    st.text_area("📝 Transcribed Text", transcription, height=200)
                    
                    # Log transcription metrics
                    analytics.log_event(
                        "transcription_complete",
                        input_mode="upload",
                        audio_duration=audio_duration,
                        transcription_time=transcription_time,
                        transcription_length=len(transcription),
                        word_count=len(transcription.split()),
                        sample_rate=sr,
                        waveform_shape=str(waveform.shape),
                        spectrogram_shape=str(spec.shape)
                    )
                    stats["transcription_count"] += 1
                    stats["total_audio_duration"] += audio_duration
                    
            except Exception as e:
                st.error(f"Error during transcription: {str(e)}")
                stats["errors"] += 1
                analytics.log_event(
                    "transcription_error",
                    input_mode="upload",
                    error_type=type(e).__name__,
                    error_message=str(e)
                )

# ==================== RECORD AUDIO MODE ====================

else:
    st.subheader("🎤 Record Your Voice")
    st.write("Use the recorder below to capture a short voice note.")
    audio_bytes = st.audio_input("Record a voice message")
    
    if audio_bytes is not None:
        # Log recording initiated
        analytics.log_event("recording_captured")
        stats["record_count"] += 1
        
        try:
            # Save raw recording
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_raw:
                tmp_raw.write(audio_bytes.read())
                raw_path = tmp_raw.name
            
            # Load raw audio file using soundfile
            data, sr = sf.read(raw_path, dtype='float32')
            audio_duration = len(data) / sr
            
            # Convert to torch tensor and ensure correct shape [channels, samples]
            if len(data.shape) == 1:
                waveform = torch.from_numpy(data).unsqueeze(0)
            else:
                waveform = torch.from_numpy(data.T)
            
            # Resample to model rate (22050 Hz)
            if sr != 22050:
                resampler = torchaudio.transforms.Resample(sr, 22050)
                waveform = resampler(waveform)
                sr = 22050
            
            waveform = waveform.to(dtype=torch.float64)

            # Ensure stereo channels for inference pipeline
            if waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)

            # Save wave data to file for preprocessing
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp2:
                data_np = waveform.transpose(0, 1).cpu().numpy()
                sf.write(tmp2.name, data_np, sr)
                tmp_path = tmp2.name

            # Debug preprocess on resampled
            spec = recognizer._preprocess_audio(tmp_path)
            if spec is None:
                st.error("Error: Preprocessing returned None for recording.")
                stats["errors"] += 1
                analytics.log_event(
                    "preprocessing_error",
                    input_mode="recording",
                    audio_duration=audio_duration
                )
            
            # Transcription
            if st.button("Transcribe Recording"):
                transcription_start = time.time()
                with st.spinner("Transcribing recorded audio..."):
                    transcription = recognizer.transcribe(tmp_path)
                
                transcription_time = time.time() - transcription_start
                
                if transcription:
                    st.success("✅ Transcription complete")
                    st.text_area("📝 Transcribed Text", transcription, height=200)
                    
                    # Log transcription metrics
                    analytics.log_event(
                        "transcription_complete",
                        input_mode="recording",
                        audio_duration=audio_duration,
                        transcription_time=transcription_time,
                        transcription_length=len(transcription),
                        word_count=len(transcription.split()),
                        sample_rate=sr,
                        waveform_shape=str(waveform.shape),
                        spectrogram_shape=str(spec.shape) if spec is not None else None
                    )
                    stats["transcription_count"] += 1
                    stats["total_audio_duration"] += audio_duration
                else:
                    st.error("Failed to transcribe the recording. Please ensure clear speech and minimal noise.")
                    stats["errors"] += 1
                    analytics.log_event(
                        "transcription_error",
                        input_mode="recording",
                        audio_duration=audio_duration,
                        error_type="empty_result"
                    )
        
        except Exception as e:
            st.error(f"Error processing recording: {str(e)}")
            stats["errors"] += 1
            analytics.log_event(
                "recording_processing_error",
                error_type=type(e).__name__,
                error_message=str(e)
            )

# Footer
st.markdown("---")
st.caption("💾 Analytics data is logged locally to `analytics.jsonl`")
