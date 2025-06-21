import streamlit as st
# Page config
st.set_page_config(
    page_title="ASR Streamlit Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

from inference import SpeechRecognizer
from pathlib import Path
import tempfile
import torchaudio
import soundfile as sf

# Load and cache the recognizer for faster performance
@st.cache_resource
def load_recognizer(model_path: str = "./model_200_fixed.pth") -> SpeechRecognizer:
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

tmp_path = None

# Upload File mode
if mode == "Upload File":
    st.subheader("📁 Upload Audio File")
    uploaded_file = st.file_uploader(
        "Select an audio file:",
        type=["wav", "mp3", "flac", "ogg", "m4a"]
    )
    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        st.audio(tmp_path, format=f"audio/{suffix.replace('.', '')}")
        if st.button("Transcribe Upload"):
            # Debug load
            waveform, sr = torchaudio.load(tmp_path)
            st.write(f"🔍 Loaded upload: sample_rate={sr}, waveform shape={waveform.shape}")
            spec = recognizer._preprocess_audio(tmp_path)
            if spec is None:
                st.error("Error: Preprocessing returned None for upload.")
            else:
                st.write(f"🔍 Spectrogram shape: {spec.shape}")
                with st.spinner("Transcribing uploaded file..."):
                    transcription = recognizer.transcribe(tmp_path)
                st.success("✅ Transcription complete")
                st.text_area("📝 Transcribed Text", transcription, height=200)

# Record Audio mode
else:
    st.subheader("🎤 Record Your Voice")
    st.write("Use the recorder below to capture a short voice note.")
    audio_bytes = st.audio_input("Record a voice message")
    if audio_bytes is not None:
        # Save raw recording
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_raw:
            tmp_raw.write(audio_bytes.read())
            raw_path = tmp_raw.name
        # Playback raw
        st.audio(raw_path, format="audio/wav")
        # Load raw and debug
        waveform, sr = torchaudio.load(raw_path)
        st.write(f"🔍 Raw recording: sample_rate={sr}, waveform shape={waveform.shape}")
                # Resample to model rate (22050 Hz)
        if sr != 22050:
            resampler = torchaudio.transforms.Resample(sr, 22050)
            waveform = resampler(waveform)
            sr = 22050
        # Ensure stereo channels for inference pipeline
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        # Save wave data to file for preprocessing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp2:
            # soundfile expects shape [frames, channels]
            data_np = waveform.transpose(0, 1).cpu().numpy()
            sf.write(tmp2.name, data_np, sr)
            tmp_path = tmp2.name
        # Playback resampled (stereo) audio
        st.audio(tmp_path, format="audio/wav")
        st.write(f"🔍 Resampled: sample_rate={sr}, channels={waveform.shape[0]}, waveform shape={waveform.shape}")
        # Debug preprocess on resampled
        spec = recognizer._preprocess_audio(tmp_path)
        if spec is None:
            st.error("Error: Preprocessing returned None for recording.")
        else:
            st.write(f"🔍 Spectrogram shape: {spec.shape}")
        # Transcription
        if st.button("Transcribe Recording"):
            with st.spinner("Transcribing recorded audio..."):
                transcription = recognizer.transcribe(tmp_path)
            if transcription:
                st.success("✅ Transcription complete")
                st.text_area("📝 Transcribed Text", transcription, height=200)
            else:
                st.error("Failed to transcribe the recording. Please ensure clear speech and minimal noise.")

# Footer
st.markdown("---")
st.write("Powered by **SpeechRecognizer** from your inference model")
