from faster_whisper import WhisperModel
import pyaudio
import numpy as np
from datetime import datetime
import threading
import queue
import time
from pyannote.audio import Pipeline
import torch
import wave
import tempfile
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Audio recording parameters
CHUNK = 1024 * 4
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000
WINDOW_SIZE = 3
OVERLAP = 1.5

# Punctuation characters to ignore in word anomaly detection
PUNCTUATION = "\"',.!?:;()[]{}—-"

# Initialize the Whisper model
logger.info("Loading Whisper model...")
model = WhisperModel("base", device="cpu", compute_type="int8")
logger.info("Whisper model loaded successfully!")

# Initialize the diarization pipeline
logger.info("Loading diarization pipeline...")

# Get token from environment variable
hf_token = os.getenv("HF_TOKEN")

# Check if token exists and is valid
if not hf_token or hf_token == "YOUR_HF_TOKEN":
    logger.error("Please set your Hugging Face token in the .env file!")
    logger.error("1. Create a token at https://huggingface.co/settings/tokens")
    logger.error("2. Accept terms at https://huggingface.co/pyannote/speaker-diarization")
    logger.error("3. Add your token to .env file: HF_TOKEN=your_token_here")
    exit(1)

try:
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=hf_token
    )
    # Test the pipeline with a simple validation
    if diarization_pipeline is None:
        raise ValueError("Pipeline initialization failed - check model access permissions")
    logger.info("Diarization pipeline loaded successfully!")
except Exception as e:
    logger.error(f"Failed to load diarization pipeline: {str(e)}")
    logger.error("Please make sure you have:")
    logger.error("1. Created a Hugging Face account")
    logger.error("2. Created an access token at https://huggingface.co/settings/tokens")
    logger.error("3. Accepted the terms at https://huggingface.co/pyannote/speaker-diarization")
    logger.error("4. Added your token to .env file: HF_TOKEN=your_token_here")
    logger.error("\nIMPORTANT: You must accept the model terms at:")
    logger.error("https://huggingface.co/pyannote/speaker-diarization")
    logger.error("Click on 'Access repository' and accept the license agreement")
    exit(1)

# Create a queue for audio chunks
audio_queue = queue.Queue()
is_recording = True

def save_audio_chunk(audio_data, sample_rate=RATE):
    """Save audio data to a temporary WAV file"""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            with wave.open(temp_file.name, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(4)  # 32-bit float
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data)
            logger.debug(f"Saved audio chunk to {temp_file.name}")
            return temp_file.name
    except Exception as e:
        logger.error(f"Error saving audio chunk: {str(e)}")
        raise

def record_audio():
    """Record audio from microphone and put chunks into the queue"""
    try:
        p = pyaudio.PyAudio()
        
        # List available input devices
        logger.info("\nAvailable Audio Input Devices:")
        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            if dev.get('maxInputChannels') > 0:
                logger.info(f"Index {i}: {dev.get('name')}")
        
        # Get default input device
        default_input = p.get_default_input_device_info()
        logger.info(f"\nUsing input device: {default_input.get('name')}")
        
        stream = p.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       input_device_index=0,  # Explicitly use MacBook Pro Microphone
                       frames_per_buffer=CHUNK)
        
        logger.info("* Recording started... (Press Ctrl+C to stop)")
        
        audio_buffer = []
        buffer_timestamps = []
        
        while is_recording:
            try:
                current_time = time.time()
                data = stream.read(CHUNK, exception_on_overflow=False)
                
                # Add to buffer
                audio_buffer.append(data)
                buffer_timestamps.append(current_time)
                
                # When buffer reaches window size, add to queue
                if len(audio_buffer) >= int(RATE * WINDOW_SIZE / CHUNK):
                    audio_queue.put((audio_buffer.copy(), buffer_timestamps.copy()))
                    
                    # Keep overlap
                    overlap_chunks = int(RATE * OVERLAP / CHUNK)
                    audio_buffer = audio_buffer[-overlap_chunks:]
                    buffer_timestamps = buffer_timestamps[-overlap_chunks:]
                
            except Exception as e:
                logger.error(f"Error reading audio chunk: {e}")
                continue
        
        logger.info("* Recording stopped")
        stream.stop_stream()
        stream.close()
        p.terminate()
        
    except Exception as e:
        logger.error(f"Error initializing audio: {e}")

def calculate_word_anomaly_score(word):
    """Calculate anomaly score for a single word"""
    duration = word.end - word.start
    score = 0.0
    
    if word.probability < 0.15:
        score += 1.0
    if duration < 0.133:
        score += (0.133 - duration) * 15
    if duration > 2.0:
        score += duration - 2.0
        
    return score

def process_audio():
    """Process audio chunks with transcription and speaker diarization"""
    while is_recording:
        if not audio_queue.empty():
            try:
                audio_data, timestamps = audio_queue.get()
                
                # Convert audio data to numpy array
                audio_array = np.frombuffer(b''.join(audio_data), dtype=np.float32)
                
                # Normalize audio
                audio_array = audio_array / np.max(np.abs(audio_array)) if np.max(np.abs(audio_array)) > 0 else audio_array
                
                # Save audio chunk to temporary file for diarization
                temp_audio_file = save_audio_chunk(b''.join(audio_data))
                logger.debug(f"Saved audio to temporary file: {temp_audio_file}")
                
                # Get speaker diarization
                diarization = diarization_pipeline(temp_audio_file)
                logger.debug("Speaker diarization completed")
                
                # Transcribe the audio
                segments, info = model.transcribe(
                    audio_array,
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=300,
                        speech_pad_ms=100,
                        max_speech_duration_s=WINDOW_SIZE
                    ),
                    word_timestamps=True,
                    condition_on_previous_text=False,
                    initial_prompt=None,
                    language="en",
                    task="transcribe",
                    beam_size=5,
                    patience=1,
                    length_penalty=1,
                    temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    compression_ratio_threshold=2.4,
                    log_prob_threshold=-1.0,
                    no_speech_threshold=0.6,
                    repetition_penalty=1.0,
                    no_repeat_ngram_size=3
                )
                logger.debug("Transcription completed")
                
                current_time = time.time()
                processing_delay = current_time - timestamps[0]
                
                # Process segments and assign speakers
                for segment in segments:
                    if segment.text.strip():
                        # Find the speaker for this segment's timestamp
                        speaker = "UNKNOWN"
                        segment_start = segment.start
                        segment_end = segment.end
                        
                        # Find overlapping speaker turns
                        for turn, _, speaker_id in diarization.itertracks(yield_label=True):
                            if (turn.start <= segment_end and turn.end >= segment_start):
                                speaker = f"SPEAKER_{speaker_id}"
                                break
                        
                        # Check for word-level anomalies
                        if segment.words:
                            words = [w for w in segment.words if w.word.strip() not in PUNCTUATION]
                            if words:
                                # Calculate word anomaly scores
                                anomaly_score = sum(calculate_word_anomaly_score(w) for w in words[:8])
                                
                                # Skip if segment seems anomalous
                                if anomaly_score >= 3 or anomaly_score + 0.01 >= len(words):
                                    continue
                        
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] {speaker}: {segment.text} (processed in {processing_delay:.2f}s)")
                
            except Exception as e:
                logger.error(f"Error processing audio: {e}")
                logger.exception("Full error traceback:")
        
        time.sleep(0.05)

def main():
    global is_recording
    
    record_thread = threading.Thread(target=record_audio)
    process_thread = threading.Thread(target=process_audio)
    
    record_thread.start()
    process_thread.start()
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("\nStopping recording...")
        is_recording = False
        record_thread.join()
        process_thread.join()

if __name__ == "__main__":
    main()
