from faster_whisper import WhisperModel
import pyaudio
import numpy as np
from datetime import datetime
import threading
import queue
import time

# Audio recording parameters
CHUNK = 1024 * 4
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000
WINDOW_SIZE = 3
OVERLAP = 1.5

# Initialize the Whisper model
print("Loading model...")
model = WhisperModel("base", device="cpu", compute_type="int8")  # Changed to base for faster initial testing
print("Model loaded!")

# Create a queue for audio chunks
audio_queue = queue.Queue()
is_recording = True

def record_audio():
    """Record audio from microphone and put chunks into the queue"""
    p = pyaudio.PyAudio()
    
    # List available input devices
    print("\nAvailable Audio Input Devices:")
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev.get('maxInputChannels') > 0:
            print(f"Index {i}: {dev.get('name')}")
    
    # Get default input device
    default_input = p.get_default_input_device_info()
    print(f"\nUsing input device: {default_input.get('name')}")
    
    stream = p.open(format=FORMAT,
                   channels=CHANNELS,
                   rate=RATE,
                   input=True,
                   input_device_index=0,  # Explicitly use MacBook Pro Microphone
                   frames_per_buffer=CHUNK)
    
    print("* Recording started... (Press Ctrl+C to stop)")
    
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
            print(f"Error recording: {e}")
            break
    
    print("* Recording stopped")
    stream.stop_stream()
    stream.close()
    p.terminate()

def process_audio():
    """Process audio chunks and transcribe"""
    while is_recording:
        if not audio_queue.empty():
            try:
                audio_data, timestamps = audio_queue.get()
                
                # Convert audio data to numpy array
                audio_array = np.frombuffer(b''.join(audio_data), dtype=np.float32)
                
                # Normalize audio
                audio_array = audio_array / np.max(np.abs(audio_array)) if np.max(np.abs(audio_array)) > 0 else audio_array
                
                segments, info = model.transcribe(
                    audio_array,
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=300,
                        speech_pad_ms=100
                    )
                )
                
                current_time = time.time()
                processing_delay = current_time - timestamps[0]
                
                for segment in segments:
                    if segment.text.strip():
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] {segment.text} (processed in {processing_delay:.2f}s)")
                
            except Exception as e:
                print(f"Error processing: {e}")
        
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
        print("\nStopping recording...")
        is_recording = False
        record_thread.join()
        process_thread.join()

if __name__ == "__main__":
    main()
