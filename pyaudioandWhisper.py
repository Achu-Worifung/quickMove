import pyaudio
import numpy as np
from faster_whisper import WhisperModel
import torch
import wave
import time

# Setting up whisper model
model_size = 'tiny'
cpu_cores = max(1, torch.get_num_threads())
model = WhisperModel(
    model_size,
    device='cpu',
    compute_type='int8',
    num_workers=cpu_cores,
    cpu_threads=8,
    download_root='./models'
)

# Recording settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # 16kHz sample rate for Whisper
CHUNK = 1024
RECORD_SECONDS = 2  # Record in 5-second segments

# Initialize PyAudio
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print("Recording...")
counter = 1

try:
    while True:
        print(f"Recording segment {counter}...")
        frames = []
        
        # Collect audio for RECORD_SECONDS
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        
        # Save the audio segment
        filename = f"recording_{counter}.wav"
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        # Convert to numpy array for transcription
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32) / 32768.0
        
        print(f"Transcribing segment {counter}...")
        segments, _ = model.transcribe(
            filename,  # Use the saved file for better results
            beam_size=5,  # Increased from 1 for better accuracy
            best_of=1,    # Increased from 1
            temperature=0.0,
            language='en',
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            word_timestamps=False
        )
        
        print(f"Transcription for segment {counter}:")
        segment_text = ""
        for segment in segments:
            segment_text += segment.text + " "
            
        if segment_text.strip():
            print(segment_text)
        else:
            print("No speech detected in this segment.")
            
        counter += 1
        
except KeyboardInterrupt:
    print("Recording stopped.")
finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()