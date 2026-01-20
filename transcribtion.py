import queue
import sys
import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad, get_speech_timestamps

# ======================
# CONFIG
# ======================
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE * 2)  # 2-second chunks
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

# VAD thresholds
VAD_THRESHOLD = 0.5
MIN_SPEECH_DURATION = 0.3  # seconds
MIN_SILENCE_DURATION = 0.8  # seconds to wait before finalizing

# ======================
# Load Models
# ======================
print("Loading models...")
whisper = WhisperModel(
    "base.en" if DEVICE == "cpu" else "medium.en",  # Use smaller models for faster processing
    device=DEVICE,
    compute_type=COMPUTE_TYPE
)
vad = load_silero_vad()
print("Models loaded\n")

# ======================
# Audio Processing
# ======================
audio_queue = queue.Queue()
accumulated_audio = []
context_text = ""
is_speaking = False
silence_counter = 0

def audio_callback(indata, frames, time, status):
    """Add incoming audio to queue"""
    if status:
        print(f"Audio error: {status}", file=sys.stderr)
    audio_queue.put(indata.copy())

def process_chunk(audio_chunk):
    """Process single audio chunk for VAD"""
    if len(audio_chunk) < SAMPLE_RATE * 0.1:  # Too short
        return False
    
    timestamps = get_speech_timestamps(
        audio_chunk,
        vad,
        sampling_rate=SAMPLE_RATE,
        threshold=VAD_THRESHOLD,
        min_speech_duration_ms=int(MIN_SPEECH_DURATION * 1000),
        min_silence_duration_ms=10,  # Small for incremental
        return_seconds=False
    )
    return len(timestamps) > 0

def transcribe_incremental():
    """Transcribe accumulated audio with context"""
    global accumulated_audio, context_text
    
    if not accumulated_audio:
        return ""
    
    # Combine all accumulated audio
    audio_data = np.concatenate(accumulated_audio)
    
    # Keep last 3 chunks for next time (for context)
    if len(accumulated_audio) > 3:
        keep_chunks = 2  # Keep fewer for context
        context_audio = accumulated_audio[-keep_chunks:]
        accumulated_audio = [np.concatenate(context_audio)] if keep_chunks > 0 else []
    else:
        accumulated_audio = [audio_data.copy()]
    
    # Transcribe
    segments, _ = whisper.transcribe(
        audio_data,
        language="en",
        beam_size=3,
        best_of=3,
        temperature=0.0,  # Use 0 for deterministic output
        condition_on_previous_text=True,
        initial_prompt=context_text[-500:] if context_text else None,  # Last 500 chars as context
        vad_filter=True,  # Let whisper do final VAD
        vad_parameters=dict(
            min_silence_duration_ms=100,
            speech_pad_ms=200
        )
    )
    
    # Get transcription
    result = " ".join([seg.text.strip() for seg in segments]).strip()
    
    # Update context
    if result:
        context_text = (context_text + " " + result)[-1000:]  # Keep last 1000 chars
    
    return result

# ======================
# Main Loop
# ======================
print("🎤 Listening... Press Ctrl+C to stop\n")

with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    blocksize=CHUNK_SIZE,
    callback=audio_callback
):
    try:
        while True:
            # Get audio chunk
            chunk = audio_queue.get()
            chunk_flat = chunk.flatten()
            
            # Check for speech in this chunk
            has_speech = process_chunk(chunk_flat)
            
            if has_speech:
                # Add to accumulated audio
                accumulated_audio.append(chunk_flat)
                is_speaking = True
                silence_counter = 0
                
                # If we have enough audio (6+ seconds), transcribe incrementally
                total_duration = sum(len(a) for a in accumulated_audio) / SAMPLE_RATE
                if total_duration >= 6.0:
                    print("⏳ Processing...", end="\r")
                    transcription = transcribe_incremental()
                    if transcription:
                        print(f"\r📝 {transcription}")
            
            elif is_speaking:
                # We were speaking, now in silence
                silence_counter += 1
                
                # If silence continues for specified time, finalize
                if silence_counter >= MIN_SILENCE_DURATION / (CHUNK_SIZE / SAMPLE_RATE):
                    if accumulated_audio:
                        print("⏳ Finalizing...", end="\r")
                        transcription = transcribe_incremental()
                        if transcription:
                            print(f"\r✅ {transcription}\n")
                    
                    # Reset
                    accumulated_audio = []
                    is_speaking = False
                    silence_counter = 0
            
            else:
                # Not speaking, reset counter
                silence_counter = 0
                
    except KeyboardInterrupt:
        # Final transcription if audio remains
        if accumulated_audio:
            transcription = transcribe_incremental()
            if transcription:
                print(f"\n📝 Final: {transcription}")
        
        print("\n\n✨ Stopped listening")
        print(f"📋 Context length: {len(context_text)} characters")