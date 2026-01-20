import queue
import sys
import numpy as np
import sounddevice as sd
import torch
import time
import scipy.signal as signal
from datetime import datetime
from collections import deque
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad, get_speech_timestamps

# ======================
# CONFIGURATION
# ======================
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE * 1)  # 1-second chunks
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# VAD thresholds
VAD_THRESHOLD = 0.7
MIN_SPEECH_DURATION = 0.3  # seconds
MIN_SILENCE_DURATION = 0.5  # seconds to wait before finalizing

# ======================
# Enhanced Model Loading with Fallback
# ======================
def load_models_with_fallback():
    """Load models with automatic fallback based on available resources"""
    
    print("🚀 Loading models...")
    
    # Define model preferences based on device
    if torch.cuda.is_available():
        print("✓ GPU detected, using optimized models")
        model_preferences = [
            ("distil-large-v3", "float16"),  # Fast and accurate
            ("medium.en", "float16"),
            ("base.en", "float16"),
        ]
    else:
        print("⚠️ Using CPU, selecting lighter models")
        model_preferences = [
            ("base.en", "int8"),
            ("tiny.en", "int8"),
        ]
    
    # Try loading Whisper model
    whisper_model = None
    for model_size, compute_type in model_preferences:
        try:
            print(f"  Attempting {model_size}...", end=" ")
            whisper_model = WhisperModel(
                model_size,
                device=DEVICE,
                compute_type=compute_type,
                cpu_threads=4 if DEVICE == "cpu" else 0,
                num_workers=2,
                download_root="./models"
            )
            print("✓ Success!")
            break
        except Exception as e:
            print(f"✗ Failed: {str(e)[:50]}...")
            continue
    
    if whisper_model is None:
        raise RuntimeError("❌ Failed to load any Whisper model")
    
    # Try loading VAD
    vad_model = None
    try:
        vad_model = load_silero_vad()
        print("✓ VAD loaded")
    except Exception as e:
        print(f"⚠️ VAD failed to load: {e}")
        print("  Continuing without VAD (using Whisper's VAD)")
    
    print()
    return whisper_model, vad_model

# ======================
# Audio Processing with Noise Reduction
# ======================
class AudioProcessor:
    """Handles audio preprocessing and enhancement"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        
    def apply_highpass_filter(self, audio_data, cutoff=100):
        """Remove low-frequency noise"""
        if len(audio_data) < 4:
            return audio_data
            
        nyquist = self.sample_rate / 2
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(2, normal_cutoff, btype='high', analog=False)
        return signal.filtfilt(b, a, audio_data)
    
    def denoise_audio(self, audio_data):
        """Simple noise reduction"""
        if len(audio_data) < 100:
            return audio_data
            
        # Apply high-pass filter
        filtered = self.apply_highpass_filter(audio_data)
        
        # Mild noise gate
        rms = np.sqrt(np.mean(filtered**2))
        threshold = max(rms * 0.15, 0.005)
        filtered[np.abs(filtered) < threshold] *= 0.3
        
        return filtered
    
    def normalize_audio(self, audio_data):
        """Normalize audio level"""
        if len(audio_data) == 0:
            return audio_data
            
        max_val = np.max(np.abs(audio_data))
        if max_val > 0.01:  # Only normalize if there's significant audio
            return audio_data * (0.8 / max_val)  # 80% of max to avoid clipping
        return audio_data

# ======================
# Smart Context Management
# ======================
class ContextManager:
    """Manages transcription context and history"""
    
    def __init__(self, max_context_chars=800, max_history=10):
        self.max_context_chars = max_context_chars
        self.max_history = max_history
        self.history = deque(maxlen=max_history)
        self.full_transcript = []
        
    def add_transcription(self, text, confidence=0):
        """Add new transcription to history"""
        if not text or len(text.strip()) < 2:
            return
            
        entry = {
            'text': text.strip(),
            'confidence': confidence,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        }
        
        self.history.append(entry)
        self.full_transcript.append(text.strip())
        
        # Keep full transcript for summary
        if len(self.full_transcript) > 50:
            self.full_transcript = self.full_transcript[-50:]
    
    def get_context(self):
        """Get context for next transcription"""
        if not self.history:
            return None
            
        # Get last 2-3 utterances
        recent = list(self.history)[-3:]
        
        # Build context string
        context_parts = []
        for i, item in enumerate(recent):
            if i < len(recent) - 1:
                context_parts.append(item['text'] + "...")
            else:
                context_parts.append(item['text'])
        
        context = " ".join(context_parts)
        
        # Trim if too long
        if len(context) > self.max_context_chars:
            words = context.split()
            trimmed = []
            length = 0
            
            for word in reversed(words):
                if length + len(word) + 1 <= self.max_context_chars:
                    trimmed.insert(0, word)
                    length += len(word) + 1
                else:
                    break
            
            context = "..." + " ".join(trimmed)
        
        return context
    
    def get_summary(self):
        """Get summary of recent conversation"""
        if len(self.history) < 3:
            return None
            
        summary = f"Recent ({len(self.history)} utterances): "
        summary += " | ".join([h['text'][:30] + "..." if len(h['text']) > 30 
                              else h['text'] for h in list(self.history)[-3:]])
        
        return summary

# ======================
# Real-time Display
# ======================
class TranscriptionDisplay:
    """Manages real-time display of transcription status"""
    
    def __init__(self):
        self.status = "idle"
        self.last_update = time.time()
        self.partial_text = ""
        
    def update(self, status, text="", force=False):
        """Update display status"""
        now = time.time()
        
        # Clear line
        sys.stdout.write("\r" + " " * 100 + "\r")
        
        # Format timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if status == "listening":
            if force or now - self.last_update > 0.5:
                dots = "." * ((int(now) % 3) + 1)
                sys.stdout.write(f"\r[{timestamp}] 🎤 Listening{dots}")
                self.last_update = now
                
        elif status == "processing":
            elapsed = int(now - self.last_update)
            dots = "." * ((elapsed % 4) + 1)
            sys.stdout.write(f"\r[{timestamp}] ⏳ Processing{dots}")
            
        elif status == "speech_detected":
            sys.stdout.write(f"\r[{timestamp}] 🔊 Speech detected")
            
        elif status == "result":
            sys.stdout.write(f"\r[{timestamp}] 📝 {text}\n")
            self.partial_text = ""
            
        elif status == "partial":
            # Show partial result
            if len(text) > 60:
                display = text[:55] + "..."
            else:
                display = text
            sys.stdout.write(f"\r[{timestamp}] 🔄 {display}")
            self.partial_text = text
            
        elif status == "silence":
            sys.stdout.write(f"\r[{timestamp}] 🔇 Silence")
            
        elif status == "error":
            sys.stdout.write(f"\r[{timestamp}] ❌ {text}")
            
        sys.stdout.flush()
        self.status = status

# ======================
# Enhanced Transcription
# ======================
class EnhancedTranscriber:
    """Handles transcription with enhanced features"""
    
    def __init__(self, whisper_model, vad_model=None):
        self.whisper = whisper_model
        self.vad = vad_model
        
    def check_speech(self, audio_chunk):
        """Check if audio contains speech using VAD"""
        if self.vad is None or len(audio_chunk) < SAMPLE_RATE * 0.1:
            return True  # Fallback to always true if no VAD
            
        try:
            timestamps = get_speech_timestamps(
                audio_chunk,
                self.vad,
                sampling_rate=SAMPLE_RATE,
                threshold=VAD_THRESHOLD,
                min_speech_duration_ms=int(MIN_SPEECH_DURATION * 1000),
                min_silence_duration_ms=50,
                return_seconds=False
            )
            return len(timestamps) > 0
        except:
            return True  # Fallback if VAD fails
    
    def transcribe(self, audio_data, context=None):
        """Transcribe audio with enhanced settings"""
        if len(audio_data) < SAMPLE_RATE * 0.5:  # Less than 0.5s
            return "", 0.0
        
        # Adjust parameters based on audio length
        audio_duration = len(audio_data) / SAMPLE_RATE
        beam_size = 3 if audio_duration < 8 else 5
        best_of = 3 if audio_duration < 8 else 5
        
        try:
            segments, info = self.whisper.transcribe(
                audio_data,
                language="en",
                beam_size=beam_size,
                best_of=best_of,
                temperature=0.0,
                condition_on_previous_text=context is not None,
                initial_prompt=context,
                vad_filter=True,
                vad_parameters={
                    "threshold": 0.4,
                    "min_speech_duration_ms": 250,
                    "min_silence_duration_ms": 100,
                    "speech_pad_ms": 200,
                },
                word_timestamps=False,
                suppress_blank=True,
                without_timestamps=True,
            )
            
            # Collect results with confidence
            results = []
            total_confidence = 0
            segment_count = 0
            
            for segment in segments:
                text = segment.text.strip()
                if text:
                    results.append(text)
                    # Use no_speech_prob to filter out non-speech
                    if hasattr(segment, 'no_speech_prob') and segment.no_speech_prob > 0.8:
                        continue
                    segment_count += 1
            
            if not results:
                return "", 0.0
            
            final_text = " ".join(results).strip()
            
            # Simple confidence estimation
            confidence = min(1.0, max(0.0, 1.0 - (segment_count / 10)))
            
            return final_text, confidence
            
        except Exception as e:
            print(f"\nTranscription error: {e}")
            return "", 0.0

# ======================
# Main Application
# ======================
class TranscriptionApp:
    """Main transcription application"""
    
    def __init__(self):
        # Load models
        self.whisper, self.vad = load_models_with_fallback()
        
        # Initialize components
        self.audio_processor = AudioProcessor(SAMPLE_RATE)
        self.context_manager = ContextManager()
        self.display = TranscriptionDisplay()
        self.transcriber = EnhancedTranscriber(self.whisper, self.vad)
        
        # State variables
        self.audio_queue = queue.Queue()
        self.accumulated_audio = []
        self.is_speaking = False
        self.silence_counter = 0
        self.last_transcription_time = time.time()
        
    def audio_callback(self, indata, frames, time_info, status):
        """Audio input callback with preprocessing"""
        if status:
            if status.input_overflow:
                print("\n⚠️ Audio overflow - some audio may be lost")
        
        # Process audio chunk
        audio_chunk = indata.copy().flatten()
        audio_chunk = self.audio_processor.denoise_audio(audio_chunk)
        audio_chunk = self.audio_processor.normalize_audio(audio_chunk)
        
        # Ensure audio data is in the correct format (float32)
        audio_chunk = audio_chunk.astype(np.float32)
        
        self.audio_queue.put(audio_chunk)
    
    def process_audio_chunk(self, chunk):
        """Process a single audio chunk"""
        # Check for speech
        has_speech = self.transcriber.check_speech(chunk)
        
        if has_speech:
            self.accumulated_audio.append(chunk)
            
            if not self.is_speaking:
                self.is_speaking = True
                self.display.update("speech_detected")
            
            self.silence_counter = 0
            
            # Check if we should transcribe incrementally
            total_duration = sum(len(a) for a in self.accumulated_audio) / SAMPLE_RATE
            
            # Transcribe every 4-5 seconds of speech
            if total_duration >= 4.5 and self.is_speaking:
                self.display.update("processing")
                
                # Get context
                context = self.context_manager.get_context()
                
                # Use last 4 chunks (8 seconds) for transcription
                audio_to_transcribe = np.concatenate(self.accumulated_audio[-4:]) \
                    if len(self.accumulated_audio) >= 4 else np.concatenate(self.accumulated_audio)
                
                # Transcribe
                text, confidence = self.transcriber.transcribe(audio_to_transcribe, context)
                
                if text:
                    # Add to context
                    self.context_manager.add_transcription(text, confidence)
                    
                    # Display
                    self.display.update("result", text)
                    
                    # Save to file
                    self.save_transcription(text)
                    
                    # Keep last 2 chunks for overlap
                    if len(self.accumulated_audio) > 2:
                        self.accumulated_audio = self.accumulated_audio[-2:]
                    
                    self.last_transcription_time = time.time()
                    
        elif self.is_speaking:
            # We were speaking, now in silence
            self.silence_counter += 1
            
            # Check if silence is long enough to finalize
            silence_time = self.silence_counter * (CHUNK_SIZE / SAMPLE_RATE)
            
            if silence_time >= MIN_SILENCE_DURATION:
                if self.accumulated_audio:
                    self.display.update("processing")
                    
                    # Get context
                    context = self.context_manager.get_context()
                    
                    # Transcribe remaining audio
                    audio_to_transcribe = np.concatenate(self.accumulated_audio)
                    text, confidence = self.transcriber.transcribe(audio_to_transcribe, context)
                    
                    if text:
                        self.context_manager.add_transcription(text, confidence)
                        self.display.update("result", text)
                        self.save_transcription(text)
                        print()  # Blank line between utterances
                
                # Reset for next utterance
                self.accumulated_audio = []
                self.is_speaking = False
                self.silence_counter = 0
                self.display.update("listening", force=True)
        
        else:
            # Not speaking, just update display occasionally
            if time.time() - self.last_transcription_time > 2:
                self.display.update("listening")
    
    def save_transcription(self, text):
        """Save transcription to file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open("transcriptions.txt", "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {text}\n\n")
        except:
            pass  # Don't crash if file save fails
    
    def run(self):
        """Main application loop"""
        print("\n" + "="*60)
        print("🎤 ENHANCED REAL-TIME TRANSCRIPTION SYSTEM")
        print("="*60)
        print(f"Device: {DEVICE.upper()}")
        print(f"Sample rate: {SAMPLE_RATE}Hz")
        print(f"Chunk size: {CHUNK_SIZE/SAMPLE_RATE:.1f}s")
        print(f"Min silence: {MIN_SILENCE_DURATION}s")
        print("="*60 + "\n")
        
        print("📋 Commands:")
        print("  • Speak naturally - system will transcribe in real-time")
        print("  • Pause 1.2 seconds to finalize utterance")
        print("  • Press Ctrl+C to stop\n")
        
        try:
            # Start audio stream
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=CHUNK_SIZE,
                callback=self.audio_callback,
                latency='low'
            ):
                print("✅ Audio stream started")
                self.display.update("listening", force=True)
                
                # Main processing loop
                while True:
                    try:
                        # Get next audio chunk
                        chunk = self.audio_queue.get(timeout=0.5)
                        self.process_audio_chunk(chunk)
                        
                    except queue.Empty:
                        # No audio data, just update display
                        self.display.update("listening")
                        
        except KeyboardInterrupt:
            print("\n\n" + "="*60)
            print("📊 SESSION SUMMARY")
            print("="*60)
            
            summary = self.context_manager.get_summary()
            if summary:
                print(f"\n{summary}")
            
            print(f"\n📝 Total utterances: {len(self.context_manager.history)}")
            print("💾 Transcript saved to: transcriptions.txt")
            print("\n✨ Session ended\n")
            
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()

# ======================
# Entry Point
# ======================
if __name__ == "__main__":
    app = TranscriptionApp()
    app.run()