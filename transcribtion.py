import queue
import sys
import numpy as np
import sounddevice as sd
import torch
import time
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad, get_speech_timestamps

# ======================
# CONFIGURATION
# ======================
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE * 0.5)  # 0.5-second chunks for snappy response
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# VAD thresholds
VAD_THRESHOLD = 0.7
MIN_SPEECH_DURATION = 0.2  # seconds
MIN_SILENCE_DURATION = 0.2  # seconds to wait before finalizing

# Transcription parameters
MAX_ACCUMULATED_DURATION = 1.5  # Transcribe every 1.5 seconds of speech
MIN_TRANSCRIPTION_AUDIO = 0.8   # Minimum 0.8 seconds of audio to transcribe

# ======================
# Fast Model Loading
# ======================
def load_models_fast():
    """Load models with minimal overhead"""
    print("Loading models...")
    
    # Use small models for speed
    if torch.cuda.is_available():
        model_size = "distil-large-v3"  # Small and fast
        compute_type = "float16"
        print("✓ Using GPU with distil-large-v3 model")
    else:
        model_size = "tiny.en"  # Even smaller for CPU
        compute_type = "int8"
        print("✓ Using CPU with tiny.en model")
    
    try:
        whisper = WhisperModel(
            model_size,
            device=DEVICE,
            compute_type=compute_type,
            cpu_threads=4 if DEVICE == "cpu" else 0,
            num_workers=2,
            download_root="./models"
        )
        print("✓ Whisper model loaded")
    except Exception as e:
        print(f"❌ Failed to load whisper: {e}")
        sys.exit(1)
    
    try:
        vad = load_silero_vad()
        print("✓ VAD loaded")
    except Exception as e:
        print(f"⚠️ VAD failed to load: {e}")
        vad = None
    
    print()
    return whisper, vad

# ======================
# Simple Audio Processor
# ======================
class SimpleAudioProcessor:
    """Minimal audio processing for speed"""
    
    @staticmethod
    def normalize_audio(audio_data):
        """Quick normalization"""
        if len(audio_data) == 0:
            return audio_data
        
        max_val = np.max(np.abs(audio_data))
        if max_val > 0.01:
            return audio_data * (0.9 / max_val)
        return audio_data

# ======================
# Fast Context Manager
# ======================
class FastContextManager:
    """Simple context management without timestamps"""
    
    def __init__(self, max_context_chars=300):
        self.max_context_chars = max_context_chars
        self.context_buffer = []
        
    def add_text(self, text):
        """Add text to context buffer"""
        if text and len(text.strip()) > 2:
            self.context_buffer.append(text.strip())
            # Keep only last 3 transcriptions
            if len(self.context_buffer) > 3:
                self.context_buffer.pop(0)
    
    def get_context(self):
        """Get context string"""
        if not self.context_buffer:
            return None
        
        # Join last transcriptions
        context = " ".join(self.context_buffer[-2:])
        
        # Truncate if too long
        if len(context) > self.max_context_chars:
            return "..." + context[-self.max_context_chars:]
        
        return context

# ======================
# Fast Transcriber
# ======================
class FastTranscriber:
    """Optimized for speed with minimal features"""
    
    def __init__(self, whisper_model, vad_model=None):
        self.whisper = whisper_model
        self.vad = vad_model
        
    def check_speech(self, audio_chunk):
        """Fast speech detection"""
        if self.vad is None or len(audio_chunk) < SAMPLE_RATE * 0.1:
            return True  # Skip VAD check if not available
            
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
            return True
    
    def transcribe_fast(self, audio_data, context=None):
        """Fast transcription with minimal parameters"""
        if len(audio_data) < SAMPLE_RATE * 0.3:  # Less than 0.3s
            return ""
        
        try:
            # Use minimal parameters for speed
            segments, _ = self.whisper.transcribe(
                audio_data,
                language="en",
                beam_size=2,  # Small beam for speed
                best_of=2,    # Fewer candidates
                temperature=0.0,
                condition_on_previous_text=context is not None,
                initial_prompt=context,
                vad_filter=False,
                # vad_parameters={
                #     "threshold": 0.3,  # More sensitive for short audio
                #     "min_speech_duration_ms": 200,
                #     "min_silence_duration_ms": 100,
                # },
                  compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                word_timestamps=False,
                 no_speech_threshold=0.6,
                suppress_blank=True,
                without_timestamps=True,
                 prepend_punctuations="\"'¿([{",
                 append_punctuations="\"'.。,，!！?？:：)]}、",
            )
            
            # Collect text
            texts = []
            for segment in segments:
                text = segment.text.strip()
                if text:
                    texts.append(text)
            
            if not texts:
                return ""
            
            return " ".join(texts).strip()
            
        except Exception as e:
            print(f"\nTranscription error: {e}")
            return ""

# ======================
# Minimal Display
# ======================
class MinimalDisplay:
    """Simple display without timestamps"""
    
    @staticmethod
    def clear_line():
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()
    
    @staticmethod
    def show_listening():
        MinimalDisplay.clear_line()
        sys.stdout.write("\r🎤 Listening...")
        sys.stdout.flush()
    
    @staticmethod
    def show_processing():
        MinimalDisplay.clear_line()
        sys.stdout.write("\r⏳ Processing...")
        sys.stdout.flush()
    
    @staticmethod
    def show_result(text):
        MinimalDisplay.clear_line()
        print(f"\r📝 {text}")
    
    @staticmethod
    def show_speech():
        MinimalDisplay.clear_line()
        sys.stdout.write("\r🔊 Speech detected")
        sys.stdout.flush()
    
    @staticmethod
    def show_silence():
        MinimalDisplay.clear_line()
        sys.stdout.write("\r🔇 Silence")
        sys.stdout.flush()

# ======================
# Main Application
# ======================
class SnappyTranscriptionApp:
    """Optimized for speed and snappy response"""
    
    def __init__(self):
        # Load models
        self.whisper, self.vad = load_models_fast()
        
        # Initialize components
        self.audio_processor = SimpleAudioProcessor()
        self.context_manager = FastContextManager()
        self.display = MinimalDisplay()
        self.transcriber = FastTranscriber(self.whisper, self.vad)
        
        # State variables
        self.audio_queue = queue.Queue()
        self.accumulated_audio = []
        self.is_speaking = False
        self.silence_counter = 0
        
    def audio_callback(self, indata, frames, time_info, status):
        """Simple audio callback"""
        if status:
            if status.input_overflow:
                print("\n⚠️ Audio overflow")
        
        # Quick processing
        audio_chunk = indata.copy().flatten()
        audio_chunk = self.audio_processor.normalize_audio(audio_chunk)
        audio_chunk = audio_chunk.astype(np.float32)
        
        self.audio_queue.put(audio_chunk)
    
    def process_audio_chunk(self, chunk):
        """Process single audio chunk"""
        # Check for speech
        has_speech = self.transcriber.check_speech(chunk)
        
        if has_speech:
            self.accumulated_audio.append(chunk)
            
            if not self.is_speaking:
                self.is_speaking = True
                self.display.show_speech()
            
            self.silence_counter = 0
            
            # Check if we should transcribe
            total_duration = sum(len(a) for a in self.accumulated_audio) / SAMPLE_RATE
            
            # Transcribe frequently for snappy feel
            if total_duration >= MAX_ACCUMULATED_DURATION and self.is_speaking:
                self.display.show_processing()
                
                # Get context
                context = self.context_manager.get_context()
                
                # Use recent audio (last 4 chunks = 2 seconds)
                audio_to_transcribe = np.concatenate(self.accumulated_audio[-4:]) \
                    if len(self.accumulated_audio) >= 4 else np.concatenate(self.accumulated_audio)
                
                # Transcribe
                text = self.transcriber.transcribe_fast(audio_to_transcribe, context)
                
                if text:
                    # Add to context
                    self.context_manager.add_text(text)
                    
                    # Display
                    self.display.show_result(text)
                    
                    # Keep only last 1 chunk for overlap
                    if len(self.accumulated_audio) > 1:
                        self.accumulated_audio = self.accumulated_audio[-1:]
                    
        elif self.is_speaking:
            # We were speaking, now in silence
            self.silence_counter += 1
            
            # Check if silence is long enough to finalize
            silence_time = self.silence_counter * (CHUNK_SIZE / SAMPLE_RATE)
            
            if silence_time >= MIN_SILENCE_DURATION:
                if self.accumulated_audio:
                    # Only transcribe if we have enough audio
                    total_duration = sum(len(a) for a in self.accumulated_audio) / SAMPLE_RATE
                    
                    if total_duration >= MIN_TRANSCRIPTION_AUDIO:
                        self.display.show_processing()
                        
                        # Get context
                        context = self.context_manager.get_context()
                        
                        # Transcribe remaining audio
                        audio_to_transcribe = np.concatenate(self.accumulated_audio)
                        text = self.transcriber.transcribe_fast(audio_to_transcribe, context)
                        
                        if text:
                            self.context_manager.add_text(text)
                            self.display.show_result(text)
                
                # Reset for next utterance
                self.accumulated_audio = []
                self.is_speaking = False
                self.silence_counter = 0
                self.display.show_listening()
        
        else:
            # Not speaking
            if self.silence_counter == 0:
                self.display.show_listening()
    
    def run(self):
        """Main application loop"""
        print("\n" + "="*50)
        print("🎤 SNAPPY TRANSCRIPTION SYSTEM")
        print("="*50)
        print(f"Device: {DEVICE.upper()}")
        print(f"Chunk size: {CHUNK_SIZE/SAMPLE_RATE:.2f}s")
        print(f"Response time: ~{MAX_ACCUMULATED_DURATION}s")
        print("="*50 + "\n")
        
        print("📋 Instructions:")
        print("• Speak naturally")
        print(f"• System responds every {MAX_ACCUMULATED_DURATION}s of speech")
        print("• Pause briefly to finalize")
        print("• Press Ctrl+C to stop\n")
        
        print("Starting...")
        
        try:
            # Start audio stream with minimal latency
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=CHUNK_SIZE,
                callback=self.audio_callback,
                latency='low'
            ):
                print("✓ Audio stream started\n")
                
                # Main processing loop
                while True:
                    try:
                        # Get next audio chunk with timeout
                        chunk = self.audio_queue.get(timeout=0.1)
                        self.process_audio_chunk(chunk)
                        
                    except queue.Empty:
                        # No audio, update display
                        if not self.is_speaking:
                            self.display.show_listening()
                        
        except KeyboardInterrupt:
            print("\n\n" + "="*50)
            print("📊 Session Complete")
            print("="*50)
            print("\n✨ Thank you!\n")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")

# ======================
# Entry Point
# ======================
if __name__ == "__main__":
    app = SnappyTranscriptionApp()
    app.run()