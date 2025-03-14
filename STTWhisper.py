import sounddevice as sd
import numpy as np
from pynput import keyboard
from scipy.io.wavfile import write
from scipy.signal import butter, lfilter
import tempfile
import os
from faster_whisper import WhisperModel


class FasterWhisperTranscriber:
    def __init__(self, model_size="large-v2", sample_rate=44100):
        self.model_size = model_size  # Store model size as a separate attribute
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        self.sample_rate = sample_rate
        self.is_recording = False
        self.recording = np.array([], dtype='float64').reshape(0, 1)
        
        # Add configurable parameters for audio preprocessing
        self.apply_noise_reduction = True
        self.apply_normalization = True
        self.apply_low_pass_filter = True
        self.cutoff_frequency = 3000  # Hz, for low-pass filter
        
    def on_press_space(self, key):
        if key == keyboard.Key.space and not self.is_recording:
            self.is_recording = True
            self.recording = np.array([], dtype='float64').reshape(0, 1)  # Reset recording buffer
            print("Recording started")

    def on_release_space(self, key):
        if key == keyboard.Key.space and self.is_recording:
            self.is_recording = False
            print("Recording stopped")

    def record_audio(self):
        frames_per_buffer = int(self.sample_rate * 0.1)  # 100ms chunks

        while self.is_recording:
            chunk = sd.rec(frames_per_buffer, samplerate=self.sample_rate, channels=1, dtype='float64')
            sd.wait()
            self.recording = np.vstack((self.recording, chunk))

        return self.recording
    
    def butter_lowpass(self, cutoff, fs, order=5):
        """Design a lowpass filter."""
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
    
    def lowpass_filter(self, data, cutoff, fs, order=5):
        """Apply lowpass filter to data."""
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y
    
    def reduce_noise(self, data):
        """Simple noise reduction by removing low amplitude noise."""
        # Calculate noise floor from silent parts (assuming first 0.1s might be silence)
        silent_part = data[:int(0.1 * self.sample_rate)] if len(data) > int(0.1 * self.sample_rate) else data
        noise_threshold = 2.0 * np.std(silent_part) if len(silent_part) > 0 else 0.01
        
        # Apply soft noise gate
        mask = np.abs(data) < noise_threshold
        data[mask] = data[mask] * 0.1  # Reduce noise by 90% rather than complete removal
        
        return data
        
    def preprocess_audio(self, recording):
        """Apply various preprocessing techniques to improve audio quality."""
        # Convert to 1D array for processing
        audio_data = recording.flatten()
        
        # Apply preprocessing as configured
        if self.apply_low_pass_filter and len(audio_data) > 0:
            audio_data = self.lowpass_filter(audio_data, self.cutoff_frequency, self.sample_rate)
            
        if self.apply_noise_reduction and len(audio_data) > 0:
            audio_data = self.reduce_noise(audio_data)
            
        # Reshape back to original shape
        return audio_data.reshape(recording.shape)

    def save_temp_audio(self, recording):
        # Preprocess the audio
        processed_recording = self.preprocess_audio(recording)
        
        # Normalize audio (automatic if apply_normalization is True)
        if self.apply_normalization and np.max(np.abs(processed_recording)) > 0:
            processed_recording = processed_recording / np.max(np.abs(processed_recording))
        
        # Convert to int16 format with improved dynamic range
        recording_int16 = (processed_recording * 32767).astype(np.int16)

        temp_file = "test_recording.wav"
        write(temp_file, self.sample_rate, recording_int16)
        print(f"Saved processed audio to {temp_file}")
        
        return temp_file

    def transcribe_audio(self, temp_file):
        # Improved parameters for difficult audio
        segments, info = self.model.transcribe(
            temp_file, 
            beam_size=5,
            temperature=0.0,  # Use a lower temperature for more focused predictions
            compression_ratio_threshold=2.4,  # Helps with challenging audio
            condition_on_previous_text=True,  # Improve context understanding
            vad_filter=True,  # Voice activity detection to filter non-speech
            vad_parameters=dict(min_silence_duration_ms=500)  # Adjust for your needs
        )
        
        print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

        full_transcription = ""
        for segment in segments:
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
            full_transcription += segment.text + " "

        try:
            os.remove(temp_file)
        except Exception as e:
            print(f"Warning: Could not remove temporary file: {e}")
            
        return full_transcription.strip()

    def run(self):
        print("\n=== Enhanced Audio Transcription System ===")
        print("Hold the spacebar to start recording. Release it to stop and transcribe.")
        print("Current settings:")
        print(f"- Model: {self.model_size}")
        print(f"- Sample rate: {self.sample_rate} Hz")
        print(f"- Noise reduction: {'ON' if self.apply_noise_reduction else 'OFF'}")
        print(f"- Audio normalization: {'ON' if self.apply_normalization else 'OFF'}")
        print(f"- Low-pass filter: {'ON' if self.apply_low_pass_filter else 'OFF'}")
        if self.apply_low_pass_filter:
            print(f"  - Cutoff frequency: {self.cutoff_frequency} Hz")
        print("=" * 40)

        with keyboard.Listener(on_press=self.on_press_space, on_release=self.on_release_space) as listener:
            try:
                while True:
                    if not self.is_recording:
                        # Wait for recording to start - use a more CPU-friendly approach
                        import time
                        time.sleep(0.1)
                        continue
                    
                    recording = self.record_audio()
                    if len(recording) > 0:  # Check if recording contains data
                        file_path = self.save_temp_audio(recording)
                        transcription = self.transcribe_audio(file_path)

                        print("\nTranscription:\n", transcription)
                        print("\nPress the spacebar to record again or Ctrl+C to exit.")
                    else:
                        print("Recording too short or empty. Please try again.")
            except KeyboardInterrupt:
                print("\nExiting program.")
            finally:
                listener.stop()


if __name__ == "__main__":
    try:
        transcriber = FasterWhisperTranscriber()
        transcriber.run()
    except KeyboardInterrupt:
        print("\nExiting program.")
    except Exception as e:
        print(f"An error occurred: {e}")