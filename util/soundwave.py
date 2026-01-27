import time
import math
from PyQt5.QtCore import QTimer, Qt, QRectF
from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtWidgets import QLabel
import numpy as np
import threading

# Make pyaudio optional to prevent crashes if it's not available
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("PyAudio not available - soundwave will use simulated audio")

class SoundWaveLabel(QLabel):
    def __init__(self, parent=None, level_callback=None):
        super().__init__(parent)
        self.num_bars = 40
        self.wave_heights = [0.0] * self.num_bars
        self.target_heights = [0.0] * self.num_bars
        
        self.max_height = 70
        self.bar_width = 4
        self.bar_spacing = 3
        self.is_recording = False
        
        # Audio setup
        self.audio = None
        self.stream = None
        self.audio_thread = None
        self.current_level = 0.0
        self.smoothed_level = 0.0
        
        # Timers
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_wave)
        
        # Random offsets for organic movement
        self.phase_offsets = [np.random.random() * 2 * math.pi for _ in range(self.num_bars)]
        self.frequency_offsets = [0.8 + np.random.random() * 0.4 for _ in range(self.num_bars)]
        
        self.setMinimumSize(200, 100)
        
        # Retry logic
        self.retry_count = 0
        self.max_retries = 3
        
        # Thread safety flag
        self._stop_monitoring = False

        # Callback for audio level changes
        self.level_callback = level_callback

    def start_recording_visualization(self):
        """Start the audio level visualization"""
        print("Starting recording visualization")
        self.is_recording = True
        self.setText("")
        self._stop_monitoring = False
        
        if PYAUDIO_AVAILABLE:
            self.start_audio_monitoring()
        else:
            print("PyAudio not available, using simulated audio")
            self.use_simulated_audio()
            
        self.update_timer.start(33)  # ~30 FPS
        
    def stop_recording_visualization(self):
        """Stop the audio level visualization"""
        print("Stopping recording visualization")
        self.is_recording = False
        self._stop_monitoring = True
        
        # Stop audio first
        self.stop_audio_monitoring()
        
        # Then stop timer
        self.update_timer.stop()
        
        # Animate bars down smoothly
        self.smoothed_level = 0.0
        self.current_level = 0.0
        
        def clear_bars():
            still_animating = False
            for i in range(self.num_bars):
                self.wave_heights[i] *= 0.85
                if self.wave_heights[i] > 1:
                    still_animating = True
                else:
                    self.wave_heights[i] = 0
            self.update()
            
            if not still_animating:
                self.clear_timer.stop()
                self.setText("Not listening")
                self.setAlignment(Qt.AlignCenter)
                self.setStyleSheet("""
                                   color: gray;
                                      font-size: 14px;
                                      font-weight: bold;
                                      font-family: Arial;
                                      text-align: center;
                                      
                                   """)

        self.clear_timer = QTimer()
        self.clear_timer.timeout.connect(clear_bars)
        self.clear_timer.start(33)
        
    def start_audio_monitoring(self):
        """Initialize audio monitoring with better error handling"""
        if not PYAUDIO_AVAILABLE:
            print("PyAudio not available, cannot start real audio monitoring")
            self.use_simulated_audio()
            return
            
        try:
            print("Starting audio monitoring")
            self.audio = pyaudio.PyAudio()
            
            # Find a working input device
            device_index = None
            for i in range(self.audio.get_device_count()):
                try:
                    device_info = self.audio.get_device_info_by_index(i)
                    if device_info['maxInputChannels'] > 0:
                        device_index = i
                        print(f"Using audio device: {device_info['name']}")
                        break
                except Exception as e:
                    continue
            
            if device_index is None:
                raise Exception("No input device found")
            
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=44100,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=1024,
                stream_callback=None  # Use blocking read instead of callback
            )
            
            self.audio_thread = threading.Thread(target=self.monitor_audio_levels, daemon=True)
            self.audio_thread.start()
            
        except Exception as e:
            print(f"Error starting audio monitoring: {e}")
            # Fallback to simulated audio
            self.use_simulated_audio()
            
    def monitor_audio_levels(self):
        """Monitor audio levels in separate thread with improved error handling"""
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.is_recording and not self._stop_monitoring and self.stream:
            try:
                # Non-blocking read with timeout
                data = self.stream.read(512, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                
                if len(audio_data) == 0:
                    continue
                
                # Calculate RMS
                rms = np.sqrt(np.mean(audio_data.astype(float)**2))
                
                # Normalize (reduced sensitivity)
                level = min(rms / 1500.0, 1.0)
                
                # Apply curve for less sensitivity
                level = level ** 0.8
                
                # Noise gate
                if level < 0.12:
                    level = 0.0
                
                self.current_level = level
                consecutive_errors = 0  # Reset error count on success

                # Invoke the callback with the current level
                if self.level_callback:
                    self.level_callback(level)
                
            except Exception as e:
                consecutive_errors += 1
                print(f"Error reading audio: {e} (consecutive errors: {consecutive_errors})")
                self.current_level = 0.0
                
                if consecutive_errors >= max_consecutive_errors:
                    print("Too many consecutive errors, stopping audio monitoring")
                    break
                    
                time.sleep(0.1)  # Brief pause before retry
    
    def use_simulated_audio(self):
        """Use simulated audio for testing"""
        print("Using simulated audio")
        def simulate():
            base_time = time.time()
            while self.is_recording and not self._stop_monitoring:
                try:
                    t = time.time() - base_time
                    # Create interesting patterns
                    level = 0.3 + 0.3 * abs(math.sin(t * 0.5))
                    level += 0.2 * abs(math.sin(t * 1.3))
                    level += 0.15 * abs(math.sin(t * 2.7))
                    level = min(level, 1.0)
                    
                    # Add periods of silence
                    if abs(math.sin(t * 0.25)) > 0.8:
                        level = 0.0
                    
                    self.current_level = level
                    time.sleep(0.05)
                except Exception as e:
                    print(f"Error in simulated audio: {e}")
                    break
        
        sim_thread = threading.Thread(target=simulate, daemon=True)
        sim_thread.start()
                
    def update_wave(self):
        """Update wave with realistic physics"""
        if not self.is_recording:
            return
        
        # Smooth the audio level
        self.smoothed_level = self.smoothed_level * 0.7 + self.current_level * 0.3
        
        # Force to zero if below threshold
        if self.smoothed_level < 0.01:
            self.smoothed_level = 0.0
        
        current_time = time.time()
        
        for i in range(self.num_bars):
            position = i / self.num_bars
            
            # Create wave pattern with multiple frequencies
            wave1 = abs(math.sin(current_time * 2.0 + position * 8.0))
            wave2 = abs(math.sin(current_time * 1.3 + position * 5.0 + self.phase_offsets[i]))
            wave3 = abs(math.sin(current_time * self.frequency_offsets[i] + position * 3.0))
            
            # Combine waves
            combined_wave_shape = (wave1 * 0.5 + wave2 * 0.3 + wave3 * 0.2)
            
            # Boost center bars
            center_boost = 1.0 - abs(position - 0.5) * 0.5
            
            # Scale by audio level
            target = combined_wave_shape * self.max_height * self.smoothed_level * center_boost
            self.target_heights[i] = max(0, target)
            
            # Physics-based animation
            if self.target_heights[i] > self.wave_heights[i]:
                # Fast rise
                self.wave_heights[i] += (self.target_heights[i] - self.wave_heights[i]) * 0.4
            else:
                # Slower fall
                self.wave_heights[i] += (self.target_heights[i] - self.wave_heights[i]) * 0.15
    
        self.update()
            
    def stop_audio_monitoring(self):
        """Stop audio monitoring safely"""
        self._stop_monitoring = True
        
        # Wait briefly for thread to notice the flag
        time.sleep(0.1)
        
        if self.stream:
            try:
                if self.stream.is_active():
                    self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                print(f"Error closing stream: {e}")
            finally:
                self.stream = None
                
        if self.audio:
            try:
                self.audio.terminate()
            except Exception as e:
                print(f"Error terminating audio: {e}")
            finally:
                self.audio = None

    def clear_buffers(self):
        """Clear internal buffers and reset state to free memory (safe to call when paused)"""
        print("Clearing soundwave buffers")
        # Reset wave data
        self.wave_heights = [0.0] * self.num_bars
        self.target_heights = [0.0] * self.num_bars
        
        # Reset audio levels
        self.current_level = 0.0
        self.smoothed_level = 0.0
        
        # Reset phase offsets for fresh animation on resume
        self.phase_offsets = [np.random.random() * 2 * math.pi for _ in range(self.num_bars)]
        
        # Force update
        self.update()

    def _get_bar_color(self, height_ratio):
        """Calculate bar color based on height"""
        color_low = QColor(0, 100, 255)      # Deep Blue
        color_mid = QColor(150, 50, 255)     # Purple
        color_high = QColor(255, 80, 150)    # Bright Magenta

        if height_ratio < 0.5:
            ratio = height_ratio * 2
            r = int(color_low.red() + ratio * (color_mid.red() - color_low.red()))
            g = int(color_low.green() + ratio * (color_mid.green() - color_low.green()))
            b = int(color_low.blue() + ratio * (color_mid.blue() - color_low.blue()))
        else:
            ratio = (height_ratio - 0.5) * 2
            r = int(color_mid.red() + ratio * (color_high.red() - color_mid.red()))
            g = int(color_mid.green() + ratio * (color_high.green() - color_mid.green()))
            b = int(color_mid.blue() + ratio * (color_high.blue() - color_mid.blue()))
        
        return QColor(r, g, b)
            
    def paintEvent(self, event):
        """Custom paint event with beautiful gradient bars"""
        if not self.is_recording and all(h == 0 for h in self.wave_heights):
            super().paintEvent(event)
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Calculate positioning
        total_width = self.num_bars * (self.bar_width + self.bar_spacing) - self.bar_spacing
        start_x = (self.width() - total_width) // 2
        center_y = self.height() // 2

        for i, height in enumerate(self.wave_heights):
            if height < 1:
                continue

            x = start_x + i * (self.bar_width + self.bar_spacing)
            bar_height = int(height)
            
            # Normalize height for color
            height_ratio = min(height / self.max_height, 1.0)
            color = self._get_bar_color(height_ratio)
            
            # Draw glow effect
            glow_color = QColor(color.red(), color.green(), color.blue(), 70)
            painter.setPen(Qt.NoPen)
            painter.setBrush(glow_color)
            painter.drawRoundedRect(
                QRectF(x - 2, center_y - bar_height // 2 - 2, 
                       self.bar_width + 4, bar_height + 4),
                4, 4
            )
            
            # Draw main bar
            painter.setBrush(color)
            painter.drawRoundedRect(
                QRectF(x, center_y - bar_height // 2, 
                       self.bar_width, bar_height),
                2, 2
            )

        painter.end()
        
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            self.stop_audio_monitoring()
        except:
            pass