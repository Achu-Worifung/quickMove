import sys
import time
import math
from PyQt5.QtCore import QTimer, Qt, QRectF
from PyQt5.QtGui import QPainter, QColor, QLinearGradient, QPainterPath, QPen
from PyQt5.QtWidgets import QLabel
import pyaudio
import numpy as np
import threading

class SoundWaveLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.num_bars = 40  # More bars for smoother look
        self.wave_heights = [0.0] * self.num_bars
        self.target_heights = [0.0] * self.num_bars
        self.velocities = [0.0] * self.num_bars
        
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
        
        #retry logic
        self.retry_count = 0
        self.max_retries = 20
        
    def start_recording_visualization(self):
        """Start the audio level visualization"""
        print("Starting recording visualization")
        self.is_recording = True
        self.setText("")
        self.start_audio_monitoring()
        self.update_timer.start(33)  # ~30 FPS for smooth animation
        
    def stop_recording_visualization(self):
        """Stop the audio level visualization"""
        print("Stopping recording visualization")
        self.is_recording = False
        self.stop_audio_monitoring()
        self.update_timer.stop()
        # Animate bars down to zero
        self.smoothed_level = 0.0
        self.current_level = 0.0
        
        # We can create a small timer to animate the bars down smoothly
        def clear_bars():
            still_animating = False
            for i in range(self.num_bars):
                self.wave_heights[i] *= 0.8
                if self.wave_heights[i] < 1:
                    self.wave_heights[i] = 0
                else:
                    still_animating = True
            self.update()
            if not still_animating:
                self.clear_timer.stop()
                self.setText("Not listening")

        self.clear_timer = QTimer()
        self.clear_timer.timeout.connect(clear_bars)
        self.clear_timer.start(33)

        
    def start_audio_monitoring(self):
        """Initialize audio monitoring"""
        try:
            print("Starting audio monitoring")
            self.audio = pyaudio.PyAudio()
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=44100,
                input=True,
                frames_per_buffer=1024
            )
            
            self.audio_thread = threading.Thread(target=self.monitor_audio_levels)
            self.audio_thread.daemon = True
            self.try_count = 0
            self.audio_thread.start()
            
        except Exception as e:
            print(f"Error starting audio monitoring: {e}")
            # Fallback to simulated audio
            # self.use_simulated_audio()
            pass
            
    def monitor_audio_levels(self):
        """Monitor audio levels in separate thread"""
        while self.is_recording and self.stream:
            try:
                data = self.stream.read(512, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                
                if len(audio_data) == 0:
                    continue
                
                # Calculate RMS
                rms = np.sqrt(np.mean(audio_data.astype(float)**2))
                
                # Normalize (adjust sensitivity here - reduced for less sensitivity)
                level = min(rms / 1500.0, 1.0)  # Increased divisor = less sensitive
                
                # Add bass boost effect (emphasize louder sounds)
                level = level ** 0.8  # Increased exponent = even less sensitive
                
                # Noise gate - if below threshold, set to zero
                if level < 0.12:
                    level = 0.0
                
                self.current_level = level
                
            except Exception as e:
                print(f"Error reading audio: {e}")
                self.current_level = 0.0 # Ensure level goes to 0 on error
                if self.retry_count < self.max_retries:
                    self.retry_count += 1
                    print(f"Retrying audio monitoring ({self.retry_count}/{self.max_retries})")
                    time.sleep(1)
                    self.start_recording_visualization() # Restart monitoring on error
                else :
                    self.setText("Not listening")
                    self.is_recording = False
                    self.stream = None
                    self.audio = None
                break
    
    def use_simulated_audio(self):
        """Use simulated audio for testing"""
        print("Using simulated audio")
        def simulate():
            base_time = time.time()
            while self.is_recording:
                t = time.time() - base_time
                # Create interesting patterns with periods of silence
                level = 0.3 + 0.3 * abs(math.sin(t * 0.5))
                level += 0.2 * abs(math.sin(t * 1.3))
                level += 0.15 * abs(math.sin(t * 2.7))
                level = min(level, 1.0)
                
                # Add periods of silence (noise gate simulation)
                if abs(math.sin(t * 0.25)) > 0.8:
                     level = 0.0
                
                self.current_level = level
                time.sleep(0.05)
        
        sim_thread = threading.Thread(target=simulate)
        sim_thread.daemon = True
        sim_thread.start()
                
    def update_wave(self):
        """Update wave with realistic physics"""
        if not self.is_recording:
            return
        
        # Smooth the audio level
        self.smoothed_level = self.smoothed_level * 0.7 + self.current_level * 0.3
        
        # If audio is below a tiny threshold, force it to zero to prevent residual noise
        if self.smoothed_level < 0.01:
             self.smoothed_level = 0.0
        
        current_time = time.time()
        
        for i in range(self.num_bars):
            # Create wave pattern with multiple frequencies
            position = i / self.num_bars
            
            # Main wave (travels across bars)
            wave1 = abs(math.sin(current_time * 2.0 + position * 8.0))
            
            # Secondary wave (different frequency)
            wave2 = abs(math.sin(current_time * 1.3 + position * 5.0 + self.phase_offsets[i]))
            
            # Tertiary subtle wave
            wave3 = abs(math.sin(current_time * self.frequency_offsets[i] + position * 3.0))
            
            # Combine waves to create the shape
            combined_wave_shape = (wave1 * 0.5 + wave2 * 0.3 + wave3 * 0.2)
            
            # Boost center bars for a classic look
            center_boost = 1.0 - abs(position - 0.5) * 0.5
            
            # *** FIX: The target height is now directly scaled by the smoothed audio level.
            # If smoothed_level is 0, the target is 0, making the visualization flat.
            target = combined_wave_shape * self.max_height * self.smoothed_level * center_boost
            
            self.target_heights[i] = max(0, target) # Ensure target is not negative
            
            # Physics-based spring animation (quick attack, slower decay)
            if self.target_heights[i] > self.wave_heights[i]:
                # Fast rise (attack)
                self.wave_heights[i] += (self.target_heights[i] - self.wave_heights[i]) * 0.4
            else:
                # Slower fall (decay)
                self.wave_heights[i] += (self.target_heights[i] - self.wave_heights[i]) * 0.15
    
        self.update()
            
    def stop_audio_monitoring(self):
        """Stop audio monitoring"""
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                print(f"Error closing stream: {e}")
        if self.audio:
            self.audio.terminate()
        self.stream = None
        self.audio = None

    def _get_bar_color(self, height_ratio):
        """
        Calculates bar color based on its height, creating a vibrant gradient.
        """
        # Define our color stops: Blue -> Purple -> Magenta
        color_low = QColor(0, 100, 255)   # Deep Blue
        color_mid = QColor(150, 50, 255)  # Purple
        color_high = QColor(255, 80, 150) # Bright Magenta

        if height_ratio < 0.5:
            # Interpolate between low and mid
            ratio = height_ratio * 2
            r = int(color_low.red() + ratio * (color_mid.red() - color_low.red()))
            g = int(color_low.green() + ratio * (color_mid.green() - color_low.green()))
            b = int(color_low.blue() + ratio * (color_mid.blue() - color_low.blue()))
        else:
            # Interpolate between mid and high
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

        # *** COLOR IMPROVEMENT: Dark background for colors to pop
        # painter.fillRect(self.rect(), QColor(20, 20, 35))

        # Calculate positioning
        total_width = self.num_bars * (self.bar_width + self.bar_spacing) - self.bar_spacing
        start_x = (self.width() - total_width) // 2
        center_y = self.height() // 2

        for i, height in enumerate(self.wave_heights):
            if height < 1:
                continue

            x = start_x + i * (self.bar_width + self.bar_spacing)
            bar_height = int(height)
            
            # Normalize height for color calculation (clamp to 1.0)
            height_ratio = min(height / self.max_height, 1.0)
            
            # *** COLOR IMPROVEMENT: Get color from the new gradient method
            color = self._get_bar_color(height_ratio)
            
            # Draw glow effect (more effective on a dark background)
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