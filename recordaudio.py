import pyaudio
import wave
 
FORMAT = pyaudio.paInt16
CHANNELS = 2 #streo use 1 for mono
RATE = 44100 
CHUNK = 1024 #udio frames per buffer 
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "file.wav"
 
audio = pyaudio.PyAudio()
 
# start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)
print ("recording...")
frames = []
 
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)
print ("finished recording")
 
 
# stop Recording
stream.stop_stream()
stream.close()
audio.terminate()
 
waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()

# CHUNK Size	Effect
# 256	Very low latency, but high CPU usage ⚡
# 512	Low latency, good for real-time applications
# 1024	Balanced performance ✅ (Common default)
# 2048+	Lower CPU usage, but more delay (good for batch processing)