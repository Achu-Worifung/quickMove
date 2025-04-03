import pyaudio
import numpy as np
from faster_whisper import WhisperModel
import torch
import wave
import os
from PyQt5.QtCore import QSettings
import tempfile


settings = QSettings("MyApp", "AutomataSimulator")
    
def get_energy_threshold(audio_data, threshold_value=0.05):
    audio_data = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0  # Normalize
    energy = np.sum(audio_data ** 2) / len(audio_data)  # Calculate energy
    return energy > threshold_value
    
def run_transcription(recording_page):
    
    beam_size = settings.value('beam')
    best_of = settings.value('best')
    temperature = settings.value('temperature')
    language = settings.value('language')
    vad_filter = settings.value('vad_filter')
    energy_threshold = settings.value('energy_threshold')
    vad_parameters = settings.value('vad_parameters')
    model_size = settings.value('model_size')
    cpu_cores = settings.value('cpu_cores')
    processing = settings.value('processing')
    CHANNELS = settings.value('channel')
    RATE = settings.value('rate')  # 16kHz sample rate for Whisper
    CHUNK = settings.value('chunks')
    silence_length = settings.value('silence')  # Minimum silence duration for VAD in seconds
    min_record_len = settings.value('minlen')  # Minimum recording length in seconds
    max_record_len = settings.value('maxlen')  # Maximum recording length in seconds
    
    line_edit = recording_page.lineEdit
    print('here is the line edit', line_edit)
    print('here is the passed page', recording_page)
    # Setting up whisper model
    
    model = WhisperModel(
        model_size,
        device=processing,
        compute_type='int8',
        num_workers=cpu_cores,
        cpu_threads=8,
        download_root='./models'
    )

    # Recording settings
    FORMAT = pyaudio.paInt16

    energy_threshold = 0.001  # Energy threshold for VAD


    # Initialize PyAudio
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("Recording...")
    counter = 1

    try:
        while True:
            frames = []
            silent_frames = 0
            has_speech = False
            
            print("Waiting for speech...")
            
            # Wait for speech to start
            while True:
                data = stream.read(CHUNK, exception_on_overflow=False)
                sound_detected = get_energy_threshold(data, threshold_value=energy_threshold)
                
                if sound_detected:
                    # print("Speech detected, recording...")
                    frames.append(data)
                    has_speech = True
                    break
            
            # Continue recording until silence is detected
            while True:
                data = stream.read(CHUNK, exception_on_overflow=False)
                sound_detected = get_energy_threshold(data, threshold_value=energy_threshold)
                recording_length = len(frames) * CHUNK / RATE
                
                frames.append(data)
                
                if sound_detected:
                    silent_frames = 0
                else:
                    silent_frames += 1
                
                # If silence detected for a long time and recording length is minimum
                silence_frames_threshold = int(silence_length * RATE / CHUNK)
                if silent_frames >= silence_frames_threshold:
                    if recording_length >= min_record_len:
                        # print(f"Silence detected after {recording_length:.2f} seconds of recording")
                        break
                
                # If we've reached the max recording length break anyway
                if recording_length >= max_record_len:
                    # print(f"Max recording length reached ({max_record_len} seconds)")
                    break
            
            # Save and transcribe the audio only if speech was detected
            if has_speech and len(frames) > 0:
                temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_filename = temp_file.name
                print('here is the temp file', temp_filename)
                temp_file.close()
                
                
                
                try:
                    with wave.open(temp_filename, 'wb') as wf:
                        wf.setnchannels(CHANNELS)
                        wf.setsampwidth(audio.get_sample_size(FORMAT))
                        wf.setframerate(RATE)
                        wf.writeframes(b''.join(frames))
                    
                    print(f"Transcribing segment {counter}...")
                    segments, _ = model.transcribe(
                        temp_filename,
                        beam_size=5,
                        best_of=1,
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
                        line_edit.setText(line_edit.text() + segment_text.strip() + " ")
                    else:
                        # print("No speech detected in this segment.")
                        pass
                        
                    counter += 1
                    
                except Exception as e:
                    # print(f"Error processing audio: {e}")
                    pass
                finally:
                    # Clean up the temporary file
                    try:
                        os.remove(temp_filename)
                    except:
                        pass
            
    except KeyboardInterrupt:
        print("Recording stopped.")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()