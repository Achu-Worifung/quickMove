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
    # print('here is the energy threshold', threshold_value)
    audio_data = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0  # Normalize
    energy = np.sum(audio_data ** 2) / len(audio_data)  # Calculate energy
    # print('here is the energy', energy)
    return energy > threshold_value
    
def run_transcription(recording_page):
    
    # --------------------loading all the variables from settings------------------
    
    beam_size = int(settings.value('beam'))
    best_of = int(settings.value('best'))
    temperature = float(settings.value('temperature'))
    language = settings.value('language')
    # vad_filter = settings.value('vad_filter')
    energy_threshold = float(settings.value('energy_threshold'))
    # vad_parameters = settings.value('vad_parameters')
    model_size = settings.value('model').lower()
    cpu_cores = int(settings.value('cpu_cores'))
    processing = settings.value('processing')
    CHANNELS = int(settings.value('channel'))
    RATE = int(settings.value('rate'))  # 16kHz sample rate for Whisper
    CHUNK = int(settings.value('chunks'))
    silence_length = float( settings.value('silence'))  # Minimum silence duration for VAD in seconds
    min_record_len = int(settings.value('minlen'))  # Minimum recording length in seconds
    
    max_record_len = int(settings.value('maxlen'))  # Maximum recording length in seconds
    computation_type = "float16" if processing == "GPU" else "int8"
    
    line_edit = recording_page.lineEdit
    
    # -----------------------------loging to debug--------------------------
    
    # print('here is the line edit', line_edit)
    # print('here is the passed page', recording_page)
    
    # Setting up whisper model
    print(f"processing {processing} model {model_size} computational type {computation_type} cpu cores {cpu_cores}" )
    model_size = model_size.lower()
    model = WhisperModel(
        model_size,
        device=processing,
        compute_type=computation_type,
        num_workers=cpu_cores,
        cpu_threads=8,
        download_root='./models' #where dowunloaded models are stored
    )

    # Recording settings
    FORMAT = pyaudio.paInt16



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
                        beam_size=beam_size,
                        best_of=best_of,
                        temperature=temperature,
                        language=language,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500),
                        word_timestamps=False
                    )
                    
                    # print(f"Transcription for segment {counter}:")
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