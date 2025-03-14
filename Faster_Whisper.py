import time
import torch
from faster_whisper import WhisperModel

def optimize_transcription():
    start_time = time.perf_counter()
    
    # 1. Choose the right model size based on your accuracy/speed requirements
    # model_size = "large-v2"  # Most accurate but slower
    # model_size = "medium"     # Good balance for most use cases
    model_size = "small"     # Faster but less accurate
    
    # 2. Optimize device and compute type selection
    # Detect if CUDA is available
    if torch.cuda.is_available():
        print("Using GPU acceleration")
        # For newer GPUs with good memory
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
        
        # For older GPUs or limited memory scenarios
        # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    else:
        print("Using CPU optimization")
        # Optimize CPU performance
        # Adjust num_workers based on your CPU (typically # of cores - 1)
        cpu_cores = max(1, torch.get_num_threads())
        print(f"Using {cpu_cores} worker threads")
        model = WhisperModel(
            model_size, 
            device="cpu", 
            compute_type="int8", 
            num_workers=cpu_cores,
            cpu_threads=8,  # Adjust based on your CPU
            download_root="./models"  # Cache models locally
        )
    
    # 3. Optimize transcription parameters
    segments, info = model.transcribe(
        "recording_1.wav", 
        beam_size=1,           # Lower this to 1-3 for faster results (less accurate) was 5
        best_of=1,             # Lower this for faster results was 3
        temperature=0.0,       # Recommended for transcription
        language="en",         # Set language if known (skips detection)
        vad_filter=True,       # Filter out non-speech, can improve speed
        vad_parameters=dict(min_silence_duration_ms=500),
        initial_prompt=None,   # Can provide context for better transcription
        word_timestamps=False  # Set to False if not needed for speed
    )
    
    # 4. Efficient processing of results
    transcript = []
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        transcript.append(segment.text)
    
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")
    
    return "".join(transcript), execution_time

# 5. Additional optimization: batch processing for multiple files
def batch_process_files(file_list):
    results = {}
    for file in file_list:
        print(f"Processing {file}...")
        transcript, exec_time = optimize_transcription()
        results[file] = {"transcript": transcript, "time": exec_time}
    return results

if __name__ == "__main__":
    # Single file optimization
    transcript, exec_time = optimize_transcription()
    
    # For batch processing:
    # files = ["recording_1.wav", "recording_2.wav"]
    # results = batch_process_files(files)
    
    # here is the DeprecationWarning
#     Using CPU optimization
# Using 4 worker threads
# config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2.37k/2.37k [00:00<?, ?B/s]
# C:\Users\achuw\OneDrive\Desktop\quick hsp\.venv\Lib\site-packages\huggingface_hub\file_download.py:142: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\Users\achuw\OneDrive\Desktop\quick hsp\models\models--Systran--faster-whisper-small. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.
# To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
#   warnings.warn(message)
# vocabulary.txt: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 460k/460k [00:00<00:00, 2.70MB/s]
# tokenizer.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2.20M/2.20M [00:00<00:00, 7.35MB/s] 
# model.bin: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 484M/484M [00:30<00:00, 15.7MB/s]
# [0.11s -> 2.11s]  Hello, hello, hello.
# Execution time: 40.50 seconds█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌| 482M/484M [00:30<00:00, 16.1MB/s]
# PS C:\Users\achuw\OneDrive\Desktop\quick hsp> python Faster_Whisper.py
# Using CPU optimization
# Using 4 worker threads
# [0.11s -> 2.11s]  Hello, hello, hello.
# Execution time: 9.04 seconds #using a small model
# PS C:\Users\achuw\OneDrive\Desktop\quick hsp> 