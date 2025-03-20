import time
import torch
from faster_whisper import WhisperModel

def optimize_transcription():
    start_time = time.perf_counter()
    
    # 1. Choose the right model size based on your accuracy/speed requirements
    # model_size = "large-v2"  # Most accurate but slower
    # model_size = "medium"     # Good balance for most use cases
    # model_size = "small"     # Faster but less accurate
    model_size = "tiny"     # Faster but less accurate
    #got decent accuracy with the tiny model
    
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
        "file.wav", 
        beam_size=1,           # Lower this to 1-3 for faster results (less accurate)
        best_of=1,             # Lower this for faster results
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