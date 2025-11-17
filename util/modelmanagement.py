import os
from PyQt5 import QtWidgets
from pyqttoast import Toast, ToastPreset
from util.util import resource_path
from faster_whisper.utils import download_model as fw_download_model
import time

WHISPER_MODEL_INFO = {
    # Original OpenAI Models
    'tiny.en': {'size': '75 MB', 'description': 'Fastest, least accurate, english only'},
    'tiny': {'size': '75 MB', 'description': 'Fastest, least accurate, multilingual'},
    'base.en': {'size': '142 MB', 'description': 'Good balance of speed and accuracy, english only'},
    'base': {'size': '142 MB', 'description': 'Good balance of speed and accuracy, multilingual'},
    'small.en': {'size': '466 MB', 'description': 'Better accuracy, slower, english only'},
    'small': {'size': '466 MB', 'description': 'Better accuracy, slower, multilingual'},
    'medium.en': {'size': '1.5 GB', 'description': 'High accuracy, moderate speed, english only'},
    'medium': {'size': '1.5 GB', 'description': 'High accuracy, moderate speed, multilingual'},
    'large-v1': {'size': '3.1 GB', 'description': 'Original large model, highest accuracy, slowest'},
    'large-v2': {'size': '3.1 GB', 'description': 'Improved large model, highest accuracy, slowest'},
    'large-v3': {'size': '3.1 GB', 'description': 'Latest large model, best accuracy especially for non-English'},
    'large': {'size': '3.1 GB', 'description': 'Alias for the latest large model (large-v3)'},
    
    # Distilled Models (Smaller & Faster)
    'distil-large-v2': {'size': '756 MB', 'description': '50% smaller and 60% faster than Large-v2 with minimal accuracy loss'},
    'distil-medium.en': {'size': '402 MB', 'description': '4x smaller and 5x faster than Medium.en, english only'},
    'distil-small.en': {'size': '166 MB', 'description': '2.5x smaller and 2x faster than Small.en, english only'},
    'distil-large-v3': {'size': '756 MB', 'description': 'Distilled version of Large-v3, great speed/size/accuracy trade-off'},
    
    # Turbo Model (New Architecture)
    'turbo': {'size': '783 MB', 'description': 'New efficient architecture. Very fast and highly accurate.'},
}

def get_model_info(model_name):
    """Get size and description for a model"""
    return WHISPER_MODEL_INFO.get(model_name.lower(), {'size': 'Unknown', 'description': 'Unknown model'})

def list_downloaded_models():
    """List all downloaded models and their sizes"""
    models_dir = resource_path('models')
    downloaded_models = []
    
    if os.path.exists(models_dir):
        for item in os.listdir(models_dir):
            model_path = os.path.join(models_dir, item)
            if os.path.isdir(model_path):
                # Calculate folder size
                total_size = 0
                for dirpath, dirnames, filenames in os.walk(model_path):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        total_size += os.path.getsize(filepath)
                
                size_mb = total_size / (1024 * 1024)
                downloaded_models.append({
                    'name': item,
                    'path': model_path,
                    'size_mb': size_mb,
                    'size_str': f"{size_mb:.1f} MB"
                })
    
    return downloaded_models

def delete_model(model_name):
    """Delete a downloaded model"""
    import shutil
    model_path = os.path.join(resource_path('models'), model_name)
    if os.path.exists(model_path):
        try:
            shutil.rmtree(model_path)
            return True, f"Model '{model_name}' deleted successfully"
        except Exception as e:
            return False, f"Error deleting model: {e}"
    else:
        return False, f"Model '{model_name}' not found"

def get_total_models_size():
    """Get total size of all downloaded models"""
    models_dir = resource_path('models')
    total_size = 0
    
    if os.path.exists(models_dir):
        for dirpath, dirnames, filenames in os.walk(models_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
    
    return total_size / (1024 * 1024)  # Return size in MB
import os, sys, socket, requests, pprint, certifi




def download_model(model_name):
    """Download a model using faster-whisper's download function"""

    try:
        cache_dir = resource_path(f"models/{model_name}")
        os.makedirs(cache_dir, exist_ok=True)
        
        model_path = fw_download_model(
            model_name, 
            cache_dir=cache_dir,
            local_files_only=False  
        )
        toast = Toast()
        toast.setTitle('Downloading Model')
        toast.setText(f"Model '{model_name}' downloaded successfully to model/{model_name}.")
        toast.setDuration(5000)
        toast.applyPreset(ToastPreset.SUCCESS)
        toast.show()
        return True, f"Model '{model_name}' downloaded successfully to model/{model_name}."
    except Exception as e:
        toast = Toast()
        toast.setTitle('Downloading Model')
        toast.setText(f"Error downloading model '{model_name}': {str(e)}")
        toast.setDuration(5000)
        toast.applyPreset(ToastPreset.ERROR)
        toast.show()
        return False, f"Error downloading model: {str(e)}"