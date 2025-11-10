import os
from util.util import resource_path
WHISPER_MODEL_INFO = {
    'tiny': {'size': '74.6 MB', 'description': 'Fastest, least accurate'},
    'base': {'size': '142.0 MB', 'description': 'Good balance of speed and accuracy'},
    'small': {'size': '463.7 MB', 'description': 'Better accuracy, slower'},
    'medium': {'size': '769 MB', 'description': 'High accuracy, moderate speed'},
    'large-v2': {'size': '1550 MB', 'description': 'Highest accuracy, slowest'},
    'large-v3': {'size': '1550 MB', 'description': 'Latest version, highest accuracy'}
}

def get_model_info(model_name):
    """Get size and description for a model"""
    return WHISPER_MODEL_INFO.get(model_name.lower(), {'size': 'Unknown', 'description': 'Unknown model'})

def list_downloaded_models():
    """List all downloaded models and their sizes"""
    models_dir = './models'
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
    model_path = os.path.join('./models', model_name)
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
    models_dir = './models'
    total_size = 0
    
    if os.path.exists(models_dir):
        for dirpath, dirnames, filenames in os.walk(models_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
    
    return total_size / (1024 * 1024)  # Return size in MB

def download_model(model_name):
    """Download a model using faster-whisper's download function"""
    from faster_whisper import download_model as fw_download_model
    try:
        fw_download_model(model_name, cache_dir=f"./models/{model_name}")
        return True, f"Model '{model_name}' downloaded successfully"
    except Exception as e:
        return False, f"Error downloading model: {e}"