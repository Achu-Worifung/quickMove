from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QProgressDialog

class ModelDownloadWorker(QThread):
    progress_signal = pyqtSignal(str)  # For status messages
    finished_signal = pyqtSignal(bool, str)  # Success, message
    
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        
    def run(self):
        try:
            self.progress_signal.emit(f"Downloading {self.model_name} model...")
            
            # Import here to avoid circular imports
            from util.modelmanagement import download_model
            success, message = download_model(self.model_name)
            
            self.finished_signal.emit(success, message)
            
        except Exception as e:
            self.finished_signal.emit(False, f"Download failed: {str(e)}")