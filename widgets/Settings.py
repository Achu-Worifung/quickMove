from PyQt5.QtCore import QSettings
from PyQt5.QtCore import QResource  # Use QResource from QtCore, not QtGui
from PyQt5.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox, QPushButton
import os
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget, QListWidgetItem, QMessageBox, QProgressBar
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QMessageBox, QGroupBox, QFrame, QSizePolicy, QWidget
)
from widgets.progressDialog import ModelDownloadWorker
from PyQt5.QtCore import Qt
from util.modelmanagement import WHISPER_MODEL_INFO, list_downloaded_models, delete_model, get_total_models_size, download_model
class Settings:
    def __init__(self, page_widget):
        super().__init__()
        self.page_widget = page_widget
        
        self.settings = QSettings("MyApp", "AutomataSimulator")
        
        self.basedir = self.settings.value("basedir")
    
    
        self.made_changes = {}

        
        # Set up the page
        self.page_setup()
        self.setup_values()
        
    def processing(self):
        print("Processing clicked")
    def setup_values(self):
        self.mange_models = self.page_widget.findChild(QPushButton, "manage_model")
        self.mange_models.clicked.connect(self.open_model_manager)
        comboBox = [self.processing, self.model, self.channel, self.chunks, self.rate]
        spinBox = [self.beam, self.core, self.best,self.silence, self.suggest_len, self.auto_searchlen]
        doubleSpinBox = [self.temp, self.energy, self.minlen, self.maxlen]
        for box in comboBox:
            print("Box name", box)
            saved_value = self.settings.value(box.objectName())
            print("Saved value", saved_value, " here si the type of the saved value", type(saved_value))
            
            
            box.currentTextChanged.connect(lambda value, b=box: self.setting_changed(b))
            index = box.findText(str(saved_value))
                
            if index != -1:
                box.setCurrentIndex(index)
            else:
                default = box.currentText()
                self.settings.setValue(box.objectName(), default)
                box.setCurrentIndex
        for box in spinBox:
            box.valueChanged.connect(lambda value, b=box: self.setting_changed(b))
            print("Box name", box)
            saved_value = self.settings.value(box.objectName())
            if saved_value:
                box.setValue(int(saved_value))
            else:
                default = box.value()
                self.settings.setValue(box.objectName(), default)
        for box in doubleSpinBox:
            print("Box name", box.objectName())
            box.valueChanged.connect(lambda value, b=box: self.setting_changed(b))
            saved_value = self.settings.value(box.objectName())
            if saved_value:
                box.setValue(float(saved_value))
            else:
                default = box.value()
                self.settings.setValue(box.objectName(), default)
            
        self.settings.sync()
    def open_model_manager(self):
        # Pass the page_widget (which is a QWidget) as the parent instead of self
        dialog = ModelManagerDialog(self.page_widget)
        dialog.exec_()

    def setting_changed(self, obj = None):
        object_name = obj.objectName()
        new_value = (obj.currentText() if isinstance(obj, QComboBox) else obj.value())
        self.made_changes[object_name] = new_value
    
    def save_settings(self):
        for change in self.made_changes:
            self.settings.setValue(change, self.made_changes[change])
        self.settings.setValue("changesmade", False)
        self.made_changes.clear()
    
    def reset_settings(self):
        defaults = self.settings.value("default_settings")
        print("Defaults", defaults)
        settings = [self.processing, self.model, self.beam, self.temp, self.core, self.best, self.energy, self.minlen, self.maxlen, self.channel, self.chunks, self.rate, self.silence, self.suggest_len, self.auto_searchlen]
        index = 0
        for box in settings:
            if isinstance(box, QComboBox):
                box.setCurrentText(defaults[index])
                self.settings.setValue(box.objectName(), defaults[index])
            elif isinstance(box, QSpinBox):
                box.setValue(int(defaults[index]))
                self.settings.setValue(box.objectName(), defaults[index])
            else:
                box.setValue(float(defaults[index]))
                self.settings.setValue(box.objectName(), defaults[index])
            index += 1
                
      
        
        
        
    def page_setup(self):
        self.processing = self.page_widget.findChild(QComboBox, "processing")
        self.model = self.page_widget.findChild(QComboBox, "model")
        self.beam = self.page_widget.findChild(QSpinBox, "beam")
        self.temp = self.page_widget.findChild(QDoubleSpinBox, "temperature")
        self.core = self.page_widget.findChild(QSpinBox, "cores")
        self.best = self.page_widget.findChild(QSpinBox, "best")
        self.suggest_len = self.page_widget.findChild(QSpinBox, "suggestion_length")
        self.auto_searchlen = self.page_widget.findChild(QSpinBox, "auto_length")
        
        self.energy = self.page_widget.findChild(QDoubleSpinBox, "energy")
        self.minlen = self.page_widget.findChild(QDoubleSpinBox, "minlen")
        self.maxlen = self.page_widget.findChild(QDoubleSpinBox, "maxlen")
        self.channel = self.page_widget.findChild(QComboBox, "channel")
        self.chunks = self.page_widget.findChild(QComboBox, "chunks")
        self.rate = self.page_widget.findChild(QComboBox, "rate")
        self.silence = self.page_widget.findChild(QSpinBox, "silencelen")
        
        self.savebtn = self.page_widget.findChild(QPushButton, "savechangesbtn")
        self.resetbtn = self.page_widget.findChild(QPushButton, "resetbtn")
        
        self.savebtn.clicked.connect(self.save_settings)
        self.resetbtn.clicked.connect(self.reset_settings)
        
        self.processing.currentIndexChanged.connect(self.processing_clicked)
    #    self.processing.currentIndexChanged.connect(self.processing_clicked)
    #    self.model = self.page_widget.value("model")
    #    self.beam = self.page_widget.value("beam")
    #    self.temp = self.page_widget.value("temperature")
    #    self.best = self.page_widget.value("best")
       
    #    self.energy =  self.page_widget.value("energy")
    #    self.minlen = self.page_widget.value("minlen")
    #    self.maxlen = self.page_widget.value("maxlen")
    #    self.channel = self.page_widget.value("channel")
    #    self.rate = self.page_widget.value("rate")
    def processing_clicked(self):
        print("Processing clicked")
        # self.settings.setValue("processing", self.processing.currentText())
        # self.settings.sync()
        print("Processing value saved")
    




class ModelManagerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Whisper Model Manager")
        self.setMinimumSize(600, 500)
        self.setup_ui()
        self.refresh_models()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)

        # ----- Available Models Section -----
        available_group = QGroupBox("Available Models")
        available_layout = QVBoxLayout()

        for model_name, info in WHISPER_MODEL_INFO.items():
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)

            # Model details
            label = QLabel(
                f"<b>{model_name}</b> ({info['size']})<br><i>{info['description']}</i>"
            )
            label.setWordWrap(True)
            row_layout.addWidget(label, stretch=1)

            # Download button
            btn = QPushButton("⬇ Download")
            btn.clicked.connect(lambda _, m=model_name: self.download_model(m))
            row_layout.addWidget(btn)

            available_layout.addWidget(row)

        available_group.setLayout(available_layout)
        layout.addWidget(available_group)

        # Separator line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)

        # ----- Downloaded Models Section -----
        downloaded_group = QGroupBox("Downloaded Models")
        downloaded_layout = QVBoxLayout()

        self.models_list = QListWidget()
        self.models_list.setSelectionMode(QListWidget.SingleSelection)
        self.models_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        downloaded_layout.addWidget(self.models_list)

        self.total_size_label = QLabel("Total models size: 0 MB")
        downloaded_layout.addWidget(self.total_size_label)

        downloaded_group.setLayout(downloaded_layout)
        layout.addWidget(downloaded_group)

        # ----- Buttons -----
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.refresh_btn = QPushButton("↻ Refresh")
        self.refresh_btn.clicked.connect(self.refresh_models)
        button_layout.addWidget(self.refresh_btn)

        self.delete_btn = QPushButton("🗑 Delete Selected")
        self.delete_btn.clicked.connect(self.delete_selected_model)
        button_layout.addWidget(self.delete_btn)

        self.close_btn = QPushButton("✖ Close")
        self.close_btn.clicked.connect(self.close)
        button_layout.addWidget(self.close_btn)

        layout.addLayout(button_layout)

        # Light styling
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ccc;
                border-radius: 6px;
                margin-top: 12px;
                padding: 8px;
            }
            QListWidget {
                border: 1px solid #bbb;
                border-radius: 4px;
            }
            QPushButton {
                padding: 6px 18px;
                border: 1px solid #888;
                border-radius: 4px;
            }
        """)

    def refresh_models(self):
        self.models_list.clear()
        downloaded_models = list_downloaded_models()

        for model in downloaded_models:
            item_text = f"{model['name']}   •   {model['size_str']}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, model['name'])
            self.models_list.addItem(item)

        total_size = get_total_models_size()
        self.total_size_label.setText(f"Total models size: {total_size:.1f} MB")

    def download_model(self, model_name):
        reply = QMessageBox.question(
            self,
            "Download Model",
            f"Do you want to download the '{model_name}' model?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        if reply == QMessageBox.Yes:
            success, message = download_model(model_name)  # <-- your function
            if success:
                QMessageBox.information(self, "Success", message)
                self.refresh_models()
            else:
                QMessageBox.critical(self, "Error", message)

    def delete_selected_model(self):
        current_item = self.models_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Warning", "Please select a model to delete")
            return

        model_name = current_item.data(Qt.UserRole)

        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete the '{model_name}' model?\n"
            "This action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            success, message = delete_model(model_name)
            if success:
                QMessageBox.information(self, "Success", message)
                self.refresh_models()
            else:
                QMessageBox.critical(self, "Error", message)