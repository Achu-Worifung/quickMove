import configparser
import os
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget, QListWidgetItem, QMessageBox, QDoubleSpinBox, QComboBox, QScrollArea, QWidget, QSizePolicy, QFrame, QGroupBox, QMessageBox, QSpinBox, QCheckBox
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QSettings
from pyqttoast import  ToastPreset
import torch
from util.modelmanagement import WHISPER_MODEL_INFO, fw_download_model, list_downloaded_models, delete_model, get_total_models_size, displayToast
from widgets.SearchWidget import resource_path
class Settings:
    def __init__(self, page_widget):
        super().__init__()
        self.page_widget = page_widget
        self.model_option = None
        self.settings = QSettings("MyApp", "AutomataSimulator")
        for key in self.settings.allKeys():
            print(key, "=", self.settings.value(key))
        
        self.basedir = self.settings.value("basedir")
    
    
        self.made_changes = {}

        
        # Set up the page
        self.page_setup()
        self.setup_values()
        
    def processing(self):
        print("Processing clicked")
    def populate_models(self, from_manager=False):
        print("Populating models")
        #setting the the modesl 
        self.models_dropdw = self.page_widget.findChild(QComboBox, "model")
            
        dw_models = list_downloaded_models()
        if self.model_option:
            if self.model_option == dw_models:
                print("Model list unchanged, skipping update.")
                return
        self.model_option = dw_models
        model_list = [model['name'] for model in dw_models]
        self.models_dropdw.clear()
        if model_list:
            self.models_dropdw.addItems(model_list)
            if from_manager:
                self.models_dropdw.setCurrentIndex(len(model_list)-1)  
        else:
            self.models_dropdw.addItem("No models downloaded")
    def setup_values(self):
        import torch
        self.mange_models = self.page_widget.findChild(QPushButton, "manage_model")
        self.mange_models.clicked.connect(self.open_model_manager)
        
        # Define widget lists by type
        comboBox = [self.processing, self.model, self.channel, self.chunks, self.rate]
        spinBox = [self.beam, self.core, self.best, self.silence, self.suggest_len, self.auto_searchlen, 
                   self.bible_con, self.partial_con, self.non_bible_con]
        doubleSpinBox = [self.temp, self.energy, self.minlen, self.maxlen, self.vad]
        
        self.populate_models()
        
        self.processing.addItem("GPU") if torch.cuda.is_available() else None
        
        # Load values from QSettings into widgets WITHOUT emitting change signals
        for box in comboBox:
            if box is None:
                continue
            saved_value = self.settings.value(box.objectName())
            # block signals while setting initial value so setting_changed is NOT invoked
            box.blockSignals(True)
            if saved_value is not None:
                index = box.findText(str(saved_value))
                if index != -1:
                    box.setCurrentIndex(index)
                else:
                    # Persist current widget text as default if not present in settings
                    default = box.currentText()
                    self.settings.setValue(box.objectName(), default)
                    # leave widget as-is
            else:
                default = box.currentText()
                self.settings.setValue(box.objectName(), default)
            box.blockSignals(False)
            # connect after initial value is set
            box.currentTextChanged.connect(lambda value, b=box: self.setting_changed(b))
            
        for box in spinBox:
            if box is None:
                continue
            box.blockSignals(True)
            saved_value = self.settings.value(box.objectName())
            if saved_value is not None:
                try:
                    box.setValue(int(saved_value))
                except Exception:
                    # fallback: keep current value
                    pass
            else:
                default = box.value()
                self.settings.setValue(box.objectName(), default)
            box.blockSignals(False)
            box.valueChanged.connect(lambda value, b=box: self.setting_changed(b))
            
        for box in doubleSpinBox:
            if box is None:
                continue
            box.blockSignals(True)
            saved_value = self.settings.value(box.objectName())
            if saved_value is not None:
                try:
                    box.setValue(float(saved_value))
                except Exception:
                    pass
            else:
                default = box.value()
                self.settings.setValue(box.objectName(), default)
            box.blockSignals(False)
            box.valueChanged.connect(lambda value, b=box: self.setting_changed(b))
        
        # Handle QCheckBox for multiple_trues
        if self.multiple_trues is not None:
            self.multiple_trues.blockSignals(True)
            saved_value = self.settings.value("multiple_trues")
            if saved_value is not None:
                self.multiple_trues.setChecked(saved_value == True or saved_value == "true")
            else:
                self.settings.setValue("multiple_trues", False)
            self.multiple_trues.blockSignals(False)
            self.multiple_trues.stateChanged.connect(lambda state, b=self.multiple_trues: self.setting_changed(b))
            
            # Validate confidence thresholds if multiple_trues is checked
            if self.multiple_trues.isChecked():
                self._validate_confidence_thresholds()
            
        self.settings.sync()

    def _validate_confidence_thresholds(self):
        """Validate and adjust confidence thresholds if multiple_trues is enabled."""
        if self.bible_con is None or self.partial_con is None:
            return
            
        bible_conf = self.settings.value("bible_confidence") or 60
        partial_conf = self.settings.value("partial_bible_confidence") or 50
        total = bible_conf + partial_conf
        
        if total > 100:
            # Reduce proportionally to equal 100
            factor = 100 / total
            bible_conf = int(bible_conf * factor)
            partial_conf = int(partial_conf * factor)
            self.settings.setValue("bible_confidence", bible_conf)
            self.settings.setValue("partial_bible_confidence", partial_conf)
            if self.warning is not None:
                self.warning.setText(f"Scaled down: Bible {bible_conf}% + Partial {partial_conf}% = 100%")
        else:
            if self.warning is not None:
                self.warning.setText("")

    def reload_ui_from_settings(self):
        """Reload UI widgets to reflect values currently persisted in QSettings
        without emitting change signals (used after discard or reset)."""
        comboBox = [self.processing, self.model, self.channel, self.chunks, self.rate]
        spinBox = [self.beam, self.core, self.best, self.silence, self.suggest_len, self.auto_searchlen,
                   self.bible_con, self.partial_con, self.non_bible_con]
        doubleSpinBox = [self.temp, self.energy, self.minlen, self.maxlen, self.vad]

        for box in comboBox:
            if box is None:
                continue
            saved_value = self.settings.value(box.objectName())
            box.blockSignals(True)
            if saved_value is not None:
                idx = box.findText(str(saved_value))
                if idx != -1:
                    box.setCurrentIndex(idx)
            box.blockSignals(False)
            
        for box in spinBox:
            if box is None:
                continue
            saved_value = self.settings.value(box.objectName())
            box.blockSignals(True)
            if saved_value is not None:
                try:
                    box.setValue(int(saved_value))
                except Exception:
                    pass
            box.blockSignals(False)
            
        for box in doubleSpinBox:
            if box is None:
                continue
            saved_value = self.settings.value(box.objectName())
            box.blockSignals(True)
            if saved_value is not None:
                try:
                    box.setValue(float(saved_value))
                except Exception:
                    pass
            box.blockSignals(False)
        
        # Handle QCheckBox for multiple_trues
        if self.multiple_trues is not None:
            self.multiple_trues.blockSignals(True)
            saved_value = self.settings.value("multiple_trues")
            if saved_value is not None:
                self.multiple_trues.setChecked(saved_value == True or saved_value == "true")
            self.multiple_trues.blockSignals(False)


    def open_model_manager(self):
        dialog = ModelManagerDialog(self.page_widget, self)  # Pass page_widget as parent and self as settings
        dialog.exec_()

    def setting_changed(self, obj = None):
        if obj is None:
            return
        object_name = obj.objectName()
        
        # Handle different widget types
        if isinstance(obj, QComboBox):
            new_value = obj.currentText()
        elif isinstance(obj, QCheckBox):
            new_value = obj.isChecked()
        else:
            new_value = obj.value()
            
        print(f'{object_name} changed to {new_value}')
        self.made_changes[object_name] = new_value
        # mark that there are unsaved changes
        self.settings.setValue("changesmade", True)
    
    def save_settings(self):
        for change in list(self.made_changes.keys()):
            self.settings.setValue(change, self.made_changes[change])
        # persist and clear pending changes
        self.settings.setValue("changesmade", False)
        self.settings.sync()
        print("Settings saved")
        for key in self.settings.allKeys():
            print(key, "=", self.settings.value(key))
        self.made_changes.clear()
    def discard_changes(self):
        # Clear pending changes and reload UI from persisted settings
        self.made_changes.clear()
        self.settings.setValue("changesmade", False)
        self.settings.sync()
        # ensure UI matches persisted settings and does NOT trigger setting_changed
        try:
            self.reload_ui_from_settings()
        except Exception:
            pass
    def reset_settings(self):
        print("Resetting settings...")
        import torch
        deafult_processing = "GPU" if torch.cuda.is_available() else "CPU"
        default_cores = max(1, torch.get_num_threads())
        
        self.settings.setValue('default_processing', deafult_processing)
        defaults_file = "settings.ini"
        config = configparser.ConfigParser()
        config.read(defaults_file)
        
        for key, value in config.items('general'):
            if value.lower() in ['true', 'false']:
                self.settings.setValue(key, config.getboolean('general', key))
            elif value.isdigit():
                self.settings.setValue(key, config.getint('general', key))
            else:
                try:
                    float_value = float(value)
                    self.settings.setValue(key, float_value)
                except ValueError:
                    self.settings.setValue(key, value)
        
        print('Settings reset to default values.')
        self.settings.sync()
        
        # Reload UI from the updated settings
        self.reload_ui_from_settings()
        self.made_changes.clear()
      
        
        
    def page_setup(self):
        # Transcription Model Settings
        self.processing = self.page_widget.findChild(QComboBox, "processing")
        self.model = self.page_widget.findChild(QComboBox, "model")
        self.beam = self.page_widget.findChild(QSpinBox, "beam")
        self.temp = self.page_widget.findChild(QDoubleSpinBox, "temperature")
        self.core = self.page_widget.findChild(QSpinBox, "cores")
        self.best = self.page_widget.findChild(QSpinBox, "best")
        self.vad = self.page_widget.findChild(QDoubleSpinBox, "vad_threshold")
        
        # Classification Model Settings
        self.bible_con = self.page_widget.findChild(QSpinBox, "bible_confidence")
        self.partial_con = self.page_widget.findChild(QSpinBox, "partial_bible_confidence")
        self.non_bible_con = self.page_widget.findChild(QSpinBox, "non_bible_confidence")
        self.multiple_trues = self.page_widget.findChild(QCheckBox, "multiple_trues")
        self.warning = self.page_widget.findChild(QLabel, "percentage_warning")
        self.warning.setText("")
        
        # Suggestion Settings
        self.suggest_len = self.page_widget.findChild(QSpinBox, "suggestion_length")
        self.auto_searchlen = self.page_widget.findChild(QSpinBox, "auto_length")
        
        # Recording Settings
        self.energy = self.page_widget.findChild(QDoubleSpinBox, "energy")
        self.minlen = self.page_widget.findChild(QDoubleSpinBox, "minlen")
        self.maxlen = self.page_widget.findChild(QDoubleSpinBox, "maxlen")
        self.channel = self.page_widget.findChild(QComboBox, "channel")
        self.chunks = self.page_widget.findChild(QComboBox, "chunks")
        self.rate = self.page_widget.findChild(QComboBox, "rate")
        self.silence = self.page_widget.findChild(QSpinBox, "silencelen")
        
        # Buttons
        self.savebtn = self.page_widget.findChild(QPushButton, "savechangesbtn")
        self.resetbtn = self.page_widget.findChild(QPushButton, "resetbtn")
        
        self.savebtn.clicked.connect(self.save_settings)
        self.resetbtn.clicked.connect(self.reset_settings)
        
        self.processing.currentIndexChanged.connect(self.processing_clicked)
    def processing_clicked(self):
        print("Processing clicked")
        # self.settings.setValue("processing", self.processing.currentText())
        # self.settings.sync()
        print("Processing value saved")
    



class ModelManagerDialog(QDialog):
    def __init__(self, parent=None, settings=None):
        super().__init__(parent)
        self.settings = settings  # Store the Settings instance
        self.setWindowTitle("Whisper Model Manager")
        self.setMinimumSize(600, 500)
        self.setup_ui()
        self.refresh_models()
        self.parent = parent
    def closeEvent(self, event):
        if self.settings and hasattr(self.settings, 'populate_models'):
            self.settings.populate_models(from_manager=True)
        event.accept()
    def setup_ui(self):
        layout = QVBoxLayout(self)

        # ----- Available Models Section -----
        available_group = QGroupBox("Available Models")
        available_layout = QVBoxLayout()

        # Create a container widget for the scrollable content
        scroll_content = QWidget()
        scroll_content.setLayout(available_layout)

        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scroll_content)
        scroll_area.setMaximumHeight(500)
        scroll_area.setStyleSheet("""
        QScrollArea {
            border: none;
            background-color: transparent;
        }
        QScrollBar:vertical {
            width: 10px;
            background: #f1f1f1;         /* light gray track instead of black */
            border-radius: 5px;
            margin: 0px;
        }
        /* Scroll handle (the part you drag) */
        QScrollBar::handle:vertical {
            background: #c1c1c1;         /* medium gray for good contrast */
            border-radius: 5px;
            min-height: 30px;
            transition: background 0.3s ease;
        }
        /* Hover effect for handle */
        QScrollBar::handle:vertical:hover {
            background: #a0a0a0;         /* darker when hovered */
        }
        /* Pressed state */
        QScrollBar::handle:vertical:pressed {
            background: #808080;
        }
        /* Remove arrow buttons */
        QScrollBar::sub-line:vertical,
        QScrollBar::add-line:vertical {
            height: 0;
            width: 0;
            background: none;
            border: none;
        }
        /* Remove arrow icons */
        QScrollBar::up-arrow:vertical,
        QScrollBar::down-arrow:vertical {
            background: none;
        }
        /* Optional: space at top/bottom (no buttons, just gap) */
        QScrollBar::add-page:vertical,
        QScrollBar::sub-page:vertical {
            background: none;
        }
    """)

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
            btn.setStyleSheet(
                """
                QPushButton
                {   
                border-style: outset;
                    border-width: 1px;
                    border-color: rgb(255, 255, 255);
                    padding: 4px;
                    font:20px;
                border-radius:0px;
                }
                """
            )

            available_layout.addWidget(row)

        layout.addWidget(scroll_area)

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
        downloaded_group.setMinimumHeight(200)

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
                background-color: #f0f0f0;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            QPushButton:pressed {
                background-color: #d0d0d0;
            }
            /* Scrollbar styling */
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                border: none;
                background-color: #f0f0f0;
                width: 12px;
                margin: 0px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #c0c0c0;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #a0a0a0;
            }
            QScrollBar::handle:vertical:pressed {
                background-color: #808080;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
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
            # Show starting toast, not success
            displayToast('Downloading Model', f"Starting download of '{model_name}'...", ToastPreset.INFORMATION, duration=3000)
            
            # Store thread as instance variable to prevent garbage collection
            self.download_thread = downloadThread(model_name)
            self.download_thread.finished.connect(self.on_download_finished)
            self.download_thread.start()

    def on_download_finished(self, success, message):
        if success:
            displayToast('Download Successful', message, ToastPreset.SUCCESS, duration=5000)
            self.refresh_models()
        else:
            displayToast('Download Failed', f"Error downloading model: {message}", ToastPreset.ERROR, duration=5000)
        
        # Clean up thread reference
        self.download_thread = None

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
                


class downloadThread(QThread):
    finished = pyqtSignal(bool, str)
    
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
    
    def run(self):
        try:
            cache_dir = resource_path(f"models/{self.model_name}")
            os.makedirs(cache_dir, exist_ok=True)
            model_path = fw_download_model(
                self.model_name,
                cache_dir=cache_dir,
                local_files_only=False
            )
            self.finished.emit(True, f"Model '{self.model_name}' downloaded successfully")
        except Exception as e:
            self.finished.emit(False, str(e))