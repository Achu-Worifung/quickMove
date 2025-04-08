from PyQt5.QtCore import QSettings
from PyQt5.QtCore import QResource  # Use QResource from QtCore, not QtGui
from PyQt5.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox, QPushButton
class Settings:
    def __init__(self, page_widget):
        super().__init__()
        self.page_widget = page_widget
        
        self.settings = QSettings("MyApp", "AutomataSimulator")
        
        self.basedir = self.settings.value("basedir")
        
    
    
        self.changes = []
        
        # Set up the page
        self.page_setup()
        self.setup_values()
        
        self.changedMade = False
        self.changes.clear() 
    def processing(self):
        print("Processing clicked")
    def setup_values(self):
        comboBox = [self.processing, self.model, self.channel, self.chunks, self.rate]
        spinBox = [self.beam, self.core, self.best,self.silence]
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
            
        # saved_processing = self.settings.value("processing")
        # index = self.processing.findText(saved_processing)
        # if index != -1:
        #     self.processing.setCurrentIndex(index)
        # else:
        #     self.processing.setCurrentIndex(0)
    def setting_changed(self, obj = None):
        self.settings.setValue("changesmade", True)
        self.changes.append(obj)
        self.settings.sync()
    
    def save_settings(self):
        for box in self.changes:
            if isinstance(box, QComboBox):
                self.settings.setValue(box.objectName(), box.currentText())
                print("Combo box name", box.objectName(), "value", box.currentText())
            elif isinstance(box, QSpinBox):
                self.settings.setValue(box.objectName(), box.value())
            elif isinstance(box, QDoubleSpinBox):
                self.settings.setValue(box.objectName(), box.value())
        self.settings.setValue("changesmade", False)
        self.settings.sync()
        self.settings.sync()
        self.changes.clear()
    
    def reset_settings(self):
        defaults = self.settings.value("default_settings")
        print("Defaults", defaults)
        settings = [self.processing, self.model, self.beam, self.temp, self.core, self.best, self.energy, self.minlen, self.maxlen, self.channel, self.chunks, self.rate, self.silence]
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
        self.settings.setValue("changesmade", False)
        self.settings.sync()
        self.changes.clear()
                
      
        
        
        
    def page_setup(self):
        self.processing = self.page_widget.findChild(QComboBox, "processing")
        self.model = self.page_widget.findChild(QComboBox, "model")
        self.beam = self.page_widget.findChild(QSpinBox, "beam")
        self.temp = self.page_widget.findChild(QDoubleSpinBox, "temperature")
        self.core = self.page_widget.findChild(QSpinBox, "cores")
        self.best = self.page_widget.findChild(QSpinBox, "best")
        
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