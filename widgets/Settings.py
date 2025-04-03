from PyQt5.QtCore import QSettings
from PyQt5.QtCore import QResource  # Use QResource from QtCore, not QtGui
from PyQt5.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox
class Settings:
    def __init__(self, page_widget):
        self.page_widget = page_widget
        
        self.settings = QSettings("MyApp", "AutomataSimulator")
        
        self.basedir = self.settings.value("basedir")
        
     
      
        
        # Set up the page
        self.page_setup()
        self.setup_values()
        
        self.changedMade = False
        
    def processing(self):
        print("Processing clicked")
    def setup_values(self):
        comboBox = [self.processing, self.model, self.channel, self.chunks, self.rate]
        spinBox = [self.beam, self.core, self.best]
        doubleSpinBox = [self.temp, self.energy, self.minlen, self.maxlen]
        for box in comboBox:
            saved_value = self.settings.value(box.objectName())
            index = box.findText(saved_value)
            if index != -1:
                box.setCurrentIndex(index)
            else:
                default = box.currentText()
                self.settings.setValue(box.objectName(), default)
                box.setCurrentIndex
        for box in spinBox:
            saved_value = self.settings.value(box.objectName())
            if saved_value:
                box.setValue(int(saved_value))
            else:
                default = box.value()
                self.settings.setValue(box.objectName(), default)
        for box in doubleSpinBox:
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