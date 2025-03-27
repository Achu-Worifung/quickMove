from PyQt5.QtCore import QSettings
from PyQt5.QtCore import QResource  # Use QResource from QtCore, not QtGui
import sys

class Settings:
    def __init__(self, page_widget):
        self.page_widget = page_widget
        
        self.settings = QSettings("MyApp", "AutomataSimulator")
        
        self.basedir = self.settings.value("basedir")
        
        # Correct way to register resources
        resource_path = self.basedir + "main_icons.rcc"  # Note: use .rcc, not .py
        if not QResource.registerResource(resource_path):
            print(f"Failed to register resource: {resource_path}")
        
        # Set up the page
        self.page_setup()
        
    def page_setup(self):
        # Your page setup code here
        pass