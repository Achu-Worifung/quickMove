from PyQt5 import QtCore

settings = QtCore.QSettings("MyApp", "AutomataSimulator")
settings.clear()

print("Settings cleared.")