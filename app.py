import sys
import os
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QVBoxLayout, QRadioButton,QMessageBox
import walk  # Placeholder for automata.json data retrieval logic
import Together #mouse and keyboard listeners

class MainWindow(QDialog):
    def __init__(self):
        super(MainWindow, self).__init__()

        # Load the UI file
        ui_path = "./ui/intro.ui"
        assert os.path.exists(ui_path), f"UI file not found: {ui_path}"
        loadUi(ui_path, self)

        # Add a layout to `scrollAreaWidgetContents` if it doesn't already have one
        if not self.scrollAreaWidgetContents.layout():
            self.scroll_layout = QVBoxLayout(self.scrollAreaWidgetContents)
        else:
            self.scroll_layout = self.scrollAreaWidgetContents.layout()

        # Load data into the UI
        self.loaddata()

    def loaddata(self):
        # Mock implementation of walk.get_data
        # Replace this with your actual `walk.get_data()` function
        data = walk.get_data()
        

        print(f"Data: {data}, Count: {len(data)}")

        if not data:
            # If no automata, display a message
            no_data_label = QtWidgets.QLabel("No automata available.")
            self.scroll_layout.addWidget(no_data_label)
            
            #creating create button
            button = QtWidgets.QPushButton("Create", self)
            self.scroll_layout.addWidget(button)
            
            #adding action listener to the button
            button.clicked.connect(self.create_new_automaton)
        else:
            # Add a radio button for each automaton
            for auto in data:
                radio_button = QRadioButton(auto.get("name"))
                self.scroll_layout.addWidget(radio_button)

                # Connect the radio button to a slot
                radio_button.toggled.connect(
                    lambda checked, name=auto.get("name"): self.radio_selected(checked, name)
                )

    def radio_selected(self, checked, name):
        if checked:  # Respond only when the button is selected
            print(f"Selected automaton: {name}")
    def create_new_automaton(self):
        # Together.crete_new_automaton()
        msg = QMessageBox()
        msg.setWindowTitle("Tracking")
        msg.setText("Tracking started \n Your mouse and keyboard events are being tracked Press 'Esc' to stop tracking")
        msg.setIcon(QMessageBox.Information)
        msg.exec_()
       

# Main application
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Initialize and display the main window
    mainwindow = MainWindow()
    mainwindow.show()

    # Run the application
    try:
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Exiting due to: {e}")
