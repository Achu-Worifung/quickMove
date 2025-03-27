import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QCheckBox
from PyQt5.QtCore import QPropertyAnimation, QSize
from PyQt5.QtGui import QIcon

class PulsingCheckboxWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Pulsing Checkbox Example')
        self.setGeometry(100, 100, 300, 200)

        layout = QVBoxLayout()

        # Create checkbox with custom styling
        self.checkbox = QCheckBox('Pulsing Checkbox')
        self.checkbox.setStyleSheet("""
            QCheckBox {
                spacing: 10px;
            }
            
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border: 2px solid #3498db;
                border-radius: 4px;
            }
            
            QCheckBox::indicator:unchecked {
                background-color: white;
            }
            
            QCheckBox::indicator:checked {
                background-color: #3498db;
                image: url(:/checkbox/checked.png);  /* Placeholder icon */
            }
            
            QCheckBox::indicator:checked:hover {
                background-color: #2980b9;
            }
        """)

        # Optional: Add pulsing animation
        self.animation = QPropertyAnimation(self.checkbox, b"geometry")
        self.checkbox.stateChanged.connect(self.start_pulse_animation)

        layout.addWidget(self.checkbox)
        self.setLayout(layout)

    def start_pulse_animation(self, state):
        if state == 2:  # Checked state
            self.animation.setDuration(1000)
            original_rect = self.checkbox.geometry()
            
            # Pulse animation
            self.animation.setStartValue(original_rect)
            self.animation.setEndValue(original_rect.adjusted(-10, -10, 10, 10))
            self.animation.start()

def main():
    app = QApplication(sys.argv)
    widget = PulsingCheckboxWidget()
    widget.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()