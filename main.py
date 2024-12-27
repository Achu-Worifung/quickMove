import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from widgets.Welcome import Welcome  # Correctly import the Welcome class

class MainWindow(QMainWindow):
    def __init__(self):  # Corrected the method name to __init__
        super().__init__()
        self.setCentralWidget(Welcome())
        self.setGeometry(100, 100, 800, 800)
        self.setWindowTitle("Automata Simulator")
        # self.seti

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    
    window.show()
    
    try:
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Exiting due to: {e}")

if __name__ == "__main__":  # Corrected the condition to __name__ == "__main__"
    main()
