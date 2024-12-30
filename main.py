import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from widgets.Welcome import Welcome  # Correctly import the Welcome class
from PyQt5.QtGui import QIcon 

class MainWindow(QMainWindow):
    def __init__(self):  # Corrected the method name to __init__
        super().__init__()
        self.setCentralWidget(Welcome())
        self.setGeometry(100, 100, 800, 800)
        self.setWindowTitle("Automata Simulator")
        self.setWindowIcon(QIcon("icon.png")) #put icon in the same directory as the main.py file
        # self.seti

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    
    window.show()
    
    try:
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Exiting due to: {e}")

if __name__ == "__main__":  
    main()
