import sys
from PyQt5.QtWidgets import QApplication
from widgets.Welcome import Welcome  # Correctly import the Welcome class

def main():
    app = QApplication(sys.argv)
    welcome = Welcome()  # Create an instance of the Welcome class
    welcome.show()

    try:
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Exiting due to: {e}")

if __name__ == "__main__":
    main()
