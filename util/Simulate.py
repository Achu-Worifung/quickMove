"""
This file is used to run the simulation of mouse and keyboard inputs.
"""
import sys
from PyQt5.QtWidgets import QApplication
from pynput import mouse, keyboard
from pynput.mouse import Button, Controller as MouseController
from pynput.keyboard import Key, Controller as KeyboardController
# import pyperclip
from PyQt5.QtCore import QSettings, QMimeData
import time
# import keyboard

# Create controllers
mouse_controller = MouseController()
keyboard_controller = KeyboardController()
mouse_button = Button
settings = QSettings("MyApp", "AutomataSimulator")
# Initialize QApplication first
app = QApplication(sys.argv)

# Now, initialize clipboard and settings
clipboard = app.clipboard()
def simClick(x, y, button = Button.left, pressed = True):

    
    """
    Simulate a mouse click at the specified coordinates.
    
    Args:
        x (int): X-coordinate for mouse position
        y (int): Y-coordinate for mouse position
        button (str or Button): Mouse button to click (left, right, or middle)
        pressed (bool): Button press state
    """
    mouse_controller.position = (x, y)  # moving the mouse to the desired location
    
    # Convert string representation to Button if necessary
    if isinstance(button, str):
        if button == 'Button.left':
            button = Button.left
        elif button == 'Button.right':
            button = Button.right
        elif button == 'Button.middle':
            button = Button.middle
        else:
            raise ValueError(f"Unknown button type: {button}")
    
    # Perform the click
    mouse_controller.click(button)
    mouse_controller.click(button)
    reset_keys()
    
def simPaste(keyvalue, pressed=True, returning=False):
    """
    Simulate pasting content from the clipboard.
    """
    QApplication.processEvents()
    settings = QSettings("MyApp", "AutomataSimulator")
    presenting_verse = settings.value("next_verse", None).strip()  # Use None as default if not set
    clipboard.setText(presenting_verse)
    

    try:
        # Press and release 'Ctrl+V' explicitly
        with keyboard_controller.pressed(keyboard.Key.ctrl):
            keyboard_controller.press('v')
            keyboard_controller.release('v')

        # Simulate pressing Enter
        keyboard_controller.tap(Key.enter.value)
        reset_keys()

        # Save the bar location
        if settings.value('bar_location') is None:
            settings.setValue('bar_location', mouse_controller.position)
    except Exception as e:
        print(f"Error during simPaste: {e}")
        reset_keys()

    
    


def simSelectAll(pressed=True):
    try:
        # print("Sending Ctrl+A to select all.")
        # Simulate Ctrl+A (Select All)
        with keyboard_controller.pressed(keyboard.Key.ctrl):
            keyboard_controller.press('a')
            keyboard_controller.release('a')

        # Simulate Ctrl+C (Copy)
        with keyboard_controller.pressed(keyboard.Key.ctrl):
            keyboard_controller.press('c')
            keyboard_controller.release('c')

        # Wait for clipboard to update
        QApplication.processEvents()  # Process any pending events
        time.sleep(0.1)  # Optionally give it a little more time
        
        # Get the text from the clipboard
        prev_verse = clipboard.text()
        # print(f"Copied text: {prev_verse}")

        # Save the text if there's anything copied
        if prev_verse:
            settings.setValue('prev_verse', prev_verse)
            # print(f"Saved previous verse: {prev_verse}")
        else:
            print("No text copied to clipboard.")
        
        reset_keys()

    except Exception as e:
        print(f"Error during simSelectAll: {e}")
        reset_keys()

    
    
def present_prev_verse():
     # Get the previous verse from QSettings
        next_verse = settings.value('prev_verse')
        next_verse = next_verse.strip()
        if not next_verse:
            print("Error: No previous verse found in settings.")
            return
        print('Previous verse present prev verse:', next_verse)
        settings.setValue('next_verse', next_verse)
        # QApplication.processEvents()  # Ensure clipboard changes are processed
        # # #switching to mime type text/plain
        # mine_data = QMimeData()
        # mine_data.setText(prev_verse)
        # clipboard.setMimeData(mine_data)
    
        # Get the search bar location from QSettings
        bar_location = settings.value("bar_location", None)
        if not bar_location or len(bar_location) != 2:
            print("Error: Invalid or missing bar location.")
            return
        # print(f"Bar location: {bar_location}")

        # Move the cursor to the bar location
        simClick(bar_location[0], bar_location[1])

        # Simulate select all, then paste
        simSelectAll()
        
        # clipboard.setText(prev_verse)
        # import time
        # time.sleep(0.1)  # Small delay to ensure the clipboard update is ready
        simPaste('key')

def reset_keys():
    """Release commonly used keys using the KeyboardController."""
    try:
        # Using Key enum for special keys and string literals for standard keys
        for key in [Key.ctrl, Key.enter, Key.shift, 'a', 'c', 'v']:
            keyboard_controller.release(key)
        # print("All keys reset.")
    except Exception as e:
        print(f"Error resetting keys: {e}")
