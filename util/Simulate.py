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
from util.Message  import warningBox
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
def simClick(x, y, button = Button.left, pressed = False):

    
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
    
   
    mouse_controller.click(button, 2)

    reset_keys()
    
def simPaste(keyvalue, pressed=True, returning=False):
    """
    Simulate pasting content from the clipboard.
    """
    clipboard.setText(keyvalue)
    

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
            
            # time.sleep(0.05)
            settings.setValue('bar_location', mouse_controller.position)
    except Exception as e:
        # print(f"Error during simPaste: {e}")
        reset_keys()

    
    


def simSelectAll(pressed=True):
    try:
        # print("Sending Ctrl+A to select all.")
        # Simulate Ctrl+A (Select All)
        with keyboard_controller.pressed(keyboard.Key.ctrl):
            keyboard_controller.press('a')
            keyboard_controller.release('a')
        
        reset_keys()

    except Exception as e:
        reset_keys()

    
    
def present_prev_verse(name = None, data = None, prev_verse_text = None):
    
    if not prev_verse_text:
        warningBox(None, 'Error', 'No previous verse found to present.')
        return
    print('here is the data passed to present_prev_verse:', prev_verse_text)

    # Get the search bar location from QSettings
    bar_location = settings.value(name, None)

    for action in data['actions']:
        if action['action'] == 'click':
            x_coord = action['location'][0]
            y_coord = action['location'][1]
            button = action['button']
            simClick(x_coord, y_coord, button)
        elif action['action'] == 'paste':
            print('pasting')
            simPaste(prev_verse_text, True)
        elif action['action'] == 'select all':
            simSelectAll(True)

def reset_keys():
    """Release commonly used keys using the KeyboardController."""
    try:
        # Using Key enum for special keys and string literals for standard keys
        for key in [Key.ctrl, Key.enter, Key.shift, 'a', 'c', 'v']:
            keyboard_controller.release(key)
        # print("All keys reset.")
    except Exception as e:
        print(f"Error resetting keys: {e}")
