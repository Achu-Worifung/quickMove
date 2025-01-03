"""
This file is used to run the simulation of mouse and keyboard inputs.
"""
from PyQt5.QtWidgets import QApplication
from pynput import mouse, keyboard
from pynput.mouse import Button, Controller as MouseController
from pynput.keyboard import Key, Controller as KeyboardController
import pyperclip
from PyQt5.QtCore import QSettings
import time
import keyboard

# Create controllers
mouse_controller = MouseController()
keyboard_controller = KeyboardController()
mouse_button = Button

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
    
def simPaste(key, pressed=True, returning=False):
    """
    Simulate pasting content from the clipboard.
    """
    settings = QSettings("MyApp", "AutomataSimulator")
    clicked_verse = settings.value("verse", None)  # Use None as default if not set

    try:
        # Press and release 'Ctrl+V' explicitly
        keyboard.press('ctrl')
        keyboard.press('v')
        keyboard.release('v')
        keyboard.release('ctrl')

        # Simulate pressing Enter
        keyboard_controller.tap(Key.enter.value)

        # Save the bar location
        settings.setValue('bar_location', mouse_controller.position)
        reset_keys()
    except Exception as e:
        print(f"Error during simPaste: {e}")
        reset_keys()

    
    


def simSelectAll(pressed=True):
    """
    Simulate 'Select All' and 'Copy' actions.
    """
    settings = QSettings("MyApp", "AutomataSimulator")
    next_verse = settings.value('next_verse')

    try:
        print("Sending Ctrl+A to select all.")
        keyboard.press('ctrl')
        keyboard.press('a')
        keyboard.release('a')
        keyboard.release('ctrl')

        time.sleep(0.1)  # Small delay to ensure selection occurs

        print("Sending Ctrl+C to copy.")
        keyboard.press('ctrl')
        keyboard.press('c')
        keyboard.release('c')
        keyboard.release('ctrl')

        time.sleep(0.1)  # Small delay to ensure copy operation completes

        clipboard_content = pyperclip.paste()  # Get clipboard content
        print(f"Clipboard content after copy: {clipboard_content}")
        print(f"Next verse is {next_verse} and prev verse is {settings.value('prev_verse')}")

        reset_keys()
    except Exception as e:
        print(f"Error during simSelectAll: {e}")
        reset_keys()

    
    


def reset_keys():
    try:
        for key in ['ctrl', 'shift', 'alt']:
            keyboard.release(key)
        print("All keys reset.")
    except Exception as e:
        print(f"Error resetting keys: {e}")
