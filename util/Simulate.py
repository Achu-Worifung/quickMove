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
    settings = QSettings("MyApp", "AutomataSimulator")
    next_verse = settings.value('next_verse')

    try:
        print("Sending Ctrl+A to select all.")
        keyboard.press('ctrl')
        keyboard.press('a')
        keyboard.release('a')
        keyboard.release('ctrl')

        time.sleep(0.1)  # Ensure selection occurs

        print("Sending Ctrl+C to copy.")
        keyboard.press('ctrl')
        keyboard.press('c')
        keyboard.release('c')
        keyboard.release('ctrl')

       

        QApplication.processEvents()  # Ensure clipboard changes are processed
        time.sleep(0.2)  # Wait for clipboard to update
        clipboard = QApplication.clipboard()
        prev_verse = clipboard.text()
        print(f"Copied text: {prev_verse}")

        if prev_verse:
            settings.setValue('prev_verse', prev_verse)
            print(f"Saved previous verse: {prev_verse}")
        else:
            print("No text copied to clipboard.")

        clipboard.setText(next_verse)
        print(f"Set next verse: {next_verse}")

        reset_keys()
    except Exception as e:
        print(f"Error during simSelectAll: {e}")
        reset_keys()


    
    


def reset_keys():
    try:
        for key in ['ctrl', 'shift', 'alt', 'a', 'c', 'v']:
            keyboard.release(key)
        print("All keys reset.")
    except Exception as e:
        print(f"Error resetting keys: {e}")
