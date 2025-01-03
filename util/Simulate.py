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
    
def simPaste(key, pressed=True, returning=False):
    #pasting the second element in the clipboard
    

    settings = QSettings("MyApp", "AutomataSimulator")

    clicked_verse = settings.value("verse", None)  # Use None as default if not set

    # keyboard_controller.type(clicked_verse) #putting the verse in program search bar
    keyboard.send('ctrl+v')
    
    #typing enter key
    keyboard_controller.tap(Key.enter.value)

    #get the bar location from where the paste occurs
    settings.setValue('bar_location', mouse_controller.position)
    
    


def simSelectAll(pressed=True):
    """the select all function will be followed by the cut function for ease of use in the display prev verse"""
    settings = QSettings("MyApp", "AutomataSimulator")
    next_verse =settings.value('next_verse')
    
    keyboard.send('ctrl+a', 'crtl+c')
    #saving the current verse 
    # prev_verse = pyperclip.paste()
    time.sleep(0.5)
    # settings.setValue('prev_verse', prev_verse)
    # pyperclip.copy(next_verse)
    
    # #putting the next verse back at the top of the clipboard
    # Update clipboard with the next verse
    # clipboard = QApplication.clipboard()
    # clipboard.setText(next_verse)
    # print('here is the pre verse', prev_verse)
    
    print(f'next verse is {next_verse} and prev verse is {settings.value("prev_verse")}')
    
    
    


