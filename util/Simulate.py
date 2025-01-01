"""
This file is used to run the simulation of mouse and keyboard inputs.
"""
from pynput import mouse, keyboard
from pynput.mouse import Button, Controller as MouseController
from pynput.keyboard import Key, Controller as KeyboardController

# Create controllers
mouse_controller = MouseController()
keyboard_controller = KeyboardController()
mouse_button = Button

def simClick(x, y, button, pressed):
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
    
def simPaste(key, pressed=True):
    keyboard_controller.press(Key.ctrl_r.value)
    keyboard_controller.press("v")
    keyboard_controller.release("v")
    keyboard_controller.release(Key.ctrl_r.value)
def simSelectAll(pressed=True):
    """the select all function will be followed by the cut function for ease of use in the display prev verse"""
    keyboard_controller.press(Key.ctrl_r.value)
    keyboard_controller.press("a")
    keyboard_controller.release("a")
    keyboard_controller.press('x')
    keyboard_controller.release('x')
    keyboard_controller.release(Key.ctrl_r.value)