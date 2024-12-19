"""
Together.py
Description: This script allows you to track mouse and keyboard events simultaneously.
Author: Worifung Achu
Date: 12.18.2024
"""

from pynput import keyboard, mouse

def create_new_automaton(edit=False):
    """
    Create a new automaton by tracking mouse and keyboard events
    
    Args:
        edit (bool): If True, stop tracking after the first event.
    
    Returns:
        list: List of tracked actions
    """
    # Store actions in a list
    actionList = []

    # Tracking state
    tracking = True

    # All acceptable hotkey combinations
    combination = [
        {keyboard.Key.ctrl_l, keyboard.KeyCode(char='v')},
        {keyboard.Key.ctrl_r, keyboard.KeyCode(char='v')},
        {keyboard.Key.ctrl_r, keyboard.KeyCode(char='V')},
        {keyboard.Key.ctrl_l, keyboard.KeyCode(char='V')}
    ]
    current = set()

    def on_copy():
        """Log copy/paste action"""
        nonlocal tracking  # Declare nonlocal to modify outer variable
        actionList.append({
            'action': 'paste',
            'button': 'ctrl+v',
            'location': 'clipboard'
        })
        print("Ctrl + V detected and logged!")
        if edit:
            tracking = False  # Stop tracking

    def on_mouse_click(x, y, button, pressed):
        """Log mouse click events"""
        nonlocal tracking  # Declare nonlocal to modify outer variable
        if pressed:
            actionList.append({
                'action': 'click',
                'location': (x, y),
                'button': str(button)
            })
            print(f"Mouse clicked at {x}, {y} with {button}")
        if edit:
            tracking = False  # Stop tracking

    def on_keyboard_press(key):
        nonlocal current, tracking  # Declare nonlocal to modify outer variables
        
        try:
            # Handle Escape key to stop tracking
            if key == keyboard.Key.esc:
                print("Stopping listeners...")
                tracking = False
                return False
            
            # Check if the pressed key is part of any combination
            if any([key in comb for comb in combination]):
                current.add(key)
                
                # Check if any complete combination is pressed
                if any(all(k in current for k in comb) for comb in combination):
                    on_copy()
        
        except AttributeError:
            pass

    def on_keyboard_release(key):
        nonlocal current  # Declare nonlocal to modify outer variables
        
        try:
            # Remove the key from current set if it was part of a combination
            if any([key in comb for comb in combination]):
                if key in current:
                    current.remove(key)
        
        except AttributeError:
            pass

    # Start listeners
    click_listener = mouse.Listener(on_click=on_mouse_click)
    keyboard_listener = keyboard.Listener(
        on_press=on_keyboard_press,
        on_release=on_keyboard_release
    )

    # Start and join listeners
    click_listener.start()
    keyboard_listener.start()

    try:
        while tracking:
            pass  # Run loop until tracking is set to False
    except Exception as e:
        print(f"An exception occurred: {e}")
    finally:
        # Stop listeners
        click_listener.stop()
        keyboard_listener.stop()

    return actionList

# Can be used for standalone testing
if __name__ == '__main__':
    actions = create_new_automaton(edit=True)
    print("Tracked Actions:", actions)
