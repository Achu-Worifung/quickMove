"""
This module creates a new automaton by tracking mouse and keyboard events."""

from pynput import keyboard, mouse
import sys

def create_new_automaton(edit=False):
    # print("Creating new automaton...")
    """
    Create a new automaton by tracking mouse and keyboard events.

    Args:
        edit (bool): If True, stop tracking after the first event.

    Returns:
        list: List of tracked actions.
    """
    # Store actions in a list
    actionList = []

    # Tracking state
    tracking = True
    event_logged = False  # Flag to ensure at least one event is logged in edit mode

    # All acceptable hotkey combinations
    combination = [
        {keyboard.Key.ctrl_l, keyboard.KeyCode(char='v')},
        {keyboard.Key.ctrl_r, keyboard.KeyCode(char='v')},
        {keyboard.Key.ctrl_r, keyboard.KeyCode(char='V')},
        {keyboard.Key.ctrl_l, keyboard.KeyCode(char='V')}
    ]
    current = set()

    def on_pasete():
        """Log copy/paste action."""
        nonlocal tracking, event_logged
        actionList.append({
            'action': 'paste',
            'button': 'ctrl+v',
            'location': 'clipboard'
        })
        # print("Ctrl + V detected and logged!")
        sys.stdout.write("Ctrl + V detected and logged!\n", flush=True)
        flush()
        event_logged = True  # Mark that an event has been logged
        if edit:
            tracking = False  # Stop tracking

    def on_mouse_click(x, y, button, pressed):
        """Log mouse click events."""
        nonlocal tracking, event_logged
        if pressed:
            actionList.append({
                'action': 'click',
                'location': (x, y),
                'button': str(button)
            })
            print(f"Mouse clicked at {x}, {y} with {button}", flush=True)
            sys.stdout.write(f"Mouse clicked at {x}, {y} with {button}\n")
            sys.stdout.flush()
            event_logged = True  # Mark that an event has been logged
            if edit:
                tracking = False  # Stop tracking

    def on_keyboard_press(key):
        nonlocal current, tracking, event_logged
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
                    on_pasete()
        except AttributeError:
            pass

    def on_keyboard_release(key):
        nonlocal current
        try:
            # Remove the key from current set if it was part of a combination
            if any([key in comb for comb in combination]):
                if key in current:
                    current.remove(key)
        except AttributeError:
            pass
    
    def flush():
        sys.stdout.flush()

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
            if edit and event_logged:
                # Stop tracking if in edit mode and an event has been logged
                break
    except Exception as e:
        print(f"An exception occurred: {e}")
    finally:
        # Stop listeners
        click_listener.stop()
        keyboard_listener.stop()

    return actionList

import sys
print(sys.executable)

create_new_automaton(edit=False)
print("Automaton creation complete!", flush=True)
