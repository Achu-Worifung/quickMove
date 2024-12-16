from pynput import keyboard, mouse

# Store actions in a list
actionList = []

# Define global listeners
click_listener = None
keyboard_listener = None
tracking = True # Flag to track the state of the listeners

# Function triggered by the hotkey (Ctrl + V)
def on_copy():
    actionList.append({
        'action': 'paste',
        'button': 'ctrl+v',
        'location': 'clipboard'
    })
    print("Ctrl + V detected and logged!")

# Mouse click handler
def on_mouse_click(x, y, button, pressed):
    if pressed:
        actionList.append({
            'action': 'click',
            'location': (x, y),
            'button': str(button)
        })
        print(f"Mouse clicked at {x}, {y} with {button}")

# Keyboard handlers for Ctrl + V
current_keys = set()  # Track currently pressed keys


def on_keyboard_press(key):
    global actionList
    try:
        if key == keyboard.Key.esc:
            print("Stopping listeners...")
            stopListener()
            print('all actions logged:', actionList)
            return False  # Stop the listener

        current_keys.add(key)  # Track key press
        if keyboard.Key.ctrl_l in current_keys and 'v' in {k.char for k in current_keys if hasattr(k, 'char')}:
            on_copy()  # Trigger copy action
    except AttributeError:
        pass  # Handle special keys (e.g., Ctrl, Alt, etc.)


def on_keyboard_release(key):
    # Remove key from tracking set
    if key in current_keys:
        current_keys.remove(key)

# Start both listeners
def startListener():
    global click_listener, keyboard_listener

    # Start the mouse listener
    click_listener = mouse.Listener(on_click=on_mouse_click)
    click_listener.start()

    # Start the keyboard listener
    keyboard_listener = keyboard.Listener(on_press=on_keyboard_press, on_release=on_keyboard_release)
    keyboard_listener.start()

# Stop both listeners
def stopListener():
    global click_listener, keyboard_listener
    if click_listener is not None:
        click_listener.stop()
    if keyboard_listener is not None:
        keyboard_listener.stop()
def crete_new_automaton():
    startListener()
    try:
        while tracking:
            pass
    except Exception as e:
        print("an Exception occured ", e)
    stopListener()


# Main function
def main():
    print("Starting listeners. Press 'Esc' to stop.")
    startListener()

    # Run in a loop to keep the main thread alive (non-blocking way)
    try:
        while tracking:
            pass  # Keep running
    except Exception as e:
        print("an Exception occured ", e)
        stopListener()


if __name__ == '__main__':
    main()
