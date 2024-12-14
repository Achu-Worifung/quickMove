
# myset = [{'brand': 'Ford', 'model': 'Mustang', 'year': 1964}, {'brand': 'Ford', 'model': 'Mustang', 'year': 1964}]
# # list and dictionary
# print(myset)

# ----------------------testing keyboard event-----------------------

from pynput import keyboard

def on_activate():
    print('Global hotkey (Ctrl+C) activated!')

def display_key(key):
    try:
        # Display character keys
        print(f"Key pressed: {key.char}")
    except AttributeError:
        # Handle special keys
        print(f'Special key {key} pressed')

# Listener to handle key presses and hotkey detection
def on_press(key):
    # Create a hotkey handler for Ctrl+C
    hotkey = keyboard.HotKey(
        keyboard.HotKey.parse('<ctrl>+c'),
        on_activate
    )
    
    try:
        # Pass the key press event to the hotkey handler
        hotkey.press(l.canonical(key))
        
        # Display the pressed key
        display_key(key)
    except Exception as e:
        print(f"Error in key press: {e}")

def on_release(key):
    try:
        # Create a hotkey handler for Ctrl+C
        hotkey = keyboard.HotKey(
            keyboard.HotKey.parse('<ctrl>+c'),
            on_activate
        )
        
        # Stop listener if Esc is pressed
        if key == keyboard.Key.esc:
            print("Exiting...")
            return False
        
        # Pass the key release event to the hotkey handler
        hotkey.release(l.canonical(key))
    except Exception as e:
        print(f"Error in key release: {e}")

# Start the listener
print("Keyboard listener started. Press Ctrl+C to trigger hotkey, ESC to exit.")
with keyboard.Listener(
        on_press=on_press,
        on_release=on_release) as l:
    l.join()