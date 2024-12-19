
from pynput import keyboard, mouse
import threading

# Store actions in a list
actionList = []

# Define global listeners
click_listener = None
keyboard_listener = None
tracking = True  # Flag to track the state of the listeners

# All acceptable hotkey combinations
combination = [
    {keyboard.Key.ctrl_l, keyboard.KeyCode(char='v')},
    {keyboard.Key.ctrl_r, keyboard.KeyCode(char='v')},
    {keyboard.Key.ctrl_r, keyboard.KeyCode(char='V')},
    {keyboard.Key.ctrl_l, keyboard.KeyCode(char='V')}
]
current = set()

# Function triggered by the hotkey (Ctrl + V)
def on_copy():
    global actionList
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

def on_keyboard_press(key):
    global current, tracking
    
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
    global current
    
    try:
        # Remove the key from current set if it was part of a combination
        if any([key in comb for comb in combination]):
            if key in current:
                current.remove(key)
    
    except AttributeError:
        pass

# Start both listeners
def startListener():
    global click_listener, keyboard_listener, tracking
    
    # Reset tracking flag
    tracking = True
    
    # Start the mouse listener
    click_listener = mouse.Listener(on_click=on_mouse_click)
    click_listener.start()

    # Start the keyboard listener in a separate thread
    keyboard_listener = keyboard.Listener(
        on_press=on_keyboard_press,
        on_release=on_keyboard_release
    )
    keyboard_listener.start()

# Stop both listeners
def stopListener():
    global click_listener, keyboard_listener, tracking
    
    # Set tracking to False
    tracking = False
    
    # Stop listeners
    if click_listener:
        click_listener.stop()
    if keyboard_listener:
        keyboard_listener.stop()
    return actionList
def create_new_automaton():
    try:
        startListener()
        
        while tracking:
            keyboard_listener.wait()  # Wait for the listener to complete
    except Exception as e:
            print(f"An exception occurred: {e}")
    finally:
            stopListener()
            
# Main function
def main():
    print("Starting listeners. Press 'Ctrl+V' to log, 'Esc' to stop.")
    
    try:
        # Start listeners
        startListener()
        
        # Keep main thread running until tracking is False
        while tracking:
            keyboard_listener.wait()  # Wait for the listener to complete
    
    except KeyboardInterrupt:
        print("Program interrupted by user.")
    
    except Exception as e:
        print(f"An exception occurred: {e}")
    
    finally:
        # Ensure listeners are stopped
        stopListener()
        
        # Print logged actions
        print('All actions logged:', actionList)

if __name__ == '__main__':
    main()