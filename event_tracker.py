from pynput import keyboard, mouse

def create_new_automaton():
    """
    Create a new automaton by tracking mouse and keyboard events
    
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
        actionList.append({
            'action': 'paste',
            'button': 'ctrl+v',
            'location': 'clipboard'
        })
        print("Ctrl + V detected and logged!")

    def on_mouse_click(x, y, button, pressed):
        """Log mouse click events"""
        if pressed:
            actionList.append({
                'action': 'click',
                'location': (x, y),
                'button': str(button)
            })
            print(f"Mouse clicked at {x}, {y} with {button}")

    def on_keyboard_press(key):
        nonlocal current, tracking
        
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
        nonlocal current
        
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
            keyboard_listener.wait()  # Wait for the listener to complete
    except Exception as e:
        print(f"An exception occurred: {e}")
    finally:
        # Stop listeners
        click_listener.stop()
        keyboard_listener.stop()

    return actionList

# Can be used for standalone testing
if __name__ == '__main__':
    actions = create_new_automaton()
    print("Tracked Actions:", actions)