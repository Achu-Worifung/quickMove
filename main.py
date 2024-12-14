from pynput.mouse import Button, Controller as MouseController
from pynput.keyboard import Key, Controller as KeyboardController
from pynput import mouse

# Initialize controllers
mouse_controller = MouseController()
keyboard_controller = KeyboardController()

# Store action list
actionList = []

def on_click(x, y, button, pressed):
    """
    Callback function to handle mouse click events
    
    Args:
    x (int): x-coordinate of the click
    y (int): y-coordinate of the click
    button (Button): Mouse button pressed
    pressed (bool): Whether the button is pressed or released
    """
    print(f'x: {x} y: {y} button: {button} pressed: {pressed}')
    location = (x, y)  # the location the mouse was clicked
    
    if pressed:
        actionList.append({
            'action': 'click', 
            'location': location, 
            'button': button
        })
    if Button.right == button:
        return False

def track_mouse_position():
    """
    Continuously print current mouse position
    """
    while True:
        try:
            # Get and print current mouse position
            current_pos = mouse_controller.position
            print(f'Current mouse position: {current_pos}')
            
            # Optional: Add a small delay to prevent overwhelming output
            import time
            time.sleep(1)
        
        except KeyboardInterrupt:
            # Allow user to stop tracking with Ctrl+C
            print("\nMouse tracking stopped.")
            break

def main():
    # Create a mouse listener for clicks
    click_listener = mouse.Listener(on_click=on_click)
    click_listener.start()
    
    try:
        # Track mouse position
        track_mouse_position()
    
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        # Stop the listener
        click_listener.stop()

if __name__ == '__main__':
    main()