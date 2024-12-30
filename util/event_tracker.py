from pynput import keyboard, mouse
from PyQt5.QtCore import QObject, pyqtSignal
import util.Message as msg

class EventTracker(QObject):
    event_recorded = pyqtSignal(dict)  # Signal for new events
    tracking_finished = pyqtSignal(list)  # Signal for complete action list

    def __init__(self):
        super().__init__()
        self.actionList = []
        self.tracking = False
        self.current_keys = set()
        self.click_listener = None
        self.keyboard_listener = None

    def create_new_automaton(self, edit=False):
        """
        Start tracking mouse and keyboard events to create a new automaton.

        Args:
            edit (bool): If True, stop after logging the first event.

        Returns:
            list: List of tracked actions.
        """
        self.tracking = True
        self.current_keys.clear()

        def on_mouse_click(x, y, button, pressed):
            if not self.tracking:
                return False
            if pressed:
                event = {
                    'action': 'click',
                    'location': (x, y),
                    'button': str(button)
                }
                self.actionList.append(event)
                self.event_recorded.emit(event)
                print(f"Mouse clicked at {x}, {y} with {button}")
                if edit:
                    self.stop_tracking()

        def on_keyboard_press(key):
            if not self.tracking:
                return False
            try:
                if key == keyboard.Key.esc:
                    print("ESC pressed. Stopping tracking.")
                    self.stop_tracking()
                    return False
                self.current_keys.add(key)
                if self.is_paste_combination():
                    self.log_paste_event(edit)
            except Exception as e:
                 msg.warningBox(None, "Error", f"Keyboard press error: {str(e)}")

        def on_keyboard_release(key):
            try:
                self.current_keys.discard(key)
            except Exception as e:
                 msg.warningBox(None, "Error", f"Keyboard release error: {str(e)}")

        # Initialize listeners
        self.click_listener = mouse.Listener(on_click=on_mouse_click)
        self.keyboard_listener = keyboard.Listener(on_press=on_keyboard_press, on_release=on_keyboard_release)

        self.click_listener.start()
        self.keyboard_listener.start()

        print("Listeners started. Tracking events...")
        while self.tracking:
            pass

    def stop_tracking(self):
        """Stop tracking events and terminate listeners."""
        self.tracking = False
        if self.click_listener and self.click_listener.running:
            self.click_listener.stop()
        if self.keyboard_listener and self.keyboard_listener.running:
            self.keyboard_listener.stop()
        self.tracking_finished.emit(self.actionList)
        print("Tracking stopped and listeners terminated.")

    def is_paste_combination(self):
        """Check if the current keys match a paste (Ctrl+V) combination."""
        paste_combinations = [
            {keyboard.Key.ctrl_l, keyboard.KeyCode(char='v')},
            {keyboard.Key.ctrl_r, keyboard.KeyCode(char='v')}
        ]
        return any(all(k in self.current_keys for k in combo) for combo in paste_combinations)

    def log_paste_event(self, edit):
        """Log a paste event."""
        event = {
            'action': 'paste',
            'button': 'ctrl+v',
            'location': 'clipboard'
        }
        self.actionList.append(event)
        self.event_recorded.emit(event)
        print("Ctrl + V detected and logged!")
        if edit:
            self.stop_tracking()
