"""
walk.py
Author: worifung achu
Date: 12.18.2024
This module provides a function to retrieve and write  data from a JSON file in a user-accessible directory."""

import os
import json
import util.Message as msg




# Use a user-accessible directory
dir_path = os.path.expanduser("~/Quick move")  # This resolves to something like C:\Users\<username>\Quick move
file_path = os.path.join(dir_path, "automata.json")  # Construct the full path for the JSON file

def get_data():
    global dir_path, file_path
    try:
        # Ensure the directory exists
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
            print("Directory created")

        # Check if the JSON file exists
        if not os.path.isfile(file_path):
            with open(file_path, "w") as f:
                json.dump({}, f)  # Initialize with an empty JSON object
                print("File created")

        # Check if the file is empty before loading
        with open(file_path, "r") as f:
            file_content = f.read().strip()
            if not file_content:
                print("File is empty. Returning empty dictionary.")
                return {}
            return json.loads(file_content)
    
    # Catch OSError
    except OSError as e:
        msg.warningBox(None, "Error", f"An error occurred while trying to access the automata.json file: {str(e)}")
        return {}  # Return an empty dictionary if an error occurs
    
    # Catch JSONDecodeError
    except json.JSONDecodeError as e:
        msg.warningBox(None, "Error", f"An error occurred while decoding the JSON: {str(e)}")
        return {}

# Example usage
# data = get_data()
# print("Data:", data)

def write_data(data):
    global file_path
    try:
      with open(file_path, "w") as f:
        json.dump(data, f)
    except OSError as e:
        msg.warningBox(None, "Error", f"An error occurred while trying to write to the automata.json file: {str(e)}")
        return 

def main():
    data = {"Automata": [
    {
      "name": "Automata",
      "actions": [
        { "action": "click", "button": "Button.left", "location": [740, 757] },
        { "action": "click", "button": "Button.left", "location": [1283, 987] },
        { "action": "paste", "button": "ctrl+v", "location": "clipboard" }
      ]
    },
    {
      "name": "Automata1",
      "actions": [
        { "action": "click", "button": "Button.left", "location": [740, 757] },
        { "action": "click", "button": "Button.left", "location": [1283, 987] },
        { "action": "paste", "button": "ctrl+v", "location": "clipboard" }
      ]
    },
    {
      "name": "Automata2",
      "actions": [
        { "action": "click", "button": "Button.left", "location": [740, 757] },
        { "action": "click", "button": "Button.left", "location": [1283, 987] },
        { "action": "paste", "button": "ctrl+v", "location": "clipboard" }
      ]
    }
  ]}
    write_data(data)
    # data = get_data()
    # print(data)

if __name__ == '__main__':
    main()
