import os
import json
import util.Message as msg
dir_path = os.path.expanduser("~/Quick move")  # This resolves to something like C:\Users\<username>\Quick move

file_path = os.path.join(dir_path, "savedVerses.json")
def getSavedVerses():
    global dir_path, file_path

    try:
        # Ensure the directory exists
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
            # print("Directory created")

        # Check if the JSON file exists
        if not os.path.isfile(file_path):
            with open(file_path, "w") as f:
                json.dump({}, f)  # Initialize with an empty JSON object
                # print("File created")

        # Check if the file is empty before loading
        with open(file_path, "r") as f:
            file_content = f.read().strip()
            if not file_content:
                # print("File is empty. Returning empty dictionary.")
                return {}
            return json.loads(file_content)

    except OSError as e:
        msg.warningBox(None, "Error", f"An error occurred while getting saved verse: {str(e)}")
        return {}  # Return an empty dictionary if an error occurs
def saveVerse(verse):
    global file_path
    try:
        with open(file_path, "w") as f:
            json.dump(verse, f)
    except OSError as e:
        msg.warningBox(None, "Error", f"An error occurred while trying to write to the saved verses file: {str(e)}")
        return


