import os
import json

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

    except OSError as e:
        print(f"An error occurred: {e}")
        return {}  # Return an empty dictionary if an error occurs
    except json.JSONDecodeError:
        print("Failed to decode JSON. Returning empty dictionary.")
        return {}

# Example usage
# data = get_data()
# print("Data:", data)

def write_data(data):
    global file_path
    y = json.dumps(data)
    with open(file_path, "w") as f:
        f.write(y)

def main():
    data = [{'emp_name': 'Shubham', 'email': 'ksingh.shubh@gmail.com', 'job_profile': 'intern'},
{'emp_name': 'Gaurav', 'email': 'gaurav.singh@gmail.com', 'job_profile': 'developer'},
{'emp_name': 'Nikhil', 'email': 'nikhil@geeksforgeeks.org', 'job_profile': 'Full Time'}]
    # write_data(data)
    data = get_data()
    print(data)

if __name__ == '__main__':
    main()
