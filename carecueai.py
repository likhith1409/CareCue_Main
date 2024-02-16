import csv
import yaml
import json

# Function to read symptoms data from CSV files
def read_data(file_path):
    data = {}
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) > 1:
                data[row[0]] = row[1:]
    return data

# Function to read chat modes and responses from YAML file
def read_chat_modes(file_path):
    with open(file_path, 'r') as yaml_file:
        chat_modes = yaml.safe_load(yaml_file)
    return chat_modes

# Function to read JSON responses
def read_responses(file_path):
    with open(file_path, 'r') as json_file:
        responses = json.load(json_file)
    return responses

# Main function containing the entire chatbot logic
def run_chatbot(name):
    # Loading symptom data
    symptom_description = read_data('Data/symptom_Description.csv')
    symptom_severity = read_data('Data/symptom_severity.csv')
    symptom_precaution = read_data('Data/symptom_precaution.csv')

    # Load chat modes and responses
    chat_modes = read_chat_modes('Data/greetings.yml')
    responses = read_responses('Data/carecuenormalchart.json')

    print("\n-----------------------------------HealthCare ChatBot-----------------------------------")

    # Initial mode is set to 'menu'
    current_mode = 'menu'
    print(f"Hello, {name.capitalize()}!\n")

    while True:
        # Display welcome message and prompt based on the current mode
        print(chat_modes['modes'][current_mode]['welcome_message'])

        if current_mode == 'menu':
            for option, description in chat_modes['modes'][current_mode]['options'].items():
                print(f"{option}) {description}")
            user_input = input(chat_modes['modes'][current_mode]['prompt']).strip().lower()
            if user_input == '1':
                current_mode = 'normal'
            elif user_input == '2':
                current_mode = 'symptom'
            elif user_input == '3':
                print("Option 3: Chart without AI")
                # Add your code here for chart generation without AI
            elif user_input == 'exit':
                break
            else:
                print("Invalid option. Please choose a valid option.")
        else:
            user_input = input(chat_modes['modes'][current_mode]['prompt']).strip().lower()
            if user_input == 'exit':
                current_mode = 'menu'
            elif current_mode == 'symptom':
                disease_input = user_input
                while True:
                    try:
                        num_days = int(input("Okay. From how many days? : "))
                        break
                    except ValueError:
                        print("Enter a valid input.")
                
                # Simulate symptom-based logic using rules
                if 'fever' in disease_input.lower() and num_days > 3:
                    print("You might have a flu. Please consult a doctor.")
                elif 'cough' in disease_input.lower():
                    print("It could be a common cold. Rest and drink plenty of fluids.")
                else:
                    print("I'm not sure. Please consult a doctor for a more accurate diagnosis.")
            elif current_mode == 'normal':
                if user_input in responses:
                    print(f"Chatbot: {responses[user_input]}")
                else:
                    print("Chatbot: I'm sorry, I can only respond to predefined queries in this mode.")

# Example usage
run_chatbot("User")
