from flask import Flask, render_template, session, jsonify, request, redirect, url_for
import requests
import json
import sqlite3
import hashlib
import openai
import re
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
import csv
import warnings
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import cv2
import os
import joblib
import pyttsx3


warnings.filterwarnings("ignore", category=DeprecationWarning)

app = Flask(__name__)
app.secret_key = 'CareCueAi1409'

#####################################################CARECUE SYMPTOM CHAT################################################################
severityDictionary=dict()
description_list = dict()
precautionDictionary=dict()

# clf = pickle.load(open('dt.pkl', 'rb'))
# le = pickle.load(open('le.pkl', 'rb'))
training = pd.read_csv('Data/Training.csv')
y = training['prognosis']
le = preprocessing.LabelEncoder()
le= le.fit(y)

cols= training.columns
cols= cols[:-1]

x = training[cols]
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
clf1  = DecisionTreeClassifier()
clf = clf1.fit(x_train,y_train)

reduced_data = training.groupby(training['prognosis']).max()





def check_pattern(dis_list,inp):
    pred_list=[]
    inp=inp.replace(' ','_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list=[item for item in dis_list if regexp.search(item)]
    if(len(pred_list)>0):
        return 1,pred_list
    else:
        return 0,[]           
def sec_predict(symptoms_exp):
    df = pd.read_csv('Data/Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
      input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])

def print_disease(node):
    node = node[0]
    val  = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x:x.strip(),list(disease)))
         

def tree_to_code(tree, feature_names,msg):


    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis=",".join(feature_names).split(",")
    symptoms_present = []

    while True:

        print("\nEnter the symptom you are experiencing  \t\t",end="->")
        disease_input = msg
        conf,cnf_dis=check_pattern(chk_dis,disease_input)
        if conf==1:
            print("searches related to input: ")
            for num,it in enumerate(cnf_dis):
                print(num,")",it)
            if num!=0:
                print(f"Select the one you meant (0 - {num}):  ", end="")
                conf_inp = int(input(""))
            else:
                conf_inp=0

            disease_input=cnf_dis[conf_inp]
            break
            # print("Did you mean: ",cnf_dis,"?(yes/no) :",end="")
            # conf_inp = input("")
            # if(conf_inp=="yes"):
            #     break
        else:
            print("Enter valid symptom.")

    while True:
        try:
            num_days=int(input("Okay. From how many days ? : "))
            break
        except:
            print("Enter valid input.")
    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if  val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            # print( "You may have " +  present_disease )
            red_cols = reduced_data.columns
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            # dis_list=list(symptoms_present)
            # if len(dis_list)!=0:
            #     print("symptoms present  " + str(list(symptoms_present)))
            # print("symptoms given "  +  str(list(symptoms_given)) )
            print("Are you experiencing any ")
            symptoms_exp=[]
            for syms in list(symptoms_given):
                inp=""
                print(syms,"? : ",end='')
                while True:
                    inp=input("")
                    if(inp=="yes" or inp=="no"):
                        break
                    else:
                        print("provide proper answers i.e. (yes/no) : ",end="")
                if(inp=="yes"):
                    symptoms_exp.append(syms)

            second_prediction=sec_predict(symptoms_exp)
            # print(second_prediction)
            calc_condition(symptoms_exp,num_days)
            if(present_disease[0]==second_prediction[0]):
                print("You may have ", present_disease[0])
                print(description_list[present_disease[0]])

                # readn(f"You may have {present_disease[0]}")
                # readn(f"{description_list[present_disease[0]]}")

            else:
                print("You may have ", present_disease[0], "or ", second_prediction[0])
                print(description_list[present_disease[0]])
                print(description_list[second_prediction[0]])

            # print(description_list[present_disease[0]])
            precution_list=precautionDictionary[present_disease[0]]
            print("Take following measures : ")
            for  i,j in enumerate(precution_list):
                print(i+1,")",j)

            # confidence_level = (1.0*len(symptoms_present))/len(symptoms_given)
            # print("confidence level is " + str(confidence_level))

    recurse(0, 1)

def disease_search(feature_names,msg):

    return_string=''
    res=''   
    ret=''
    chk_dis=",".join(feature_names).split(",")
    symptoms_present = []
    conf,cnf_dis=check_pattern(chk_dis,msg)
    if conf==1:
        session['searches']=cnf_dis
        for num,it in enumerate(cnf_dis):
            return_string=return_string+str(num)+') '+it+'\n'  # Removed <br> and added newline (\n) here

        if num!="":
            res = (
                'Searches related to the input:\n' 
                + return_string 
                + '\nSelect the one you meant (0 -{}): '.format(num)
            )


            session['confirm'] =1
        else:
            res='No related data found.Please try again with another input'
            session['again'] = 1

    else:      
        res='Please enter valid symptoms'
        session['again'] = 1   
    return res


def calc_condition(exp, days, conversation_history):
    if ((len(exp) + 1) * days) / (len(exp) + 1) > 13:
        # Generate response using OpenAI API based on conversation history
        bot_response = generate_openai_response2(conversation_history)
        return bot_response
    else:
        # Generate response using OpenAI API based on conversation history
        bot_response = generate_openai_response2(conversation_history)
        return bot_response

def chatbot_response(tree,feature_names,msg):  
    c=""   
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    symptoms_present = []
    pre=""
    num_days = session.get('num_days', 0)
    inp= session.get('inp', 0)
    flag_rec = session.get('flag_rec', 0)
    disease_input = session.get('disease_input', 0)
    syms_exp=session.get('syms_exp', 0)
    second_prediction = ""
    def recurse(node, depth):
        c=""
        pre=""
        second_prediction = ""
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if  val <= threshold:
                return recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                return recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            # print( "You may have " +  present_disease )
            red_cols = reduced_data.columns
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            sym_list=[]
            for syms in list(symptoms_given):
                sym_list.append(syms)
            l=len(sym_list)
            print (f" symptoms are: {sym_list}") 
            print (f" length of list: {l}")    

            session['syms']=1
            syms = session.get('syms', 0)
            count=session.get('count', 0)
            
            
            if count < len(sym_list):
                if count >=1 and count< len(sym_list):
                    while True:
                        if msg=="yes" or msg=="no":
                            break
                        else:
                            return "Provide proper answer i.e. (yes/no) : "
                    if(msg=="yes"):
                        
                        syms_exp.append(sym_list[count-1])
                        session['syms_exp']=syms_exp
                            
                        print(f"Exp symptoms are: {syms_exp}")
                count=count+1
                print (f"count is: {count}")
                session['count']=count
                if count==len(sym_list):
                    session['syms']=0
                return "Are you experiencing any {}".format(sym_list[count-1] +"?"+"(yes/no)")   
            else:
                second_prediction=sec_predict(syms_exp)
                c=calc_condition(syms_exp,num_days,conversation_history)
                return c
                    
                
        
    return recurse(0,1)

        
        

################################################# IMAGE CLASSIFICATION ###########################################################
# Load the trained logistic regression model
model = joblib.load('Data/logistic_regression_model.pkl')


def classify_image(image):
    # Preprocess the image (resize, flatten)
    resized_image = cv2.resize(image, (64, 64))
    flattened_image = resized_image.flatten()

    # Perform classification using the trained model
    predicted_class = model.predict([flattened_image])[0]
    return predicted_class

def get_medicine_details(medicine_name):
    # Lookup medicine details in the JSON dataset
    for medicine_entry in medicine_data:
        if isinstance(medicine_entry['name'], list):
            for medicine in medicine_entry['name']:
                if medicine == medicine_name:
                    return {'uses': medicine_entry['uses'], 'side_effects': medicine_entry['side_effects']}
        else:
            if medicine_entry['name'] == medicine_name:
                return {'uses': medicine_entry['uses'], 'side_effects': medicine_entry['side_effects']}
    return None

@app.route('/upload_image', methods=['POST'])
def upload_image():
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    
    try:
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({'error': str(e)})

    
    predicted_medicine = classify_image(image)

    
    medicine_details = get_medicine_details(predicted_medicine)

    if medicine_details:
        medicine_details['name'] = predicted_medicine 
        return jsonify(medicine_details)
    else:
        return jsonify({'error': 'Medicine details not found'})

#########################################################################################################################
# Replace 'YOUR_API_KEY' with your actual News API key
NEWS_API_KEY = 'Your_Api_key'
openai.api_key = 'Your_Api'


NEWS_API_BASE_URL = 'https://newsapi.org/v2/top-headlines'


#######################################################################################################################
# Assuming your JSON data is loaded from a file
with open('Data/medidetails.json') as f:
    medicine_data = json.load(f)

def load_responses():
    responses = {}
    with open('Data/Conversation.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            responses[row['question']] = row['answer']
    return responses


# Example news data
example_article = {
    'headline': 'Achieve These Amazing Health Benefits By Consuming Oranges - NDTV',
    'source': 'NDTV News',
    'description': "Oranges are a great source of vitamin C and antioxidants, promoting overall health.",
    'link': 'https://www.ndtv.com/health/achieve-these-amazing-health-benefits-by-consuming-oranges-4959192'
}

# Database Initialization
conn = sqlite3.connect('user_database.db')
cursor = conn.cursor()

# Create the users table if it doesn't exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT UNIQUE,
        password TEXT
    )
''')
conn.commit()
#################################################### SIGNUP AND LOGIN ###############################################
# Signup route
@app.route('/signup', methods=['POST'])
def signup():
    try:
        data = request.get_json()

        # Extract user details from the request
        name = data.get('name')
        email = data.get('email')
        password = data.get('password')

        # Hash the password before storing it in the database
        hashed_password = hashlib.sha256(password.encode()).hexdigest()

        # Insert user details into the database
        conn = sqlite3.connect('user_database.db')
        cursor = conn.cursor()
        cursor.execute('INSERT INTO users (name, email, password) VALUES (?, ?, ?)', (name, email, hashed_password))
        conn.commit()
        conn.close()

        return jsonify({'success': 'User registered successfully'})
    except Exception as e:
             return jsonify({'error': str(e)})


# Login route
@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()

        # Extract user credentials from the request
        email = data.get('email')
        password = data.get('password')

        # Hash the password before comparing with the database
        hashed_password = hashlib.sha256(password.encode()).hexdigest()

        # Check if the user exists in the database
        conn = sqlite3.connect('user_database.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE email=? AND password=?', (email, hashed_password))
        user = cursor.fetchone()
        conn.close()

        if user:
            # Set a session variable to indicate the user is logged in
            session['user'] = {'id': user[0], 'name': user[1], 'email': user[2]}
            return jsonify({'success': 'Login successful', 'name': user[1]})
        else:
            return jsonify({'error': 'Invalid email or password'})
    except Exception as e:
        return jsonify({'error': str(e)})



# Update the '/logout' route
@app.route('/logout')
def logout():
     # Check if the user is logged in
    if 'user' in session:
        # Clear the session to log out the user
        session.clear()
        return jsonify({'success': 'Logout successful'})
    else:
        return jsonify({'error': 'User not logged in'})

#######################################################################################################
    
@app.route('/normal_chart')
def normal_chart():
    session['mode'] = 'normal'
    return 'You are in Normal Chart mode.'

@app.route('/symptom_chart')
def symptom_chart():
    session['mode'] = 'symptom'
    return 'You are in Symptom Chart mode.'

@app.route('/logo_chart')
def logo_chart():
    session['mode'] = 'logo'
    return 'You are in OpenAi Chart mode.'

############################################### MEDICINE DETAILS WITH NAME ########################################################
@app.route('/get_details')
def get_details():
    medicine_name = request.args.get('name', '')

    for medicine in medicine_data:
        if isinstance(medicine['name'], list):
            if medicine_name in medicine['name']:
                return jsonify({'uses': medicine['uses'], 'side_effects': medicine['side_effects']})
        elif isinstance(medicine['name'], str):
            if medicine_name == medicine['name']:
                return jsonify({'uses': medicine['uses'], 'side_effects': medicine['side_effects']})

    return jsonify({'error': 'Medicine not found'})

################################################ HEALTH NEWS  ####################################################
def get_health_news():
    try:
        # Set the parameters for the News API request
        params = {
            'country': 'in',  # India
            'category': 'health',
            'apiKey': NEWS_API_KEY,
        }

        # Make the request to the News API
        response = requests.get(NEWS_API_BASE_URL, params=params)

        # Check the status code
        if response.status_code != 200:
            # If the status code is not 200, log the error and return the example article
            print(f"Error fetching news from API: Status code {response.status_code}")
            return [example_article]

        # Parse the response JSON
        news_data = response.json()

        # Extract relevant information from the response
        health_news = []
        for article in news_data.get('articles', []):
            headline = article.get('title', '')
            source = article.get('source', {}).get('name', '')
            link = article.get('url', '')

            # Extract the description from the API response
            description = article.get('description', '')

            health_news.append({'headline': headline, 'source': source, 'link': link, 'description': description})

        if not health_news:
            # If the API response is empty, use the example article
            health_news.append(example_article)

        return health_news

    except requests.RequestException as e:
        # Log the error or handle it as needed
        print(f"Error fetching news from API: {e}")

        # Fallback to the example article
        return [example_article]

# Route to fetch health news
@app.route('/get_health_news')
def fetch_health_news():
    health_news = get_health_news()
    return jsonify(health_news)

############################################## CHATBOT SECTION #######################################################

@app.route('/')
def index():
    session['iteration_conversation'] = {'user_responses': [], 'bot_responses': []}
    session['num_responses'] = 0
    session['confirm'] = 0
    session['searches'] = ''
    session['disease'] = 0
    session['num_days'] = 0
    session['hi'] = 0
    session['flag_rec'] = 0
    session['disease_input'] = ""
    session['syms_exp']=[]
    session['syms']=0
    session['count']=0
    session['again'] = 0
    session['conversation_history'] = {'user_responses': [], 'bot_responses': []}

    health_news = get_health_news()
    current_index = session.get('news_index', 0)

    chat_history = session.get('chat_history', [])

    user_info = session.get('user')
    user_name = user_info.get('name') if user_info else None

    return render_template('index.html', health_news=health_news, current_index=current_index, chat_history=chat_history, user=user_name)


@app.route('/set_mode', methods=['POST'])
def set_mode():
    mode = request.json.get('mode')
    session['mode'] = mode
    return '', 204

def find_matching_response(user_input, responses):
    matched_response = None
    user_words = user_input.lower().split()
    for question, answer in responses.items():
        question_words = question.lower().split()
        matching_words = [word for word in user_words if word in question_words]
        if len(matching_words) >= 2:
            matched_response = answer
            break
    return matched_response

def generate_voice_response(text):
    # Function to generate voice response using pyttsx3
    engine = pyttsx3.init("sapi5")
    voices = engine.getProperty('voices')
    
    # Find the Zira voice
    for voice in voices:
        if "Zira" in voice.name:
            engine.setProperty('voice', voice.id)
            break
            
    engine.setProperty('rate', 150)  # You can adjust the speech rate as needed
    engine.say(text)
    engine.runAndWait()

@app.route('/process_voice_input', methods=['POST'])
def process_voice_input():
    user_message = request.json['user_message']
    
    openai_available = True  # Set this to False if OpenAI API is not available
    
    # Load responses each time the function is called
    responses = load_responses()
    
    if openai_available:
        # Call the OpenAI API to generate a bot response
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0125",
            messages=[{"role": "user", "content": user_message}]
        )
        
        bot_response = response['choices'][0]['message']['content']
    else:
        # Check if any two words in the user's input match with any part of the question
        matched_response = find_matching_response(user_message, responses)
        
        if matched_response:
            bot_response = matched_response
        else:
            bot_response = "I'm sorry, I couldn't understand. Please try again."
    
    # Generate voice response from text
    generate_voice_response(bot_response)
    
    return jsonify({'user_message': user_message, 'bot_response': bot_response, 'voice_response': 'bot_response.mp3'})



def generate_openai_response(user_input):
    # Define the prompt for the conversation
    prompt = f"In OpenAi Chat mode. User: {user_input}\nAI:"

    # Call OpenAI's API to generate a response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": "You are in OpenAi Chat mode."},
            {"role": "user", "content": user_input}
        ],
        max_tokens=150
    )

    # Extract and return the generated response
    return response.choices[0].message['content']


def generate_openai_response2(conversation_history):
    """
    Generate a response from OpenAI based on conversation history.

    Args:
    - conversation_history (dict): A dictionary containing the conversation history.

    Returns:
    - str: The response generated by OpenAI.
    """
    # Extract user responses and bot responses from conversation history
    user_responses = conversation_history.get('user_responses', [])
    bot_responses = conversation_history.get('bot_responses', [])

    # Combine user and bot responses
    combined_responses = [f"User: {user}\nBot: {bot}" for user, bot in zip(user_responses, bot_responses)]

    # Manual prompt
    manual_prompt = "check this details and give me report of health condition and should be short and with 2 points of precautions in it"

    # Concatenate combined responses and manual prompt
    prompt = "\n".join(combined_responses) + "\n" + manual_prompt

    # Request response from OpenAI using the chat completions endpoint
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0125",  # Choose appropriate chat model
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
    )

    return response.choices[0].message["content"].strip()


@app.route('/send_message', methods=['POST'])
def send_message():
    global conversation_history
    user_input = request.form['userInput']
    mode = session.get('mode', 'normal')  # Default mode is 'normal'

    def generate_response(user_input, mode):
        if mode == 'normal':
            responses = load_responses()
            # Check for matching responses
            matching_responses = [response for question, response in responses.items() if sum(word in user_input.lower().split() for word in question.lower().split()) >= 2]
            if matching_responses:
                return matching_responses[0]  # Return the first matching response
            else:
                return "Sorry, I couldn't find a matching response."

        

        elif mode == 'symptom':
            if user_input is None:
                return "Please provide a valid input.\n"

            session['num_responses'] = session.get('num_responses', 0) + 1
            num_responses = session.get('num_responses', 0)
            num_days = session.get('num_days', 0)
            confirm = session.get('confirm', 0)
            hi = session.get('hi', 0)
            disease = session.get('disease', 0)
            again = session.get('again', 0)
            flag_rec = session.get('flag_rec', 0)

            # Store user input in conversation history
            conversation_history.setdefault('user_responses', []).append(user_input)

            if num_responses == 1:
                session['hi'] = 1
                conversation_history.setdefault('bot_responses', []).append("Hello {}!\n".format(user_input))
                return "Hello {}!\n".format(user_input) + "\nEnter the symptom you are experiencing:\n"

            elif confirm == 1:
                searches = session.get('searches', 0)
                disease_in = searches[int(user_input)]
                session['disease_input'] = disease_in
                session['confirm'] = 0
                session['disease'] = 1
                # Store bot response in conversation history
                conversation_history.setdefault('bot_responses', []).append("So you are experiencing {}\n".format(disease_in))
                return "You mentioned experiencing {}.\nGot it! How many days have you been experiencing {} symptoms?\n".format(disease_in, disease_in)

            elif disease == 1:
                try:
                    n_days = int(user_input)
                    session['num_days'] = n_days
                    session['disease'] = 0
                    bot_response, new_session = calc_condition(cols, user_input, conversation_history)
                    # Store bot response in conversation history
                    conversation_history.setdefault('bot_responses', []).append(bot_response)
                    print(f"bot_response: {bot_response}")
                    if new_session:
                        return bot_response + "\nEnter the symptom you are experiencing:\n"
                    else:
                        return bot_response
                except:
                    conversation_history.setdefault('bot_responses', []).append("Enter a valid number of days.\n")
                    return "Enter a valid number of days.\n"

            elif hi == 1 or again == 1:
                session['hi'] = 0
                bot_response = disease_search(cols, user_input)
                # Store bot response in conversation history
                conversation_history.setdefault('bot_responses', []).append(bot_response)
                return bot_response

            bot_response = chatbot_response(clf, cols, user_input)
            # Store bot response in conversation history
            conversation_history.setdefault('bot_responses', []).append(bot_response)
            print(f"bot_response: {bot_response}")
            return bot_response

            
        elif mode == 'logo':
            bot_response = generate_openai_response(user_input)  # Generate response using OpenAI

        return bot_response

    bot_response = generate_response(user_input, mode)

    return jsonify({'user': user_input, 'bot': bot_response})

conversation_history = {}


if __name__ == '__main__':
    app.run(debug=True)