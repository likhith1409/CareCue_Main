from flask import Flask, render_template, session, jsonify, request, redirect, url_for
import requests
import json
import sqlite3
import hashlib

app = Flask(__name__)
app.secret_key = 'carecue2023'

# Replace 'YOUR_API_KEY' with your actual News API key
# NEWS_API_KEY = 'b5034f4679a54b5bb1f7a2434d802186'
NEWS_API_KEY = 'Your_API'

NEWS_API_BASE_URL = 'https://newsapi.org/v2/top-headlines'

# Assuming your JSON data is loaded from a file
with open('Data/medidetails.json') as f:
    medicine_data = json.load(f)

# Function to load conversation data from the JSON file
def load_responses():
    with open('Data/carecuenormalchart.json', 'r') as file:
        return json.load(file)

# Function to save conversation data to the JSON file
def save_responses(responses):
    with open('Data/carecuenormalchart.json', 'w') as file:
        json.dump(responses, file)

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
    # Clear the session to log out the user
    session.clear()
    return jsonify({'success': 'Logout successful'})


@app.route('/normal_chart')
def normal_chart():
    # Add your logic for normal chart here
    return 'You are in Normal Chart mode.'

@app.route('/symptom_chart')
def symptom_chart():
    # Add your logic for symptom chart here
    return 'You are in Symptom Chart mode.'

@app.route('/logo_chart')
def logo_chart():
    # Add your logic for logo chart here
    return 'You are in Logo Chart mode.'

@app.route('/get_details')
def get_details():
    medicine_name = request.args.get('name', '').lower()

    for medicine in medicine_data:
        if medicine['name'].lower() == medicine_name:
            return jsonify({'uses': medicine['uses'], 'side_effects': medicine['side_effects']})

    return jsonify({'error': 'Medicine not found'})

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

@app.route('/')
def index():
    health_news = get_health_news()
    current_index = session.get('news_index', 0)

    chat_history = session.get('chat_history', [])

    user_info = session.get('user')
    user_name = user_info.get('name') if user_info else None

    return render_template('index.html', health_news=health_news, current_index=current_index, chat_history=chat_history, user=user_name)


@app.route('/send_message', methods=['POST'])
def send_message():
    user_input = request.form['userInput']

    # Load responses data
    responses = load_responses()

    # Get bot response based on user input
    bot_response = responses.get(user_input, "I'm a simple bot, and I don't understand that yet.")

    # Add user input to the conversation
    responses["user"] = user_input
    # Add bot response to the conversation
    responses["bot"] = bot_response

    # Save updated responses back to the JSON file
    save_responses(responses)

    return jsonify({'user': user_input, 'bot': bot_response})

if __name__ == '__main__':
    app.run(debug=True)
