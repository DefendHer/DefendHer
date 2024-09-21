import cv2
import pandas as pd
from flask import Flask, Response, render_template, request, jsonify, redirect, url_for, flash, session
from dotenv import load_dotenv
import os
import google.generativeai as genai
import logging
from pymongo import MongoClient
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
from bson.objectid import ObjectId
import jwt
import datetime
import joblib
from flask_mail import Mail, Message
import secrets
import requests
import folium
import time
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from flask import Flask, request, jsonify, render_template
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import folium
import requests
import traceback
from flask import Flask, request, jsonify
import vonage
from flask import Flask, request, jsonify, url_for
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash
from bson import ObjectId
from tensorflow.keras.applications.vgg16 import VGG16 # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array# type: ignore
import gdown
import tensorflow as tf  
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input# type: ignore
from tensorflow.keras.preprocessing.image import img_to_array, load_img# type: ignore
from keras.models import load_model # type: ignore
from keras.preprocessing.image import img_to_array # type: ignore
from deepface import DeepFace

import threading
# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your_secret_key')  # Use a secure key

# MongoDB Configuration
app.config["MONGO_URI"] = "mongodb+srv://DefendHer:defendher123@defendher.mxs7f.mongodb.net/DefendHer"
mongo = PyMongo(app)



# Flask-Mail Configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'  # Replace with your SMTP server
app.config['MAIL_PORT'] = 587  # Typically 587 for TLS or 465 for SSL
app.config['MAIL_USE_TLS'] = True  # Set to False if using SSL
app.config['MAIL_USERNAME'] = 'defendher52@gmail.com'  # Replace with your email username
app.config['MAIL_PASSWORD'] = 'fxrq nwho uoct zobl'  # Replace with your email password
app.config['MAIL_DEFAULT_SENDER'] = ('DefendHer', 'defendher52@gmail.com')  # Replace with your name and email
mail = Mail(app)
s = URLSafeTimedSerializer(app.config['SECRET_KEY'])


# Configure the Generative AI modeloovbfk
api_key = os.getenv("AIzaSyA_mYeQS0jqOfebqCSVPQ_gy0c5i1d1ZHs")  # Use environment variable for security
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])



def get_gemini_response(question):
    try:
        response = chat.send_message(question, stream=True)
        return response
    except genai.types.generation_types.BrokenResponseError:
        logger.error("Encountered BrokenResponseError. Retrying...")
        chat.rewind()
        response = chat.send_message(question, stream=True)
        return response

# Load your text prediction model
pipe_lr = joblib.load(open("model/text_safe_unsafe.pkl", "rb"))



# Emoji dictionary for safe and unsafe labels
safety_emoji_dict = {
    "safe": "✅", "unsafe": "⚠️"
}

def predict_safety(docx):
    results = pipe_lr.predict([docx])
    return results[0]


# Initialize geocoder
geolocator = Nominatim(user_agent="safe_route_predictor")

# Your OpenRouteService API key
ORS_API_KEY = '5b3ce3597851110001cf6248295620b8636d462c8c09581404de5dad'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        raw_text = request.form['text']
        prediction = predict_safety(raw_text)
        return render_template('result.html', raw_text=raw_text, prediction=prediction,
                               emoji=safety_emoji_dict.get(prediction, ""))
    


# @app.route('/harass')
# def harass():
#     return render_template('harass.html')


@app.route('/texts')
def texts():
    return render_template('text.html')

@app.route('/first')
def first():
    return render_template('first.html')

@app.route('/second')
def second():
    return render_template('second.html')

@app.route('/third')
def third():
    return render_template('third.html')

@app.route('/fourth')
def fourth():
    return render_template('fourth.html')

@app.route('/video')
def video():
    return render_template('video.html')

@app.route('/alert')
def alert():
    return render_template('alert.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    guardian_number = data.get('guardianNumber')

    if mongo.db.users.find_one({"email": email}):
        return jsonify({"error": "User already exists"}), 400

    hashed_password = generate_password_hash(password)
    user_id = mongo.db.users.insert_one({
        "name": name,
        "email": email,
        "password": hashed_password,
        "guardianNumber": guardian_number,
        "is_verified": False,  # User is not verified yet
    }).inserted_id

    token = s.dumps(email, salt='email-confirm')
    verification_link = url_for('verify_email', token=token, _external=True)
    send_verification_email(email, verification_link)

    return jsonify({"message": "A verification email has been sent to your email address. Please verify to complete the signup."}), 201

def send_verification_email(email, verification_link):
    msg = Message('Email Verification', sender='your_email@example.com', recipients=[email])
    msg.body = f'Please click the link to verify your email: {verification_link}'
    mail.send(msg)

@app.route('/verify_email/<token>', methods=['GET'])
def verify_email(token):
    try:
        email = s.loads(token, salt='email-confirm', max_age=3600)  # 1 hour expiration
    except SignatureExpired:
        return jsonify({"error": "The verification link has expired"}), 400
    except BadSignature:
        return jsonify({"error": "Invalid verification link"}), 400

    user = mongo.db.users.find_one({"email": email})
    if user:
        if not user['is_verified']:
            mongo.db.users.update_one({"email": email}, {"$set": {"is_verified": True}})
            message = "Email verified successfully! You can now log in."
            alert_type = "success"
        else:
            message = "Email already verified"
            alert_type = "info"
    else:
        message = "Invalid verification link"
        alert_type = "error"
    
    return render_template("verify_email.html", message=message, alert_type=alert_type)
    


@app.route('/signin', methods=['POST'])
def signin():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    user = mongo.db.users.find_one({"email": email})
    if user and check_password_hash(user['password'], password):
        if user['is_verified']:
            token = jwt.encode({
                'user_id': str(user['_id']),
                'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
            }, app.config['SECRET_KEY'], algorithm="HS256")
            return jsonify({"message": "Login successful!", "token": token, "name": user['name'], "email": user['email']}), 200
        else:
            return jsonify({"error": "Please verify your email before logging in."}), 400
    else:
        return jsonify({"error": "Invalid email or password"}), 400


@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.form
    name = data.get('name')
    surname = data.get('surname')
    email = data.get('email')
    message = data.get('message')

    # Send email confirmation
    msg = Message(subject='Thank you for contacting us!',
                  sender=app.config['MAIL_USERNAME'],
                  recipients=[email])
    msg.body = f'Dear {name} {surname},\n\nThank you for contacting us. We have received your message:\n\n"{message}"\n\nWe will get back to you as soon as possible.\n\nBest regards,\nYour Team'
    mail.send(msg)

    # Save the message to MongoDB
    mongo.db.messages.insert_one({
        "name": name,
        "surname": surname,
        "email": email,
        "message": message,
        "timestamp": datetime.datetime.utcnow()
    })

    # Return a rendered HTML template instead of JSON
    return render_template('index.html', success=True)

@app.route('/forgot_password', methods=['GET'])
def forgot_password_form():
    return render_template('forgot.html')

@app.route('/forgot_password', methods=['POST'])
def forgot_password():
    email = request.form.get('email')

    user = mongo.db.users.find_one({'email': email})

    if user:
        token = secrets.token_urlsafe(20)
        mongo.db.users.update_one({'_id': user['_id']}, {'$set': {'reset_token': token}})
        send_password_reset_email(email, token)
        return jsonify({'message': 'Reset password link has been to your email.', 'status': 'success'})
    else:
        return jsonify({'message': 'Email address not found.', 'status': 'error'})

def send_password_reset_email(email, token):
    reset_url = url_for('reset_password', token=token, _external=True)
    msg = Message('Password Reset Request', sender=('DefendHer', 'yashlakhan27@gmail.com'), recipients=[email])
    msg.body = f'Hello,\n\nTo reset your password, click on the following link:\n{reset_url}\n\nIf you did not request this reset, please ignore this email.'
    mail.send(msg)

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    user = mongo.db.users.find_one({'reset_token': token})
    
    if user:
        if request.method == 'POST':
            new_password = request.form.get('password')
            hashed_password = generate_password_hash(new_password)
            mongo.db.users.update_one({'_id': user['_id']}, {'$set': {'password': hashed_password}})
            mongo.db.users.update_one({'_id': user['_id']}, {'$unset': {'reset_token': ''}})
            flash('Your password has been reset successfully.', 'success')
            return redirect(url_for('login'))
        else:
            return render_template('reset.html', token=token)
    else:
        flash('Invalid or expired token. Please try again.', 'error')
        return redirect(url_for('forgot_password_form'))

@app.route('/map')
def map():
    return render_template('map.html')












@app.route('/predict_route', methods=['POST'])
def predict_route():
    try:
        data = request.get_json()
        start_location = data.get('start_location')
        end_location = data.get('end_location')
        police_station_image = data.get('police_station_image')  # Optional

        if not start_location or not end_location:
            return jsonify({'error': 'Start location or end location missing in request'}), 400

        start_coords = get_coordinates(start_location)
        end_coords = get_coordinates(end_location)

        if start_coords and end_coords:
            # Get routes using different profiles
            routes = []
            for profile in ['cycling-regular', 'foot-walking']:  # Removed 'driving-car'
                route = get_route(start_coords, end_coords, profile=profile)
                if route:
                    routes.append((profile, route))

            if len(routes) >= 2:
                route_map_html = generate_routes_map(start_coords, end_coords, routes, police_station_image)
                return jsonify({'map': route_map_html})
            else:
                return jsonify({'error': 'Less than two routes found'}), 404
        else:
            return jsonify({'error': 'Unable to geocode locations'}), 400

    except Exception as e:
        error_message = f'Unexpected error occurred: {str(e)}'
        print(error_message)
        print(traceback.format_exc())
        return jsonify({'error': error_message}), 500

def get_coordinates(location):
    try:
        loc = geolocator.geocode(location, timeout=10)
        if loc:
            return loc.longitude, loc.latitude  # Ensure correct order
        else:
            return None
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        print(f"Geocoding request timed out for {location}. Error: {str(e)}")
        return None

def get_route(start_coords, end_coords, profile='cycling-regular'):
    url = f'https://api.openrouteservice.org/v2/directions/{profile}/geojson'
    headers = {
        'Authorization':'5b3ce3597851110001cf6248295620b8636d462c8c09581404de5dad',
        'Content-Type': 'application/json'
    }
    payload = {
        'coordinates': [start_coords, end_coords]
    }

    response = requests.post(url, headers=headers, json=payload)
    data = response.json()

    print(f"API Response Status Code: {response.status_code}")
    print(f"API Response Data: {data}")

    if response.status_code == 200 and 'features' in data:
        return data['features']
    else:
        return []

def generate_routes_map(start_coords, end_coords, routes, police_station_image):
    route_map = folium.Map(location=[start_coords[1], start_coords[0]], zoom_start=12)

    # Add start and end markers
    folium.Marker(
        location=[start_coords[1], start_coords[0]],
        popup='Start Location',
        icon=folium.Icon(color='darkgreen', icon='star', icon_color='white')
    ).add_to(route_map)

    folium.Marker(
        location=[end_coords[1], end_coords[0]],
        popup='End Location',
        icon=folium.Icon(color='darkred', icon='flag', icon_color='white')
    ).add_to(route_map)

    # Display routes
    colors = ['green', 'red']  # Removed red color
    for i, (profile, route) in enumerate(routes):
        if i >= len(colors):
            break
        route_coords = route[0]['geometry']['coordinates']
        folium.PolyLine(
            locations=[(lat, lng) for lng, lat in route_coords],
            color=colors[i],
            weight=5,
            opacity=0.7,
            popup=f'Route {i + 1} ({profile})'
        ).add_to(route_map)

    # Add markers for nearest police stations and hospitals
    police_stations, hospitals = get_nearest_facilities(start_coords)
    
    for station in police_stations:
        folium.Marker(
            location=station,
            popup='Police Station',
            icon=folium.CustomIcon(
                icon_image=police_station_image if police_station_image else 'static/images/police-badge.png',
                icon_size=(30, 30)
            )
        ).add_to(route_map)

    for hospital in hospitals:
        folium.Marker(
            location=hospital,
            popup='Hospital',
            icon=folium.Icon(color='darkblue', icon='plus', icon_color='white')
        ).add_to(route_map)

    return route_map._repr_html_()

def get_nearest_facilities(coords):
    # Example placeholder coordinates for hospitals and police stations
    # Replace these with actual data if available
    police_stations = [
        (coords[1] + 0.01, coords[0] + 0.01),
        (coords[1] - 0.01, coords[0] - 0.01)
    ]
    hospitals = [
        (coords[1] + 0.02, coords[0] + 0.02),
        (coords[1] - 0.02, coords[0] - 0.02)
    ]
    return police_stations, hospitals






@app.route('/logout', methods=['POST'])
def logout():
    response = jsonify({"message": "Successfully logged out"})
    response.set_cookie('token', '', expires=0)
    return response

@app.route('/user-details', methods=['GET'])
def user_details():
    token = request.headers.get('Authorization')
    if not token:
        return jsonify({"error": "Token is missing"}), 401

    try:
        decoded_token = jwt.decode(token.split(" ")[1], app.config['SECRET_KEY'], algorithms=["HS256"])
        user_id = decoded_token['user_id']
        user = mongo.db.users.find_one({"_id": ObjectId(user_id)})

        if user:
            return jsonify({
                "name": user['name'],
                "email": user['email']
            })
        else:
            return jsonify({"error": "User not found"}), 404
    except jwt.ExpiredSignatureError:
        return jsonify({"error": "Token expired"}), 401
    except jwt.InvalidTokenError:
        return jsonify({"error": "Invalid token"}), 401

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if 'chat_history' not in session:
        session['chat_history'] = []

    if request.method == 'POST':
        user_input = request.form['input']
        if user_input:
            try:
                response = get_gemini_response(user_input)
                session['chat_history'].append(("You", user_input))
                for chunk in response:
                    session['chat_history'].append(("Bot", chunk.text))
                session.modified = True
            except Exception as e:
                logger.error(f"Error during response generation: {e}")
                session['chat_history'].append(("Bot", "Sorry, I encountered an error. Please try again."))

            return redirect(url_for('chat'))

    return render_template('chat.html', chat_history=session.get('chat_history', []))

# Initialize the chat object correctly
def configure_chat_model():
    try:
        # Assuming `start_chat` is a method that initializes a chat session
        chat_instance = model.start_chat(history=[])
        return chat_instance
    except Exception as e:
        logger.error(f"Error initializing chat model: {e}")
        return None

chat = configure_chat_model()  # Properly initialize chat

def get_gemini_response(question):
    if chat is None:
        logger.error("Chat model is not initialized.")
        return []

    try:
        response = chat.send_message(question, stream=True)
        return response
    except genai.types.generation_types.BrokenResponseError:
        logger.error("Encountered BrokenResponseError. Retrying...")
        chat.rewind()
        response = chat.send_message(question, stream=True)
        return response


from flask import Flask, render_template
import feedparser



# URL for Google News RSS feed
RSS_FEED_URL = 'https://rss.app/feeds/v7byX1bTXzlxwIVQ.xml'

@app.route('/news')
def news():
    # Parse the RSS feed
    feed = feedparser.parse(RSS_FEED_URL)
    
    # Extract news entries
    entries = feed.entries

    # Render the feed entries using an HTML template
    return render_template('news.html', entries=entries)








# Load the trained model
model = joblib.load('model/support_prediction_model.pkl')

# Define the possible support labels with additional information
support_labels = {
    "Rape": [
        "counseling: Professional guidance and emotional support to help you navigate your challenges.",
        "medical care: Access to healthcare services for your physical and mental well-being.",
        "legal assistance: Expert advice and representation to help you understand and exercise your legal rights.",
        "shelter: Safe and secure temporary housing for individuals in need of refuge.",
        "reintegration support: Programs and resources to assist you in returning to your community and rebuilding your life."
    ],
    "Sexual Harassment": [
        "counseling: Emotional support to help you deal with the trauma of harassment.",
        "legal assistance: Guidance on how to report harassment and understand your legal protections.",
        "workplace mediation: Help in resolving workplace harassment issues.",
        "support Groups: Connect with others who have experienced similar situations for mutual support."
    ],
    "Domestic Violence": [
        "shelter: Safe temporary housing for those escaping violence.",
        "counseling: Emotional and psychological support to help you through difficult times.",
        "legal assistance: Guidance on protective orders, divorce, and custody rights.",
        "medical care: Access to healthcare services for any physical injuries.",
        "reintegration support: Resources to help you start a new life away from the abuser."
    ],
    "Kidnapping": [
        "reintegration support: Assistance in reuniting with your family and community.",
        "counseling: Psychological support to cope with the trauma of kidnapping.",
        "legal assistance: Help in understanding your legal rights and navigating the justice system.",
        "medical care: Immediate healthcare services for any physical and mental needs."
    ],
    "Assault": [
        "counseling: Emotional support to help you process the trauma of the assault.",
        "legal assistance: Guidance on reporting the assault and understanding your legal rights.",
        "medical care: Immediate healthcare services for physical injuries sustained during the assault.",
        "shelter: Temporary safe housing if you need protection from the assailant.",
        "reintegration support: Programs to help you recover and move forward after the assault."
    ]
}




@app.route('/support')
def support():
    return render_template('support.html')

@app.route('/predict2', methods=['POST'])
def predict2():
    try:
        data = request.get_json()

        # Extract the incident type from the input data
        incident_type = data.get('incident_type', '').lower()

        # Convert JSON data to DataFrame
        df = pd.DataFrame([data])

        # Make prediction
        prediction = model.predict(df)

        # Get support labels based on the incident type
        available_support = support_labels.get(incident_type, support_labels['Rape'])  # Default to rape if incident type is not found

        # Convert prediction to readable format
        support_needed = []
        for idx, value in enumerate(prediction[0]):
            if value == 1 and idx < len(available_support):
                support_needed.append(available_support[idx])

        if not support_needed:
            support_needed.append("General support: We're here to help you with any challenges you might be facing.")

        return jsonify({'support_needed': ', '.join(support_needed)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500



# Initialize the Vonage client
vonage_client = vonage.Client(key="f8de3521", secret="4hP6QOPAK51MDuku")
sms_service = vonage.Sms(vonage_client)

@app.route('/sms')
def sms():
    return render_template('sms.html')

@app.route('/send_location_sms', methods=['POST'])
def send_location_sms():
    try:
        # Fetch the location using an IP-based geolocation API
        response = requests.get('https://ipinfo.io/')
        location_data = response.json()
        latitude, longitude = location_data['loc'].split(',')
        location_url = f"https://www.google.com/maps?q={latitude},{longitude}"

        # Compose the message
        message = f"Here's my location: {location_url}"

        # Send SMS
        responseData = sms_service.send_message(
            {
                "from": "DefendHer",
                "to": "919653333402",  # Replace with the guardian number
                "text": message,
            }
        )

        if responseData["messages"][0]["status"] == "0":
            return jsonify({"status": "success", "message": "Message sent successfully."})
        else:
            return jsonify({"status": "error", "message": responseData['messages'][0]['error-text']}), 500

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500








# Load models (VGG16 and harassment detection model)
def download_model(file_id, output_path):
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)

def load_cached_model(file_id, model_path):
    if not os.path.exists(model_path):
        download_model(file_id, model_path)
    return load_model(model_path)

file_id = "1GP2IdE-mPdQ9D3ouDIqbIU-3gl26Kgf2"
model_path = "weight.hdf5"
harassment_model = load_cached_model(file_id, model_path)
base_model = VGG16(weights='imagenet', include_top=False)

# Function to preprocess image for harassment detection
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Function to make prediction for uploaded image (harassment detection)
def make_prediction(image_path):
    new_image = preprocess_image(image_path)
    new_image_features = base_model.predict(new_image)
    new_image_features = new_image_features.reshape(1, -1)
    predictions = harassment_model.predict(new_image_features)
    predicted_class_index = np.argmax(predictions[0])
    class_labels = {0: 'Healthy Workspace Environment :)', 1: '!! Sexual Harassment Detected !!'}
    predicted_class_label = class_labels[predicted_class_index]
    return predicted_class_label

# Harassment detection: Generate frames for live camera feed
def generate_harassment_frames(harassment_model, base_model):
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        resized_frame = cv2.resize(frame, (224, 224))
        preprocessed_frame = img_to_array(resized_frame)
        preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)
        preprocessed_frame = preprocess_input(preprocessed_frame)

        features = base_model.predict(preprocessed_frame)
        features_flatten = features.reshape(1, -1)
        prediction = harassment_model.predict(features_flatten)[0]
        class_label = np.argmax(prediction)
        class_prob = prediction[class_label]
        label = "Harassment" if class_label == 1 else "Non-Harassment"
        prob_text = f"{label} ({class_prob:.2f})"

        cv2.putText(frame, prob_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Face and emotion detection: Generate frames for live camera feed
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
detected_emotion = "Detecting..."

def generate_emotion_frames():
    global detected_emotion
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            try:
                analysis = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                if isinstance(analysis, list):
                    analysis = analysis[0]
                detected_emotion = analysis['dominant_emotion']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, detected_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            except Exception as e:
                print(f"Error in emotion analysis: {e}")
                detected_emotion = "Error"

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Route for harassment detection via live video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_harassment_frames(harassment_model, base_model),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Route for face and emotion detection via live video feed
@app.route('/videos')
def videos():
    return Response(generate_emotion_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Image upload and harassment detection
@app.route('/harass', methods=['GET', 'POST'])
def harass():
    if request.method == 'POST':
        if 'image' in request.files:
            file = request.files['image']
            if file.filename != '':
                upload_folder = os.path.join('static', 'uploads')
                os.makedirs(upload_folder, exist_ok=True)
                image_path = os.path.join(upload_folder, file.filename)
                file.save(image_path)
                prediction = make_prediction(image_path)
                return render_template('harass_result.html', prediction=prediction, image_path=image_path)
    return render_template('harass.html')

# Route for face emotion detection page
@app.route('/face')
def face():
    return render_template('face.html')

if __name__ == '__main__':
    app.run(debug=True)
