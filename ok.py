from flask import Flask, render_template, Response, request, redirect
import cv2
import numpy as np
import gdown  # To download the model from Google Drive
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os

app = Flask(__name__)

# Function to download the model from Google Drive
def download_model(file_id, output_path):
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)

# Load the model with caching
def load_cached_model(file_id, model_path):
    if not os.path.exists(model_path):
        download_model(file_id, model_path)
    return load_model(model_path)

# Google Drive file ID and model path
file_id = "1GP2IdE-mPdQ9D3ouDIqbIU-3gl26Kgf2"
model_path = "weight.hdf5"

# Load the model
model = load_cached_model(file_id, model_path)
base_model = VGG16(weights='imagenet', include_top=False)

# Function to preprocess the image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension as the model expects it
    img = preprocess_input(img)  # Use VGG16 preprocessing
    return img

# Function to make predictions for an uploaded image
def make_prediction(image_path):
    new_image = preprocess_image(image_path)

    # Extract features using the VGG16 base model
    new_image_features = base_model.predict(new_image)

    # Reshape the features
    new_image_features = new_image_features.reshape(1, -1)

    # Make predictions
    predictions = model.predict(new_image_features)

    # Since your model has 2 output neurons (softmax), you can use argmax to get the predicted class index
    predicted_class_index = np.argmax(predictions[0])

    # Map the index back to class labels
    class_labels = {0: 'Healthy Workspace Environment :)', 1: '!! Sexual Harassment Detected !!'}
    predicted_class_label = class_labels[predicted_class_index]
    return predicted_class_label

# Function to generate frames for the live camera feed
def generate_frames(model, base_model):
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

        prediction = model.predict(features_flatten)[0]
        class_label = np.argmax(prediction)
        class_prob = prediction[class_label]

        label = "Harassment" if class_label == 1 else "Non-Harassment"
        prob_text = f"{label} ({class_prob:.2f})"
        
        # Overlay prediction on the frame
        cv2.putText(frame, prob_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame in byte format to be displayed in the browser
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Route for the combined page with both live camera and image upload
@app.route('/harass', methods=['GET', 'POST'])
def harass():
    if request.method == 'POST':
        # Handle image upload
        if 'image' in request.files:
            file = request.files['image']
            if file.filename != '':
                upload_folder = os.path.join('static', 'uploads')
                os.makedirs(upload_folder, exist_ok=True)
                image_path = os.path.join(upload_folder, file.filename)
                file.save(image_path)
                prediction = make_prediction(image_path)
                return render_template('harass_result.html', prediction=prediction, image_path=image_path)

    return render_template('harass_live.html')

# Route for the live video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(model, base_model),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '_main_':
    app.run(debug=True)