from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
import cv2
import numpy as np
from custom_layers import GatedConv2D
import os

app = Flask(__name__)

# Define the directory where uploaded files will be stored
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the saved models
model_digit = tf.keras.models.load_model('mnist_digit_recognition_model.h5', custom_objects={'GatedConv2D': GatedConv2D})
model_roman = tf.keras.models.load_model('roman_numeral_model.h5')
IMAGE_WIDTH, IMAGE_HEIGHT = 64, 64

class_mapping_roman = {
    
    1: 'I',
    2: 'II',
    3: 'III',
    4: 'IV',
    5: 'V',
    6: 'VI',
    7: 'VII',
    8: 'VIII',
    9: 'IX'
    # Add more mappings for your classes if needed
}
roman_to_digit_mapping = {
    'I': '1',
    'II': '2',
    'III': '3',
    'IV': '4',
    'V': '5',
    'VI': '6',
    'VII': '7',
    'VIII': '8',
    'IX': '9'
    # Add more mappings for Roman numerals as needed
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file part"

    uploaded_file = request.files['image']

    if uploaded_file.filename == '':
        return "No selected file"

    if uploaded_file:
        filename = secure_filename(uploaded_file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_file.save(file_path)

        new_image = tf.keras.preprocessing.image.load_img(file_path, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT))
        new_image_array = tf.keras.preprocessing.image.img_to_array(new_image)
        new_image_array = np.expand_dims(new_image_array, axis=0)
        new_image_array /= 255.0

        # Make predictions
        predictions_roman = model_roman.predict(new_image_array)
        predicted_class_roman = np.argmax(predictions_roman, axis=1)[0]
        roman_numeral = class_mapping_roman.get(predicted_class_roman, "Unknown")
        confidence_score_roman = predictions_roman[0][predicted_class_roman]
        predicted_roman = roman_to_digit_mapping.get(roman_numeral, "Unknown")
        # Print the predicted class and confidence score
        print("Predicted Class (Roman):", predicted_roman)
        print("Confidence Score (Roman):", confidence_score_roman)
        return render_template('prediction.html', predicted_roman=predicted_roman)



@app.route('/predictDigit', methods=['POST'])
def predictDigit():
    if 'image' not in request.files:
        return "No file part"

    uploaded_file = request.files['image']

    if uploaded_file.filename == '':
        return "No selected file"

    if uploaded_file:
        # Load and preprocess the new image for digit prediction
        filename = secure_filename(uploaded_file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_file.save(file_path)
        img = cv2.imread(file_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

        # Normalize the new image
        new_img = tf.keras.utils.normalize(resized, axis=1)
        new_img = np.array(new_img).reshape(-1, 28, 28, 1)

        # Make predictions on the new image
        predictions_digit = model_digit.predict(new_img)
        predicted_digit = np.argmax(predictions_digit)
        print("Predicted Class (Digit):", predicted_digit)

        return render_template('prediction1.html', predicted_digit=predicted_digit)

if __name__ == '__main__':
    app.run(debug=True)