# Import necessary modules
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import cv2
import numpy as np
from custom_layers import GatedConv2D 

# Initialize the Flask app
app = Flask(__name__)

# Define a function to preprocess and predict the image
def predict_digit(image_path):
    # Load and preprocess the new image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

    # Normalize the new image
    new_img = tf.keras.utils.normalize(resized, axis=1)
    new_img = np.array(new_img).reshape(-1, 28, 28, 1)

    # Make predictions on the new image
    predictions = model.predict(new_img)
    predicted_digit = np.argmax(predictions)
    return predicted_digit

# Define a route to render the upload form
@app.route('/')
def index():
    return render_template('upload.html')

# Define a route to handle the image upload and display predictions
@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file:
            image_path = "temp.jpg"  # Save the uploaded image temporarily
            uploaded_file.save(image_path)
            predicted_digit = predict_digit(image_path)
            return render_template('result.html', predicted_digit=predicted_digit)

if __name__ == '__main__':
    # Load your trained model with a custom object scope
    model = tf.keras.models.load_model('mnist_digit_recognition_model.h5', custom_objects={'GatedConv2D': GatedConv2D})

    app.run(debug=True)
