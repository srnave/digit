# from flask import Flask, render_template, request
# from werkzeug.utils import secure_filename
# import tensorflow as tf
# import numpy as np
# import os

# app = Flask(__name__)

# # Define the directory where uploaded files will be stored
# UPLOAD_FOLDER = 'uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Load the saved model
# loaded_model = tf.keras.models.load_model('roman_numeral_model.h5')
# IMAGE_WIDTH, IMAGE_HEIGHT = 64, 64


# class_mapping = {
#     0: 'I',
#     1: 'II',
#     2: 'III',
#     3: 'IV',
#     4: 'V',
#     5: 'VI',
#     6: 'VII',
#     7: 'VIII',
#     8: 'IX',
#     9: 'X'
#     # Add more mappings for your classes if needed
# }

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'image' not in request.files:
#         return "No file part"

#     uploaded_file = request.files['image']

#     if uploaded_file.filename == '':
#         return "No selected file"

#     if uploaded_file:
#         filename = secure_filename(uploaded_file.filename)
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         uploaded_file.save(file_path)

#         new_image = tf.keras.preprocessing.image.load_img(file_path, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT))
#         new_image_array = tf.keras.preprocessing.image.img_to_array(new_image)
#         new_image_array = np.expand_dims(new_image_array, axis=0)
#         new_image_array /= 255.0

#         # Make predictions
#         predictions = loaded_model.predict(new_image_array)
#         predicted_class = np.argmax(predictions, axis=1)[0]
#         roman_numeral = class_mapping.get(predicted_class, "Unknown")
#         confidence_score = predictions[0][predicted_class]

#         return render_template('prediction.html', predicted_class=predicted_class)

# if __name__ == '__main__':
#     app.run(debug=True)
