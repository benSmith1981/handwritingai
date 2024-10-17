from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

# Register the custom CTC loss function
@register_keras_serializable(package="Custom", name="ctc_loss")
def ctc_loss(args):
    y_true, y_pred, input_length, label_length = args
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

# Load the model with custom_objects
model = load_model('handwriting_model.h5', custom_objects={'ctc_loss': ctc_loss}, compile=False)
# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained handwriting recognition model
# model = load_model('handwriting_model.h5', compile=False)

# Function to preprocess the uploaded image to match the input shape of the model
def preprocess_image(image):
    # Convert image to grayscale and resize to (32, 128)
    image = image.convert('L')
    image = image.resize((128, 32), Image.ANTIALIAS)
    # Normalize and reshape for model input
    image = ImageOps.invert(image)  # Invert colors if necessary
    image_array = np.array(image)
    image_array = image_array.astype('float32') / 255.0
    image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded file from the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        # Open the image file
        image = Image.open(file)
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make a prediction using the model
        prediction = model.predict(processed_image)
        
        # Post-process prediction (this would depend on how your model's output is formatted)
        predicted_text = np.argmax(prediction, axis=-1)  # Example: get the predicted character indexes
        predicted_text = ''.join([str(char) for char in predicted_text[0]])  # Convert indexes to text
        
        return jsonify({'prediction': predicted_text})

if __name__ == '__main__':
    app.run(debug=True)
