from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import os

# === Constants ===
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
EXPLANATIONS = {
    'glioma': 'Glioma is a type of tumor that starts in the glial cells of the brain or spinal cord.',
    'meningioma': 'Meningioma is a tumor that originates in the meninges.',
    'notumor': 'This indicates no tumor detected in the image.',
    'pituitary': 'Pituitary tumors are abnormal growths in the pituitary gland.'
}

# === Flask App Setup ===
app = Flask(__name__, static_folder='../frontend', static_url_path='/')
CORS(app)

# === Load the TFLite Model ===
interpreter = tf.lite.Interpreter(model_path="brain_tumor_classifier.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
prediction_history = []

# === Utility: Preprocess Image ===
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((240, 240))
    img_array = np.asarray(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# === Routes ===
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    preprocessed = preprocess_image(file.read())

    interpreter.set_tensor(input_details[0]['index'], preprocessed)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    confidence = float(np.max(output_data))
    predicted_class = CLASS_NAMES[np.argmax(output_data)]
    explanation = EXPLANATIONS[predicted_class]

    result = {
        'predicted_class': predicted_class,
        'confidence': f"{confidence * 100:.2f}%",
        'explanation': explanation
    }

    prediction_history.append(result)
    return jsonify(result)

@app.route('/history', methods=['GET'])
def get_history():
    return jsonify(prediction_history)

@app.route('/<path:path>')
def serve_static_files(path):
    return send_from_directory(app.static_folder, path)

# === Run App ===
if __name__ == '__main__':
    app.run(debug=True)
