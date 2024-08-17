from flask import Flask, request, jsonify, send_from_directory, render_template
import requests
import os
from PIL import Image
import numpy as np
import base64
import json

app = Flask(__name__, template_folder='templates', static_folder='static')

UPLOAD_FOLDER = 'uploads'
TEST_DATA_FOLDER = 'cxr_test_data'
EXTERNAL_TEST_DATA_FOLDER = 'external_test_data'

ENDPOINT_URL = 'https://cxrml-webservice-bxxs7kj6aq-uc.a.run.app/predict' 

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/about')
def about_page():
    return render_template('about.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        filename = 'uploaded_image.jpg'
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        file.save(file_path)

        try:
            img = Image.open(file_path)
            img.verify()
        except Exception as e:
            return jsonify({'error': 'File is not a valid image'})

        with open(file_path, 'rb') as image_file:
            image_bytes = image_file.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        response = requests.post(ENDPOINT_URL, json={'image_bytes': image_base64})
        result = response.json()
        gradient_data = np.array(result['gradient_data'])

        return jsonify({'filename': filename, 'gradient_data': gradient_data.tolist(), 'predicted_label': result['prediction']})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/cxr_test_data/<path:filename>')
def test_data_file(filename):
    file_path = os.path.join(TEST_DATA_FOLDER, filename)
    if not os.path.exists(file_path):
        return "File not found", 404
    return send_from_directory(TEST_DATA_FOLDER, filename)

@app.route('/external_test_data/<path:filename>')
def test_external_data_file(filename):
    file_path = os.path.join(EXTERNAL_TEST_DATA_FOLDER, filename)
    if not os.path.exists(file_path):
        return "File not found", 404
    return send_from_directory(EXTERNAL_TEST_DATA_FOLDER, filename)

@app.route('/list_test_images', methods=['GET'])
def list_test_images():
    testFolder = request.args.get('testFolder', default=TEST_DATA_FOLDER, type=str)
    if os.path.exists(testFolder):
        images = []
        for root, dirs, files in os.walk(testFolder):
            for file in sorted(files):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    relative_path = os.path.relpath(os.path.join(root, file), testFolder)
                    images.append(relative_path)
        images.sort()
        return jsonify(images)
    return jsonify([])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)

