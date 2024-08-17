from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from timm import create_model
from pathlib import Path

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

class MLModel(nn.Module):
    def __init__(self, num_classes=3):
        super(MLModel, self).__init__()
        self.base_model = create_model('densenet121', pretrained=True)
        self.base_model.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        in_features = self.base_model.classifier.in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

class GuidedBackpropagation:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.model.eval()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            if isinstance(module, nn.ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)
            return None

        for module in self.model.modules():
            module.register_backward_hook(hook_function)

    def generate_gradients(self, input_image, target_class):
        input_image.requires_grad = True
        output = self.model(input_image)
        self.model.zero_grad()
        output[0, target_class].backward(retain_graph=True)
        return input_image.grad.data.cpu().numpy()[0]

def load_image(image_path, transform):
    image = Image.open(image_path).convert('L')
    image = transform(image).unsqueeze(0)
    return image

def convert_to_grayscale(im_as_arr):
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    return np.expand_dims(grayscale_im, axis=0)

def save_gradient_data(gradient, file_name):
    np.save(file_name, gradient)

def run_model(image_path, model_path, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MLModel(num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    guided_bp = GuidedBackpropagation(model)

    image = load_image(image_path, transform).to(device)
    output = model(image)
    target_class = output.argmax().item()

    class_names = ['Covid', 'Normal', 'Pneumonia']

    guided_grads = guided_bp.generate_gradients(image, target_class)
    grayscale_guided_grads = convert_to_grayscale(guided_grads)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    npy_file = Path(output_dir) / 'guided_grads.npy'
    save_gradient_data(grayscale_guided_grads, npy_file)
    return npy_file, class_names[target_class]


@app.route('/')
def index():
    return render_template('index-test.html')

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
        file.save(file_path)
        npy_file, predicted_label = run_model(file_path, 'best_model.pth', OUTPUT_FOLDER)
        gradient_data = np.load(npy_file).tolist()  # Convert numpy array to list for JSON serialization
        return jsonify({'filename': filename, 'gradient_data': gradient_data, 'predicted_label': predicted_label})


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5002, debug=True)
