import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from io import BytesIO
from timm import create_model
from flask import Flask, request, jsonify
import os
from pathlib import Path
import base64

app = Flask(__name__)

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

def load_image(image_bytes, transform):
    image = Image.open(BytesIO(image_bytes)).convert('L')
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

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_base64 = data['image_bytes']
    image_bytes = base64.b64decode(image_base64)
    model_path = 'best_model.pth'
    output_dir = 'output'

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
    
    image = load_image(image_bytes, transform).to(device)
    output = model(image)
    target_class = output.argmax().item()
    
    class_names = ['Covid', 'Normal', 'Pneumonia']
    
    guided_grads = guided_bp.generate_gradients(image, target_class)
    grayscale_guided_grads = convert_to_grayscale(guided_grads)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    npy_file = Path(output_dir) / 'guided_grads.npy'
    
    save_gradient_data(grayscale_guided_grads, npy_file)
        
    gradient_data = np.load(npy_file).tolist()  # Convert numpy array to list for JSON serialization
    
    return jsonify({'gradient_data': gradient_data, 'prediction': class_names[target_class]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
