import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from timm import create_model
from torchvision.datasets import ImageFolder
from tqdm import tqdm

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

def apply_threshold(gradient, threshold_ratio):
    flat = gradient.flatten()
    flat.sort()
    threshold_value_index = min(int(len(flat) * threshold_ratio), len(flat) - 1)
    threshold_value = flat[threshold_value_index]
    return np.where(gradient > threshold_value, gradient, 0)

def create_mask(original_image, threshold=30):
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    height = mask.shape[0]
    exclude_top = int(height * 0.15)
    exclude_bottom = int(height * 0.1)
    mask[:exclude_top, :] = 0
    mask[-exclude_bottom:, :] = 0
    width = mask.shape[1]
    exclude_left = int(width * 0.15)
    exclude_right = int(width * 0.15)
    mask[:, :exclude_left] = 0
    mask[:, -exclude_right:] = 0
    return mask

def add_text_to_image(image, text, font_size=24, color=(255, 0, 0)):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    draw.text(((image_pil.width - text_width) / 2, 10), text, font=font, fill=color)
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

def save_side_by_side_image(original, overlay, file_name, correct, true_label, pred_label):
    combined_image = np.concatenate((original, overlay), axis=1)
    if correct:
        text = f"Correct! Predicted: {pred_label}"
        combined_image = add_text_to_image(combined_image, text, color=(0, 255, 0))
    else:
        text = f"Incorrect! Predicted: {pred_label}, True: {true_label}"
        combined_image = add_text_to_image(combined_image, text, color=(255, 0, 0))
    cv2.imwrite(file_name, combined_image)

def save_gradient_images(gradient, original_image, alpha=0.3, threshold_ratio=0.85):
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    gradient = np.uint8(gradient * 255)
    gradient = gradient.transpose(1, 2, 0)

    gradient_thresholded = apply_threshold(gradient, threshold_ratio)
    gradient_resized = cv2.resize(gradient_thresholded, (original_image.shape[1], original_image.shape[0]))
    gradient_resized = np.stack([gradient_resized] * 3, axis=-1)
    yellow_highlight = np.zeros_like(original_image)
    yellow_highlight[:, :, 0] = 255
    yellow_highlight[:, :, 1] = 255

    mask = gradient_resized > 0
    exclude_mask = create_mask(original_image)
    yellow_overlay = np.where(mask & (exclude_mask[:, :, None] > 0), cv2.addWeighted(yellow_highlight, alpha, original_image, 1 - alpha, 0), original_image)

    return yellow_overlay

def main(image_path, model_path, class_names):
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

    image = load_image(image_path, transform).to(device)
    output = model(image)
    target_class = output.argmax().item()

    true_label = Path(image_path).stem.split('_')[0]
    correct = class_names[target_class] == true_label

    guided_bp = GuidedBackpropagation(model)
    guided_grads = guided_bp.generate_gradients(image, target_class)
    grayscale_guided_grads = convert_to_grayscale(guided_grads)

    original_image = cv2.imread(str(image_path))
    original_image_resized = cv2.resize(original_image, (224, 224))

    def update_overlay(threshold_ratio):
        adjusted_ratio = threshold_ratio / 100.0
        overlay_image = save_gradient_images(grayscale_guided_grads, original_image_resized, alpha=0.3, threshold_ratio=adjusted_ratio)
        display_image = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
        cv2.imshow('Guided Backpropagation Overlay', display_image)

    def on_trackbar(val):
        update_overlay(val)

    cv2.namedWindow('Guided Backpropagation Overlay')
    cv2.createTrackbar('Threshold', 'Guided Backpropagation Overlay', 85, 100, on_trackbar)
    update_overlay(85)

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def interactive_main(image_dir, model_path):
    image_paths = list(Path(image_dir).rglob('*.jpg')) + list(Path(image_dir).rglob('*.png'))
    
    if not image_paths:
        print("No images found in the specified directory.")
        return

    dataset = ImageFolder(root=image_dir, transform=None)
    class_names = dataset.classes

    selected_image_path = random.choice(image_paths)
    main(selected_image_path, model_path, class_names)

image_dir = '/Users/justinhuang/Documents/Developer/ML/CXRML/CXRData/test'
model_path = 'best_model.pth'

interactive_main(image_dir, model_path)
