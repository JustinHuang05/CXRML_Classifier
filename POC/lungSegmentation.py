import ssl
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from torchvision import transforms
from PIL import Image

ssl._create_default_https_context = ssl._create_unverified_context

# Load the pre-trained lung segmentation model (U-Net with ResNet34 backbone)
def load_model():
    model = smp.Unet(
        encoder_name='resnet50',         # Use ResNet34 as the backbone
        encoder_weights='imagenet',      # Pre-trained on ImageNet
        in_channels=3,                   # Input channels (RGB images)
        classes=1                        # Output is a binary mask (lungs or not lungs)
    )
    model.eval()  # Set the model to evaluation mode
    return model

# Preprocess the input image (resize, normalize, etc.)
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')  # Load the image and convert to RGB
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to (256, 256)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
    ])
    return preprocess(image).unsqueeze(0)  # Add batch dimension for the model

# Generate the lung mask from the model
def get_lung_mask(model, input_image):
    with torch.no_grad():
        output = model(input_image)  # Run forward pass
        mask = torch.sigmoid(output).squeeze(0).cpu().numpy()  # Apply sigmoid and remove batch dim
        mask = mask[0]  # First (and only) channel in the output mask
        mask = (mask > 0.5).astype(np.uint8)  # Convert to binary mask (0 or 1)
    return mask

# Apply the mask to the original image, masking out everything except the lungs
def apply_mask_to_image(original_image_path, mask, output_image_path):
    original_image = cv2.imread(original_image_path)  # Load the original image
    original_image = cv2.resize(original_image, (256, 256))  # Resize to match the mask size

    # Create a mask for the areas that are not lungs (invert the lung mask)
    lung_mask = np.stack([mask]*3, axis=-1)  # Stack to create a 3-channel mask
    masked_image = np.where(lung_mask == 1, original_image, 0)  # Apply mask, make non-lung regions black

    # Save the masked image
    cv2.imwrite(output_image_path, masked_image)
    print(f'Masked image saved to {output_image_path}')

if __name__ == "__main__":
    image_path = '/Users/justinhuang/Documents/Developer/ML/CXRML/POC/CXRData/test/COVID/COVID_20.png'  # Input image path
    output_image_path = 'masked_lung_image.jpg'  # Path to save the masked image

    # Load the lung segmentation model
    model = load_model()
    
    # Preprocess the input image
    input_image = preprocess_image(image_path)

    # Generate the lung mask
    mask = get_lung_mask(model, input_image)

    # Apply the mask and save the result
    apply_mask_to_image(image_path, mask, output_image_path)
