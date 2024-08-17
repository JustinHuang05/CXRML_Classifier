import requests
import base64
import json

# Correct URL for your Cloud Run service
url = 'https://cxrml-webservice-bxxs7kj6aq-uc.a.run.app/predict'

# Path to the image you want to test
image_path = 'sample_image.png'  # e.g., 'test_image.jpg'

# Load the image and encode it in base64
with open(image_path, 'rb') as image_file:
    image_bytes = image_file.read()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')

# Create the JSON payload
payload = {
    'image_bytes': image_base64
}

# Send the POST request
response = requests.post(url, json=payload)

# Check the response
if response.status_code == 200:
    result = response.json()
    print('Prediction:', result['prediction'])
    print('Gradient Data:', result['gradient_data'])
else:
    print('Failed to get a prediction. Status code:', response.status_code)
    print('Response:', response.text)
