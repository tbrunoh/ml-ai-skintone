import torch as T
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
from google.cloud import aiplatform
import os
from dotenv import load_dotenv
load_dotenv()

PROJECT_ID = os.getenv("GCLOUD_PROJECT_ID")  # Replace with your project ID
LOCATION = os.getenv("GCLOUD_LOCATION")  # Replace with your region (e.g., "us-central1")
ENDPOINT_ID = os.getenv("VERTEX_ENDPOINT")  # Replace with your endpoint id

aiplatform.init(project=PROJECT_ID, location=LOCATION)
endpoint = aiplatform.Endpoint(endpoint_name=ENDPOINT_ID)

# Define a function to preprocess the image
def preprocess_image(image_path_or_url):
    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Check if the input is a URL or a local path
    if image_path_or_url.startswith("http://") or image_path_or_url.startswith("https://"):
        response = requests.get(image_path_or_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_path_or_url).convert("RGB")

    # Apply the transformations
    return transform(image).unsqueeze(0)  # Add batch dimension

# Define a function to classify the image
def classify_image(image_path_or_url):
    try:
        # Preprocess the image
        input_tensor = preprocess_image(image_path_or_url)

        # Convert the tensor to a list of lists for Vertex AI
        instances = input_tensor.tolist()

        # Perform inference using Vertex AI
        prediction = endpoint.predict(instances=instances)

        # Extract the predicted class from the prediction response
        predicted_class = T.argmax(T.tensor(prediction.predictions[0]), dim=0).item()

        return {
                "num":predicted_class
                }
    except Exception as e:
        return f"Error during classification: {e}"

# Example usage
image_path_or_url = 'https://sdrimltdata.blob.core.windows.net/melanintrainingimages/mst/PXL_20220922_185757301.jpg'  #input("Enter the image path or URL: ")
result = classify_image(image_path_or_url)
print(result)
