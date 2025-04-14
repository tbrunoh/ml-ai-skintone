
import torch as T
from models.model import SkinToneCNNModel
import torchvision.transforms as transforms
from dotenv import load_dotenv
import os
from PIL import Image
import requests
from io import BytesIO
load_dotenv()


model_save_path = f'{os.getenv("MODEL_SAVE_PATH")}/skin_tone_model.pth'  # Default path if not set in environment


# Load the saved model for inference
model_inference = SkinToneCNNModel(num_classes=10, image_size=512, channels=3)
model_inference.load_state_dict(T.load(model_save_path))
model_inference.eval()  # Set the model to evaluation mode
print("Model loaded for inference.")



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
        
        # Perform inference
        with T.no_grad():
            output = model_inference(input_tensor)
            predicted_class = T.argmax(output, dim=1).item()
        
        return {
                "num":predicted_class
                }
    except Exception as e:
        return f"Error during classification: {e}"

# Example usage
image_path_or_url = 'https://sdrimltdata.blob.core.windows.net/melanintrainingimages/mst/PXL_20220922_185757301.jpg'  #input("Enter the image path or URL: ")
result = classify_image(image_path_or_url)
print(result)