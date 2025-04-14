
import torch as T
from mlaist.models.model import SkinToneCNNModel
import torchvision.transforms as transforms
from dotenv import load_dotenv
import os
from PIL import Image
import requests
from io import BytesIO
load_dotenv()


model_save_path = f'{os.getenv("MODEL_SAVE_PATH")}/skin_tone_model.pth'  # Default path if not set in environment

# Create a mapping for Monk Skin Tone names to numbers
monk_skin_tone_mapper = {
    0: "Very Light",
    1: "Light",
    2: "Light Medium",
    3: "Medium",
    4: "Medium Deep",
    5: "Deep",
    6: "Very Deep",
    7: "Dark",
    8: "Very Dark",
    9: "Deepest Dark"
}
mst_orbs_mapper = {
    0: "https://sdrimltdata.blob.core.windows.net/melanintrainingimages/orbs/MST_1.png",
    1: "https://sdrimltdata.blob.core.windows.net/melanintrainingimages/orbs/MST_2.png",
    2: "https://sdrimltdata.blob.core.windows.net/melanintrainingimages/orbs/MST_3.png",
    3: "https://sdrimltdata.blob.core.windows.net/melanintrainingimages/orbs/MST_4.png",
    4: "https://sdrimltdata.blob.core.windows.net/melanintrainingimages/orbs/MST_5.png",
    5: "https://sdrimltdata.blob.core.windows.net/melanintrainingimages/orbs/MST_6.png",
    6: "https://sdrimltdata.blob.core.windows.net/melanintrainingimages/orbs/MST_7.png",
    7: "https://sdrimltdata.blob.core.windows.net/melanintrainingimages/orbs/MST_8.png",
    8: "https://sdrimltdata.blob.core.windows.net/melanintrainingimages/orbs/MST_9.png",
    9: "https://sdrimltdata.blob.core.windows.net/melanintrainingimages/orbs/MST_10.png"
}
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
                "number":predicted_class,
                "name":monk_skin_tone_mapper[predicted_class],
                "orb": mst_orbs_mapper[predicted_class]
                }
    except Exception as e:
        return f"Error during classification: {e}"

# # Example usage
# image_path_or_url = 'https://sdrimltdata.blob.core.windows.net/melanintrainingimages/mst/PXL_20220922_185757301.jpg'  #input("Enter the image path or URL: ")
# result = classify_image(image_path_or_url)
# print(result)