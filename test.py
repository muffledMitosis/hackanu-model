import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import io
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Adjustable Variables
IMAGE_SIZE = (224, 224)  # Image size for resizing
ENABLE_RENAMING = True  # Toggle renaming functionality (set to True to enable renaming)

# Function to load the trained model using a relative path
def load_trained_model(model_filename='asbestos_detector.h5'):
    script_dir = os.path.dirname(os.path.realpath(__file__))  # Get the script's directory
    model_path = os.path.join(script_dir, model_filename)
    return load_model(model_path)

# Function to predict if an image contains asbestos and show the confidence level and risk scale
def predict_asbestos_risk(model, image_path):
    # Load the image and preprocess it
    img = image.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img) / 255.  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Get the prediction
    prediction = model.predict(img_array)[0][0]  # Sigmoid output is between 0 and 1

    # Confidence calculation is corrected
    confidence_asbestos = (1 - prediction) * 100  # Confidence in asbestos presence
    confidence_non_asbestos = prediction * 100  # Confidence in non-asbestos

    # Risk scale: 1 means 100% asbestos, 0 means 100% not asbestos
    risk_scale = 1 - round(prediction, 2)

    # Determine class based on prediction threshold
    class_label = 'Not asbestos' if prediction > 0.5 else 'Asbestos'

    # Output the result
    print(f"Prediction for {os.path.basename(image_path)}:")
    if prediction < 0.5:
        print(f"{confidence_asbestos:.2f}% sure this is asbestos.")
    else:
        print(f"{confidence_non_asbestos:.2f}% sure this is not asbestos.")
    print(f"Risk scale: {risk_scale}\n")

# Function to run predictions on all images in a given directory
def run_predictions_on_directory(model, test_images_dir):
    # List all files in the directory
    for image_filename in os.listdir(test_images_dir):
        image_path = os.path.join(test_images_dir, image_filename)
        
        # Check if it's a valid image file by its extension
        if image_filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
            predict_asbestos_risk(model, image_path)

# Function to rename files in the directory to "test 1", "test 2", ..., "test x"
def rename_files_in_directory(directory):
    files = sorted(os.listdir(directory))
    for i, filename in enumerate(files):
        # Only rename image files
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
            new_filename = f"test {i + 1}{os.path.splitext(filename)[1]}"  # Keep the original file extension
            src = os.path.join(directory, filename)
            dst = os.path.join(directory, new_filename)
            os.rename(src, dst)
            print(f"Renamed {filename} -> {new_filename}")

model = load_trained_model('asbestos_detector 18 epoch.h5')

# if __name__ == '__main__':
#     # Load the trained model using a relative path
#     model = load_trained_model('asbestos_detector.h5')  # Ensure the model is in the same directory as the script

#     # Directory containing test images (relative path)
#     test_images_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test images')

#     # Toggle renaming functionality
#     if ENABLE_RENAMING:
#         rename_files_in_directory(test_images_dir)

#     # Run predictions for all files in the test images directory
#     run_predictions_on_directory(model, test_images_dir)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image contents
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img) / 255.
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]

    # Compute confidence
    confidence_asbestos = (1 - prediction) * 100
    confidence_non_asbestos = prediction * 100
    risk_scale = 1 - round(prediction, 2)
    class_label = 'asbestos' if prediction > 0.5 else 'not asbestos'

    result = {
        'filename': file.filename,
        'class_label': class_label,
        'confidence_asbestos': confidence_asbestos,
        'confidence_non_asbestos': confidence_non_asbestos,
        'risk_scale': risk_scale
    }

    return JSONResponse(content=result)