import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import io
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

genai.configure(api_key=os.environ["API_KEY"])
gen_model = genai.GenerativeModel("gemini-1.5-flash")

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

    prompt = f"""

    You are an expert who specialises in making risk assessment plans. Imagine you work for the government, meaning
    you are incredibly strict with your assessments. You have been asked to provide a risk assessment based on
    an asbestos detection test for a building renovation:
    Probability of Asbestos: {confidence_asbestos:.2f}%
    Classification: {class_label}

    I need you to do the following:
    1. the format of the report should be the following way. It is always a table with the following 5 columns:
        a. Preventative/risk and minimisation strategy
        b. Identified Risk 
        c. Potential Source of Risk 
        d. Actions Required 
        e. Proposed Outcomes
    
    2. Depending on the classification, provide a risk assessment for both cases (Asbestos and Not Asbestos).
        a. If there is asbestos, depending on the probability, I need you to provide a detailed risk 
           assessment plan in the table format.
        b. If there is no asbestos, mention that there is no risk of asbestos but provide a general risk, but thorough
    Format the response as a structured report.
        c. At the end of the report, always mention that more images should be taken for a more accurate assessment.
        Also, mention that the report is based on the current image and the probability of asbestos detection, and that 
        the risk assessment is subject to change based on further testing. Also mention that an expert should be consulted if needed.
    
        3. Generate the report in an html format. Make sure to include the probability and classification in the report. But 
         write them in a way that is easy to understand for a layman.
    """

    response = gen_model.generate_content(prompt) # takes a prompt as input and returns a response

    result = {
        'filename': file.filename,
        'class_label': class_label,
        'confidence_asbestos': confidence_asbestos,
        'risk_assessment_plan': response.text
    }

    return JSONResponse(content=result)

