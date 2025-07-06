import os
import gdown
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

MODEL_PATH = "final_model.h5"
GDRIVE_ID = "1LEj0ozx2HtSHm7vX5m279GlxHek3F7MY"

# Download model from Google Drive if not present
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={GDRIVE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)
    print("✅ Download complete.")

# Load model
print("📦 Loading model...")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

def preprocess_image(image, target_size=(224, 224)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0]
    return prediction
