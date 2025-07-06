# model.py
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

print("📦 Starting model load...")
model = load_model("final_model.h5")
print("✅ final_model.h5 loaded successfully")

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

