# app.py
import streamlit as st
from PIL import Image
from model import predict
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Mammogram Classifier", layout="centered")

st.title("🩺 Mammogram Classifier")
st.write("Upload a **.png** mammogram image to get a prediction.")

uploaded_file = st.file_uploader("Choose a PNG image", type=["png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Mammogram", use_column_width=True)

    st.write("Processing and predicting...")
    prediction = predict(image)

    class_names = ["Benign", "Malignant"]
    pred_label = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.success(f"Prediction: **{pred_label}** with confidence **{confidence:.2f}**")

    fig, ax = plt.subplots()
    ax.bar(class_names, prediction, color=["green", "red"])
    ax.set_ylabel("Confidence")
    ax.set_title("Prediction Confidence")
    st.pyplot(fig)
