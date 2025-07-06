#  Breast Cancer Prediction Using Mammograms

This project is a web application built with **Streamlit** that uses a trained **EfficientNet-based deep learning model** to predict the presence of breast cancer from mammogram images (in `.png` format).

<div align="center">
  <img src="assets/demo.png" alt="App Demo Screenshot" width="80%"/>
</div>

---

##  Live Demo

[Launch the Streamlit App](https://share.streamlit.io/pragyapanwar23/breast-cancer-prediction/main/app.py)  
*(replace with actual link once deployed)*

---

##  How It Works

1. User uploads a `.png` mammogram image.
2. The image is preprocessed and passed to the trained CNN.
3. The model returns a prediction: **Benign (0)** or **Malignant (1)**.
4. Optionally: explainable visualizations can be added.

---

##  Model Details

- **Architecture**: EfficientNetB3 (transfer learning)
- **Framework**: TensorFlow / Keras
- **Input shape**: Resized mammograms to match model (e.g., 224x224)
- **Output**: Binary classification

---

##  Setup & Run Locally

###  Requirements

Install dependencies:

```bash
pip install -r requirements.txt
