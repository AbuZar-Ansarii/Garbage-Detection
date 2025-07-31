import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load model and define classes
model = load_model("garbage_classifier_cnn.h5")
class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Title
st.title("🗑️ Garbage Classification App")
st.markdown("Upload an image of waste, and the model will predict its type.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert to RGB and display
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width =True)

    # Preprocess image
    img = image.resize((150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)
    pred_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Display result
    st.success(f"Prediction: **{pred_class.capitalize()}**")
    st.info(f"Confidence: {confidence*100:.2f}%")
