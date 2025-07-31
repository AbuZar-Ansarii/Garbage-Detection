import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Set custom page config
st.set_page_config(page_title="Garbage Classifier", layout="centered")

# Load model and class names
model = load_model("garbage_classifier_cnn.h5")
class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Title section
st.markdown("<h1 style='text-align: center;'>🗑️ Garbage Classification</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an image of waste and click predict to classify it.</p>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("📤 Choose an image file (JPG, PNG):", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

    # Predict button
    if st.button("🔍 Predict"):
        with st.spinner("Predicting..."):
            # Preprocess image
            img = image.resize((150, 150))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Prediction
            prediction = model.predict(img_array)
            pred_class = class_labels[np.argmax(prediction)]
            confidence = np.max(prediction)

        # Results display
        st.success(f"🎯 Prediction: **{pred_class.capitalize()}**")
        st.info(f"📊 Confidence: **{confidence*100:.2f}%**")
