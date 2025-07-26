import requests
import streamlit as st
from PIL import Image

st.title("🧠 Alzheimer MRI Classifier")
uploaded_file = st.file_uploader("Upload a brain MRI image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    uploaded_file.seek(0)

    if st.button("Run Prediction"):
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        response = requests.post("http://backend:8000/predict", files=files)

        if response.ok:
            result = response.json()
            st.write("### 🧠 Prediction Result")
            st.write(f"Prediction: {result['prediction']}")
            st.write(f"Confidence: {result['confidence']:.1%}")
