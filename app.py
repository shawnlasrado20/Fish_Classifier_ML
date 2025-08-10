import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Set page title
st.title("🐟 Fish Species Classification")

# Load your trained model (with cached loading)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('mobilenet_fish_classifier.h5')  # your model filename
    return model

model = load_model()

# Correct class names list matching your model output classes
class_names = [
    'animal fish',
    'animal fish bass',
    'fish sea_food black_sea_sprat',
    'fish sea_food gilt_head_bream',
    'fish sea_food hourse_mackerel',
    'fish sea_food red_mullet',
    'fish sea_food red_sea_bream',
    'fish sea_food sea_bass',
    'fish sea_food shrimp',
    'fish sea_food striped_red_mullet',
    'fish sea_food trout'
]

def preprocess_image(image):
    # Resize and normalize image to match model input
    size = (224, 224)  # update if different for your model
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = np.array(image) / 255.0  # normalize pixels to [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array

# Image uploader
uploaded_file = st.file_uploader("Upload an image of a fish", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    input_img = preprocess_image(image)

    # Prediction
    predictions = model.predict(input_img)[0]
    top_idx = np.argmax(predictions)
    confidence = predictions[top_idx]

    # Display prediction and confidence (safe index assumed)
    st.markdown(f"### Prediction: **{class_names[top_idx]}**")
    st.markdown(f"Confidence: **{confidence * 100:.2f}%**")

    # Show full confidence scores for all classes
    st.markdown("#### Confidence Scores:")
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {predictions[i] * 100:.2f}%")
else:
    st.write("Please upload an image to classify.")
