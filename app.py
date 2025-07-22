import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import base64
from io import BytesIO

def Image_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return encoded

model = tf.keras.models.load_model("densenet121.h5")
class_names = ['Kabam', 'Pithakabam', 'Pithalipitham', 'Pitham', 'Pithavatham']
class_descriptions = {
    "Kabam": "Associated with coldness and heaviness in the body; may indicate respiratory issues or sluggish metabolism.",
    "Pithakabam": "A mixed condition involving both fire and mucus; could suggest indigestion, inflammation, or mucus imbalance.",
    "Pithalipitham": "Linked with excess bile and acidity; might suggest digestive disturbances, skin rashes, or irritability.",
    "Pitham": "Primarily bile-dominant; commonly related to acidity, ulcers, and heat-related disorders.",
    "Pithavatham": "Combination of bile and wind; may reflect nervousness, hyperacidity, or fluctuating energy levels."
}

st.set_page_config(page_title="Neykuri Classifier", layout="wide")

st.markdown("""
    <style>
        .title {
            font-size: 42px;
            text-align: center;
            margin-bottom: 10px;
        }
        .upload-note {
            text-align: center;
            font-size: 16px;
            margin-top: -10px;
            margin-bottom: 30px;
        }
        .pred-card {
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            text-align: center;
            transition: 0.3s;
        }
        .pred-card:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .prediction {
            margin-top: 12px;
            font-weight: 600;
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title"> Neykuri Pattern Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="upload-note">Upload up to <b>5 Neikuri images</b> for prediction</div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader("Upload Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

def crop_to_bowl(pil_img):
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 5
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return pil_img.resize((224, 224))

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    if w < 50 or h < 50:
        return pil_img.resize((224, 224))

    cropped = img[y:y+h, x:x+w]
    cropped_img = Image.fromarray(cropped).resize((224, 224))
    return cropped_img

def predict_image(pil_img):
    cropped_img = crop_to_bowl(pil_img)
    img_array = np.array(cropped_img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    return predicted_class, cropped_img

if uploaded_files:
    st.subheader("Prediction Results")
    cols = st.columns(min(len(uploaded_files), 5))

    for i, uploaded_file in enumerate(uploaded_files[:5]):
        image = Image.open(uploaded_file).convert("RGB")
        predicted_class, processed_img = predict_image(image)
        description = class_descriptions[predicted_class]

        with cols[i]:
            st.markdown('<div class="pred-card">', unsafe_allow_html=True)

            img_bytes = processed_img.resize((224, 224))
            st.markdown(
                f'<img src="data:image/png;base64,{Image_to_base64(img_bytes)}" '
                f'style="width:100%; height:auto; border-radius:8px;" />',
                unsafe_allow_html=True
            )

            st.markdown(f'<div class="prediction">Predicted: {predicted_class}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:14px; color:#555;">{description}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)


