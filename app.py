import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

MODEL_PATH  = "model/plant_disease_model.keras"
LABELS_PATH = "class_names.json"

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH) as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_model()

def format_label(raw):
    if "___" in raw:
        plant, disease = raw.split("___", 1)
    elif "_" in raw:
        plant, disease = raw.split("_", 1)
    else:
        return raw.replace("_", " ").title(), "Unknown"
    return plant.replace("_", " ").title(), disease.replace("_", " ").title()

# ── Page ──────────────────────────────────────────────────────────
st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿")
st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image to detect the disease.")

uploaded = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_column_width=True)

    arr = np.array(img.resize((224, 224)), dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    with st.spinner("Analysing..."):
        preds = model.predict(arr, verbose=0)[0]

    idx        = int(np.argmax(preds))
    confidence = preds[idx] * 100
    plant, disease = format_label(class_names[idx])

    st.markdown("---")
    st.subheader("🔍 Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("🌱 Plant",      plant)
    col2.metric("🦠 Disease",    disease)
    col3.metric("📊 Confidence", f"{confidence:.1f}%")

    if confidence < 50:
        st.warning("⚠️ Low confidence — try a clearer image.")
    else:
        st.success("✅ Prediction complete!")