import os
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="Dog vs Cat Classifier | M_FAYAZ_KHAN",
    page_icon="🐾",
    layout="centered"
)

# --- Custom CSS (theme-aware using currentColor and inherit) ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .header {
            text-align: center;
            padding: 2rem 0 1rem 0;
        }
        .header h1 {
            font-size: 2rem;
            font-weight: 700;
            color: inherit;
            margin-bottom: 0.2rem;
        }
        .header p {
            opacity: 0.6;
            font-size: 0.95rem;
            color: inherit;
        }

        [data-testid="stFileUploader"] {
            border: 2px dashed rgba(128,128,128,0.4);
            border-radius: 12px;
            padding: 1rem;
            background: rgba(128,128,128,0.05);
        }

        .result-card {
            border-radius: 16px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid rgba(128,128,128,0.2);
            background: rgba(128,128,128,0.05);
            margin-top: 1rem;
        }
        .result-label {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .result-confidence {
            font-size: 1rem;
            opacity: 0.6;
            color: inherit;
            margin-bottom: 1rem;
        }

        .footer {
            text-align: center;
            opacity: 0.4;
            font-size: 0.8rem;
            padding: 2rem 0 1rem 0;
            color: inherit;
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- Load Model ---
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "dog_cat_final.keras")
    return tf.keras.models.load_model(model_path)

model = load_model()

# --- Preprocessing ---
def preprocess(image):
    img = image.convert("RGB")
    img = img.resize((128, 128))
    arr = np.array(img).astype("float32") / 255.0
    return arr.reshape(1, 128, 128, 3)

# --- Header ---
st.markdown("""
    <div class="header">
        <h1>🐾 Dog vs Cat Classifier</h1>
        <p>Upload a photo — our CNN model will identify it instantly</p>
    </div>
""", unsafe_allow_html=True)


st.warning("⚠️ This model is trained only on dogs and cats. Please do not upload human or other images — results will be incorrect.")
st.markdown("---")

# --- Upload ---
file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if file is not None:
    image = Image.open(file)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, use_container_width=True, caption="Uploaded Image")

    with st.spinner("Analyzing..."):
        processed = preprocess(image)
        pred = model.predict(processed, verbose=0)[0][0]

    is_dog = pred > 0.5
    confidence = pred if is_dog else 1 - pred

    st.markdown("---")

    # --- Confidence threshold check ---
    if confidence < 0.85:
        st.markdown("""
            <div class="result-card">
                <div class="result-label">🤔</div>
                <div class="result-confidence" style="font-size:1.1rem; opacity:1;">
                    This doesn't look like a cat or dog!. Please upload a clear dog or cat image.
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        label = "🐶 Dog" if is_dog else "🐱 Cat"
        color = "#f59e0b" if is_dog else "#6366f1"

        st.markdown(f"""
            <div class="result-card">
                <div class="result-label" style="color: {color};">{label}</div>
                <div class="result-confidence">Confidence: {confidence:.1%}</div>
            </div>
        """, unsafe_allow_html=True)

        st.progress(float(confidence))

        with st.expander("📊 Detailed Probabilities"):
            c1, c2 = st.columns(2)
            with c1:
                st.metric("🐶 Dog", f"{pred:.1%}")
            with c2:
                st.metric("🐱 Cat", f"{1 - pred:.1%}")

else:
    st.markdown("""
        <div style="text-align:center; padding: 2rem; opacity: 0.4;">
            <div style="font-size: 3rem;">📁</div>
            <p>Upload a JPG or PNG image to get started</p>
        </div>
    """, unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
    <div class="footer">
        Developed by <strong>M_FAYAZ_KHAN</strong> &nbsp;|&nbsp; CNN Model trained on Dogs vs Cats Dataset
    </div>
""", unsafe_allow_html=True)