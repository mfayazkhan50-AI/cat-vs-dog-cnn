# 🐾 Cat vs Dog Classifier

A deep learning web app that classifies images as **Dog** or **Cat** using a Convolutional Neural Network (CNN) built from scratch with TensorFlow/Keras.

🔗 **Live Demo:** [cat-vs-dog-recognizer.streamlit.app](https://cat-vs-dog-recognizer.streamlit.app/)

---

## 📸 Preview

| Upload | Prediction |
|--------|------------|
| Upload any dog or cat image | Get instant prediction with confidence score |

---

## 🧠 Model Architecture

Built from scratch using TensorFlow/Keras Sequential API:

```
Input (256x256x3)
    ↓
Conv2D(32) → BatchNorm → MaxPooling
    ↓
Conv2D(64) → BatchNorm → MaxPooling
    ↓
Conv2D(128) → BatchNorm → MaxPooling
    ↓
Flatten → Dense(128) → Dropout(0.3)
    ↓
Dense(64) → Dropout(0.2)
    ↓
Dense(1, sigmoid) → Output
```

---

## 📊 Training Results

| Metric | Value |
|--------|-------|
| Dataset | Dogs vs Cats (20,000 images) |
| Input Size | 256 × 256 |
| Best Val Accuracy | **85.6%** |
| Optimizer | Adam |
| Loss | Binary Crossentropy |
| Early Stopping | patience=3 |

Training was stopped early at **Epoch 2** (best weights) to prevent overfitting.

---

## 🛠️ Tech Stack

- **Model:** TensorFlow / Keras
- **App:** Streamlit
- **Deployment:** Streamlit Cloud
- **Version Control:** Git + Git LFS (model ~170MB)

---

## 🚀 Run Locally

```bash
git clone https://github.com/mfayazkhan50-AI/cat-vs-dog-cnn
cd cat-vs-dog-cnn
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure

```
cat-vs-dog-cnn/
│
├── app.py                        # Streamlit web app
├── dog_cat_classifier_v2.keras   # Trained CNN model
├── requirements.txt              # Dependencies
└── README.md                     # You are here
```

---

## 🔍 How It Works

1. User uploads a JPG/PNG image
2. Image is resized to 256×256 and normalized
3. CNN model predicts probability (sigmoid output)
4. Result shown with confidence score and probability breakdown

---

## 👨‍💻 Developer

**M. Fayaz Khan** — Aspiring AI/ML Engineer  
Self-taught, building real projects from scratch.

---

## 📌 Note

This is a learning project built as part of a structured AI/ML journey.  
Model trained on Google Colab, deployed on Streamlit Cloud.
