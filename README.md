#  Cat vs Dog Classifier — Model Progression Study

> **From 83% to 97.34% accuracy — a step-by-step CNN journey from scratch to fine-tuned MobileNetV2.**

A deep learning project that classifies images as **Dog** or **Cat** using four progressive approaches, showing exactly how each technique improves real-world performance.

🔗 **Live Demo:** [cat-vs-dog-recognizer.streamlit.app](https://cat-vs-dog-recognizer.streamlit.app/)

---

## 📸 Preview

| Upload | Prediction |
|--------|------------|
| Upload any dog or cat image | Get instant prediction with confidence score |

---

## 🧠 Model Architectures

### 1. Scratch CNN
> No augmentation — pure baseline to measure improvement.

Built from scratch using TensorFlow/Keras Functional API:

```
Input (128×128×3)
    ↓
Conv2D(32, relu) → MaxPooling
    ↓
Conv2D(64, relu) → MaxPooling
    ↓
Conv2D(128, relu) → MaxPooling
    ↓
Flatten → Dense(128, relu) → Dropout(0.3)
    ↓
Dense(1, sigmoid) → Output
```

---

### 2. Scratch CNN + Data Augmentation
> Same architecture, but teaches the model to handle real-world image variation.

```
RandomFlip (horizontal)
RandomRotation (±10%)
RandomZoom (±10%)
    ↓
[Same CNN architecture as above]
```

---

### 3. Transfer Learning (MobileNetV2)
> Why train from scratch when ImageNet already learned the visual world?

Pretrained MobileNetV2 (ImageNet) used as a frozen feature extractor:

```
Input (128×128×3)
    ↓
MobileNetV2 (frozen, ImageNet weights)
    ↓
GlobalAveragePooling2D
    ↓
Dense(128, relu) → Dropout(0.3)
    ↓
Dense(1, sigmoid) → Output
```

---

### 4. Fine-Tuned MobileNetV2
> Unfreezing the last 30 layers and retraining carefully — the final performance push.

```
MobileNetV2 (last 30 layers unfrozen)
    ↓
Adam(learning_rate=1e-5) — to preserve pretrained features
    ↓
[Same custom head as Transfer Learning]
```

---

## 📊 Training Results

| Model | Val Accuracy | Key Detail |
|-------|-------------|------------|
| Scratch CNN | 83.00% | Baseline, no augmentation |
| Scratch + Augmentation | 88.72% | RandomFlip / Rotation / Zoom |
| Transfer Learning | 96.88% | MobileNetV2 frozen, ImageNet weights |
| **Fine-Tuned MobileNetV2** | **97.34%** | Last 30 layers unfrozen |

**Common Training Config:**

| Setting | Value |
|---------|-------|
| Dataset | Dogs vs Cats (Kaggle) |
| Train / Val Split | 20,000 / 5,000 images |
| Input Size | 128 × 128 |
| Optimizer | Adam |
| Loss | Binary Crossentropy |
| Early Stopping | patience=3, restore_best_weights=True |

---

## 🛠️ Tech Stack

- **Model:** TensorFlow / Keras
- **Pretrained Base:** MobileNetV2 (ImageNet)
- **App:** Streamlit
- **Deployment:** Streamlit Cloud
- **Training Hardware:** Tesla T4 GPU (Kaggle)
- **Version Control:** Git + Git LFS

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
├── app.py                            # Streamlit web app
├── cat-vs-dog-data_augmented.keras   # Scratch CNN + Augmentation model
├── dog_cat_final.keras               # Final fine-tuned MobileNetV2 model
├── requirements.txt                  # Dependencies
└── README.md                         # You are here
```

---

## 🔍 How It Works

1. User uploads a JPG/PNG image
2. Image is resized to 128×128 and normalized to [0, 1]
3. Model predicts probability via sigmoid output
4. Result shown with confidence score and probability breakdown

---

## 💡 Key Learnings

- Data augmentation reduces overfitting and improves generalization by ~5%
- Transfer learning delivers drastically better accuracy with far less training time
- Fine-tuning with `learning_rate=1e-5` pushes accuracy further without destroying pretrained features
- `EarlyStopping` with `restore_best_weights=True` is essential to avoid overfitting

---

## 👨‍💻 Developer

**M. Fayaz Khan** — Aspiring AI/ML Engineer  
Self-taught, building real projects from scratch.  
🔗 [LInkedIn](https://www.linkedin.com/in/muhammad-fayaz-khan-271487381/)

---

## 📌 Note

This is a learning project built as part of a structured AI/ML journey.  
All models trained on Kaggle (Tesla T4 GPU), final app deployed on Streamlit Cloud.