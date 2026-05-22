# 🤟 AI-Based Bidirectional Sign Language Communication System

> Real-time Indian Sign Language ↔ Text/Voice using MediaPipe + TensorFlow + Streamlit

---

## 📸 Demo Screenshot
### Dashboard 

 <img width="1016" height="447" alt="image" src="https://github.com/user-attachments/assets/48fee1e4-ec1b-4335-b605-f241c2073e15" />

 ### Live 
 <img width="1906" height="841" alt="image" src="https://github.com/user-attachments/assets/cccbbdca-949c-4d4a-a2e2-d12ea3486a78" />

 
### Regional Language Translation
 <img width="1016" height="331" alt="image" src="https://github.com/user-attachments/assets/5e1701bc-ea4c-4d89-a10e-c806a6ea2a18" />
 
 ### Regional Language Translation
 <img width="1016" height="366" alt="image" src="https://github.com/user-attachments/assets/a377e4ad-3f22-4db2-932f-580965f7a4a6" />

 ## 📁 Project Structure

```
sign_language_system/
│
├── collect_data.py          # Step 1 – Dataset collection via webcam
├── train_model.py           # Step 2 – Train neural network
├── app.py                   # Step 3 – Streamlit web app (main UI)
├── requirements.txt         # All dependencies
│
├── modules/                 # Core AI modules
│   ├── __init__.py
│   ├── hand_detector.py     # MediaPipe hand landmark detection
│   ├── gesture_recognizer.py# TensorFlow gesture classification
│   ├── sentence_builder.py  # Gesture → sentence accumulation
│   ├── tts_engine.py        # Text-to-speech (pyttsx3 / gTTS)
│   └── sign_display.py      # Text/Voice → Sign visuals
│
├── dataset/
│   └── landmarks.csv        # Generated during data collection
│
├── model/
│   ├── gesture_model.h5     # Saved Keras model (after training)
│   ├── label_encoder.npy    # Gesture class names
│   ├── norm_mean.npy        # Normalization stats
│   ├── norm_std.npy
│   └── training_history.png # Accuracy/Loss plot
│
└── signs/
    ├── words/               # Sign images per word (Hello.jpg, etc.)
    └── alphabets/           # Fingerspelling A–Z images
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.9 or 3.10 (recommended)
- Webcam connected
- Windows / Linux / macOS

### 2. Create Virtual Environment
```bash
# Create venv
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/macOS)
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the System

### Step 1 — Collect Dataset
```bash
python collect_data.py
```
- A camera window opens for each gesture
- Press **S** to start recording samples
- Press **Q** to skip a gesture
- Collects 300 samples per gesture by default

### Step 2 — Train the Model
```bash
python train_model.py
```
- Trains the neural network (~100 epochs with early stopping)
- Saves model to `model/gesture_model.h5`
- Prints accuracy report and saves training graph

### Step 3 — Launch the Web App
```bash
streamlit run app.py
```
- Opens automatically at: **http://localhost:8501**
- Go to **Tab 1** → Click **▶️ Start Camera**
- Perform ISL gestures in front of your webcam

---

## 🎯 Features

| Feature | Description |
|---------|-------------|
| Real-time Gesture Recognition | 21 MediaPipe landmarks → TensorFlow classifier |
| Sentence Formation | Debounced accumulation of gesture words |
| Text-to-Speech | pyttsx3 (offline) or gTTS (online) |
| Text → Sign Visuals | Word or fingerspelling sign images |
| Voice → Sign Visuals | Microphone → Google STT → Sign grid |
| Streamlit UI | 4-tab interface with live camera feed |

---

## 📊 Model Architecture

```
Input (42 features: 21 landmarks × x,y)
    ↓
Dense(256) → BatchNorm → Dropout(0.4)
    ↓
Dense(256) → BatchNorm → Dropout(0.3)
    ↓
Dense(128) → BatchNorm → Dropout(0.3)
    ↓
Dense(64) → Dropout(0.2)
    ↓
Dense(N_classes) → Softmax
```

---

## 🤟 Supported Gestures (Default)

**Words:** Hello, Thank You, Yes, No, Please, Sorry, Help, Water, Food, I Love You, Good, Bad, Stop, Come, Go

**Alphabets:** A–Z (fingerspelling fallback)

---

## ➕ Adding New Gestures

1. Add gesture name to `GESTURES` list in `collect_data.py`
2. Place sign image in `signs/words/<GESTURE_NAME>.jpg`
3. Re-run `python collect_data.py`
4. Re-run `python train_model.py`

---

## Dataset Information
ISL Alphabets, Numbers, Greetings, Common Words, Healthcare Signs
Training:72%
Validation:14%
Testing:14%
Accuracy:99.01%

---

## 🛠️ Common Issues

| Issue | Fix |
|-------|-----|
| Camera not found | Change `camera_index` in Streamlit sidebar |
| Low accuracy | Collect 500+ samples, improve lighting |
| pyttsx3 error on Linux | `sudo apt-get install espeak` |
| No microphone | Use Text Input mode in Tab 2 |

---

## 📋 Requirements

- opencv-python
- mediapipe
- tensorflow
- numpy, pandas, scikit-learn
- streamlit
- pyttsx3, gTTS
- SpeechRecognition
- Pillow, matplotlib

  ## 🎓 Final Conclusion

- The AI-Based Bidirectional Sign Language Communication System provides an intelligent and accessible solution for reducing communication barriers between deaf and hearing individuals. By integrating Computer Vision, MediaPipe, TensorFlow, and Streamlit, the system enables real-time Indian Sign Language recognition and supports bidirectional communication through text and speech conversion.

- The project demonstrates how Artificial Intelligence can be applied to create inclusive technologies that improve accessibility and social interaction. With high model accuracy and support for future expansion, the system can be further enhanced by adding more gestures, multilingual support, and advanced deep learning models for better real-world performance.

- This project represents a step toward building smarter and more inclusive communication systems for society.

---

*Built for inclusive communication — bridging the gap between deaf/mute individuals and the world.*
