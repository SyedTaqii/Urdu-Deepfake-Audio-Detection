import streamlit as st
import numpy as np
import pandas as pd
import librosa
import joblib
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer

# ========== CONFIG ========== 
st.set_page_config(page_title="Real-Time Prediction App", layout="wide")

# ========== AUDIO FEATURE EXTRACTOR ========== 
def extract_mfcc(audio_file, sr=16000, n_mfcc=13, max_len=200):
    y, _ = librosa.load(audio_file, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_len:
        pad = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc.flatten().reshape(1, -1)

# ========== LOAD MODEL FUNCTION ========== 
@st.cache_resource
def load_pickle_model(path):
    return joblib.load(path)

@st.cache_resource
def load_keras_model(path):
    return load_model(path)

# ========== MODEL FILE OPTIONS ========== 
AUDIO_MODELS = {
    "SVM": "audio_models/svm_model.pkl",
    "Logistic Regression": "audio_models/log_reg_model.pkl",
    "Perceptron": "audio_models/perceptron_model.pkl",
    "DNN": "audio_models/dnn_model.h5",
}

DEFECT_MODELS = {
    "Tuned Logistic Regression": "defect_models/log_reg_model_defect.pkl",
    "Tuned SVM": "defect_models/svm_model_defect.pkl",
    "Tuned DNN (Keras)": "defect_models/dnn_model_defect.h5",
    "Tuned Perceptron ": "defect_models/perceptron_model_defect.pkl",
}

# ========== SIDEBAR ========== 
task = st.sidebar.selectbox("Select Task", ["🎙️ Deepfake Audio Detection", "🐞 Software Defect Prediction"])

# ========== TASK 1: AUDIO ========== 
if task == "🎙️ Deepfake Audio Detection":
    st.title("🎙️ Deepfake Audio Detection")
    
    # Model Selection Dropdown for Audio
    model_name = st.sidebar.selectbox("Choose Model", list(AUDIO_MODELS.keys()))
    model_path = AUDIO_MODELS[model_name]
    
    # Audio File Upload
    audio_file = st.file_uploader("Upload a WAV file", type=["wav"])
    
    if audio_file:
        st.audio(audio_file)
        
        # Extract features from audio
        features = extract_mfcc(audio_file)
        
        if st.button("Run Prediction"):
            if model_name == "DNN":
                model = load_keras_model(model_path)
                prob = float(model.predict(features)[0][0])
                pred = int(prob >= 0.5)
            else:
                model = load_pickle_model(model_path)
                if hasattr(model, "predict_proba"):
                    prob = float(model.predict_proba(features)[0][1])
                elif hasattr(model, "decision_function"):
                    raw_score = model.decision_function(features)
                    prob = float(1 / (1 + np.exp(-raw_score[0])))
                else:
                    prob = 0.5  # fallback
                pred = model.predict(features)[0]
            
            label = "Deepfake" if pred else "Bonafide"
            st.success(f"Prediction: **{label}**")
            st.metric("Confidence", f"{prob:.2f}")

# ========== TASK 2: MULTI-LABEL ========== 
elif task == "🐞 Software Defect Prediction":
    st.title("🐞 Software Defect Multi-Label Prediction")
    
    # Model Selection for Defect Prediction
    model_name = st.sidebar.selectbox("Choose Model", list(DEFECT_MODELS.keys()))
    model_path = DEFECT_MODELS[model_name]
    
    # Input CSV file upload
    input_csv = st.file_uploader("Upload a feature vector (CSV with 1 row)", type=["csv"])
    
    if input_csv:
        input_df = pd.read_csv(input_csv)
        features = input_df.values
        
        if st.button("Run Prediction"):
            if model_name == "Tuned DNN (Keras)":
                model = load_keras_model(model_path)
                probs = model.predict(features)[0]
                preds = (probs >= 0.5).astype(int)
            else:
                model = load_pickle_model(model_path)
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(features)[0]
                else:
                    probs = [None] * features.shape[1]
                preds = model.predict(features)[0]
            
            # Displaying Prediction Results
            st.write("### Prediction Results")
            labels = input_df.columns.tolist() if len(input_df.columns) == len(preds) else [f"Label {i+1}" for i in range(len(preds))]
            
            for i, label in enumerate(labels):
                status = "✅" if preds[i] else "❌"
                conf = f" (confidence: {probs[i]:.2f})" if probs[i] is not None else ""
                st.write(f"- {label}: {status}{conf}")

# Footer
st.write("Built with Streamlit for real-time predictions!")
