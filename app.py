import streamlit as st
import numpy as np
import torch
import torchaudio
import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import hamming_loss, precision_score, f1_score
import os

# Set paths for the saved models
audio_model_dir = './audio_models'
defect_model_dir = './defect_models'

# Load Pre-trained Models for Audio Detection
audio_models = {
    'SVM': joblib.load(os.path.join(audio_model_dir, 'svm_model.pkl')),
    'Logistic Regression': joblib.load(os.path.join(audio_model_dir, 'log_model.pkl')),
    'Perceptron': joblib.load(os.path.join(audio_model_dir, 'perceptron_model.pkl')),
    'DNN': joblib.load(os.path.join(audio_model_dir, 'dnn_model.h5'))
}

# Load Pre-trained Models for Defect Detection
defect_models = {
    'SVM': joblib.load(os.path.join(defect_model_dir, 'svm_model_defect.pkl')),
    'Logistic Regression': joblib.load(os.path.join(defect_model_dir, 'log_reg_model_defect.pkl')),
    'Perceptron': joblib.load(os.path.join(defect_model_dir, 'perceptron_model_defect.pkl')),
    'DNN': joblib.load(os.path.join(defect_model_dir, 'dnn_model_defect.h5'))
}

# Load the TfidfVectorizer that was used for training (ensure that it was saved during training)
defect_tfidf_vectorizer = joblib.load(os.path.join(defect_model_dir, 'tfidf_vectorizer.pkl'))

# Preprocess Audio for Prediction
def preprocess_audio(file):
    SAMPLE_RATE = 16000
    NUM_MFCC = 13
    DURATION = 3  # seconds
    MAX_LEN = SAMPLE_RATE * DURATION

    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=NUM_MFCC,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23}
    )

    # Load audio file
    waveform, sr = torchaudio.load(file)

    # Resample if necessary
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)

    # Pad or truncate the waveform to fit the desired length
    if waveform.shape[1] < MAX_LEN:
        pad_len = MAX_LEN - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_len))
    else:
        waveform = waveform[:, :MAX_LEN]

    # Extract MFCC features
    mfcc = mfcc_transform(waveform).squeeze().T  # shape: (time_steps, n_mfcc)
    features = mfcc.numpy().flatten()

    return features

# Preprocess Defect Data
def preprocess_defect_data(data, vectorizer=None):
    # Separate numeric and text columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    text_cols = data.select_dtypes(include=[object]).columns

    # Handle missing values for numeric columns (using mean imputation)
    imputer = SimpleImputer(strategy='mean')
    data[numeric_cols] = imputer.fit_transform(data[numeric_cols])

    # Handle text columns - here, we'll use TF-IDF vectorizer to transform text data into numeric features
    if vectorizer is None:
        # Use the pre-loaded vectorizer for inference
        text_data = vectorizer.transform(data['report'])
    else:
        text_data = vectorizer.fit_transform(data['report'])

    # Convert the sparse matrix to a DataFrame
    text_data_df = pd.DataFrame(text_data.toarray(), columns=vectorizer.get_feature_names_out())

    # Combine the processed text data with the rest of the dataset
    data_imputed = pd.concat([data.drop(columns=text_cols), text_data_df], axis=1)
    
    return data_imputed

# Streamlit UI Setup
st.title("Real-time Prediction for Audio and Defect Detection")

# Create Tabs to separate Audio and Defect Prediction interfaces
tab_selection = st.sidebar.selectbox(
    'Select a task to perform:',
    ['Audio Detection', 'Defect Detection']
)

if tab_selection == 'Audio Detection':
    st.header("Audio Detection: Deepfake vs Bonafide")
    
    # Model Selection Dropdown for Audio
    model_choice = st.selectbox('Select a Model for Audio Prediction', ('SVM', 'Logistic Regression', 'Perceptron', 'DNN'))
    st.write(f"Using {model_choice} for Audio Prediction.")
    
    # Audio File Upload
    audio_file = st.file_uploader("Upload Audio File", type=["wav"])
    
    if audio_file:
        # Preprocess and predict for audio
        audio_features = preprocess_audio(audio_file)
        model = audio_models[model_choice]
        prediction = model.predict([audio_features])
        st.write(f"Prediction: {'Bonafide' if prediction == 0 else 'Deepfake'}")
        
        # Optionally, print confidence scores
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba([audio_features])[0]
            st.write(f"Confidence: {max(proba)*100:.2f}%")

elif tab_selection == 'Defect Detection':
    st.header("Software Defect Prediction")

    # Input CSV file upload
    uploaded_file = st.file_uploader("Upload Defect Feature Data (CSV)", type=["csv"])

    if uploaded_file:
        # Read the CSV data
        data = pd.read_csv(uploaded_file)
        st.write(f"Data preview:\n{data.head()}")
        
        # Preprocess defect data
        processed_data = preprocess_defect_data(data, vectorizer=defect_tfidf_vectorizer)
        scaler = StandardScaler()
        processed_data_scaled = scaler.fit_transform(processed_data)

        # Model Selection for Defect Prediction
        model_choice_defect = st.selectbox('Select a Model for Defect Prediction', ('SVM', 'Logistic Regression', 'Perceptron', 'DNN'))
        model_defect = defect_models[model_choice_defect]

        # Make prediction
        prediction_defect = model_defect.predict(processed_data_scaled)
        st.write(f"Predicted Labels: {prediction_defect}")
        
        # Confidence scores for multi-label prediction
        if hasattr(model_defect, 'predict_proba'):
            probas = model_defect.predict_proba(processed_data_scaled)
            for i, prob in enumerate(probas):
                st.write(f"Sample {i+1}: {prob}")

# Footer
st.write("Built with Streamlit for real-time predictions!")