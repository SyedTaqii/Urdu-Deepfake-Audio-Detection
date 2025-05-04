import numpy as np
import joblib
import torch
import torchaudio
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

# Preprocessing parameters
SAMPLE_RATE = 16000
NUM_MFCC = 13
DURATION = 3  # seconds
MAX_LEN = SAMPLE_RATE * DURATION

# MFCC transformation
mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=SAMPLE_RATE,
    n_mfcc=NUM_MFCC,
    melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23}
)

# Dataset Paths (adjust this according to your local paths)
bonafide_path = "D:\Git\Hugging-Face\deepfake_detection_dataset_urdu\Bonafide"
spoofed_tacotron_path = "D:\Git\Hugging-Face\deepfake_detection_dataset_urdu\Spoofed_Tacotron"
spoofed_tts_path = "D:\Git\Hugging-Face\deepfake_detection_dataset_urdu\Spoofed_TTS"

# Function to load audio files
def load_audio_files(path, label):
    audio_files = []
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(path):
        for file in files:
            # Check if the file is a .wav file
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                try:
                    waveform, sample_rate = torchaudio.load(file_path)
                    audio_files.append((waveform, sample_rate, label))
                except Exception as e:
                    print(f"Error loading file {file_path}: {e}")
    return audio_files

# Load data from all directories recursively
bonafide_data = load_audio_files(bonafide_path, "Bonafide")
spoofed_tacotron_data = load_audio_files(spoofed_tacotron_path, "Spoofed")
spoofed_tts_data = load_audio_files(spoofed_tts_path, "Spoofed")

# Print the total number of files loaded from each folder
print(f"Bonafide data loaded: {len(bonafide_data)} files")
print(f"Spoofed_Tacotron data loaded: {len(spoofed_tacotron_data)} files")
print(f"Spoofed_TTS data loaded: {len(spoofed_tts_data)} files")

# Combine all data into one list
all_audio_data = bonafide_data + spoofed_tacotron_data + spoofed_tts_data

# Check the total number of samples loaded
print(f"Total number of samples loaded: {len(all_audio_data)}")

# Preprocess each audio file and extract MFCC features
def preprocess(example):
    waveform, sr, label = example
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)

    # Pad or truncate the waveform to fit the desired length
    if waveform.shape[1] < MAX_LEN:
        pad_len = MAX_LEN - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_len))
    else:
        waveform = waveform[:, :MAX_LEN]

    mfcc = mfcc_transform(waveform).squeeze().T  # shape: (time_steps, n_mfcc)
    return {'features': mfcc.numpy(), 'label': label}

# Process dataset and extract features
processed_data = [preprocess(example) for example in tqdm(all_audio_data)]

# Convert to feature matrix
X = np.array([ex['features'].flatten() for ex in processed_data])
y = np.array([ex['label'] for ex in processed_data])

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Model Building
print("Initializing models...")
svm_model = SVC(kernel='rbf', probability=True)
log_model = LogisticRegression(max_iter=2000)
perceptron_model = Perceptron(max_iter=1000)
dnn_model = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=200)

# Split the dataset into training and testing sets
print("Splitting and training...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Training each model
print("Training SVM...")
svm_model.fit(X_train, y_train)

print("Training Logistic Regression...")
log_model.fit(X_train, y_train)

print("Training Perceptron...")
perceptron_model.fit(X_train, y_train)

print("Training DNN...")
dnn_model.fit(X_train, y_train)

# Save the trained models
joblib.dump(svm_model, './audio_models/svm_model.pkl')
joblib.dump(log_model, './audio_models/log_model.pkl')
joblib.dump(perceptron_model, './audio_models/perceptron_model.pkl')
joblib.dump(dnn_model, './audio_models/dnn_model.h5')

# Evaluation function
def evaluate(model, name):
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred

    print(f"\n{name} Evaluation:")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score : {f1_score(y_test, y_pred):.4f}")
    try:
        print(f"AUC-ROC  : {roc_auc_score(y_test, y_proba):.4f}")
    except:
        print("AUC-ROC  : N/A")

# Evaluate the models
evaluate(svm_model, "SVM")
evaluate(log_model, "Logistic Regression")
evaluate(perceptron_model, "Perceptron")
evaluate(dnn_model, "Deep Neural Network")