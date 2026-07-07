import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

DATASET_PATH = "dataset/Archive/AudioWAV"

def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=3)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

features = []
labels = []

for file in os.listdir(DATASET_PATH):
    if file.endswith(".wav"):
        path = os.path.join(DATASET_PATH, file)
        feature = extract_features(path)
        features.append(feature)

        if "ANG" in file:
            labels.append("Angry")
        elif "HAP" in file:
            labels.append("Happy")
        elif "SAD" in file:
            labels.append("Sad")
        else:
            labels.append("Neutral")

X = np.array(features)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/emotion_model.pkl")

print("Model trained successfully!")

