from flask import Flask, render_template, request
import librosa
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load model
model = joblib.load("model/emotion_model.pkl")

# Feature extraction
def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=3)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']

        if file.filename == '':
            return "No file selected"

        file_path = "temp.wav"
        file.save(file_path)

        # extract features
        features = extract_features(file_path)
        features = np.array(features).reshape(1, -1)

        # prediction
        prediction = model.predict(features)[0]

        # accuracy (confidence)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            accuracy = round(np.max(proba) * 100, 2)
        else:
            accuracy = "N/A"

        return render_template(
            'index.html',
            prediction=prediction,
            accuracy=accuracy
        )

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)