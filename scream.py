from flask import Flask, request, jsonify
import torch
import torchaudio
import numpy as np
import speech_recognition as sr
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

recognizer = sr.Recognizer()

# Function to record audio from microphone
def record_audio(duration=10, fs=16000):
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=None)
            text = recognizer.recognize_google(audio, language='fr-FR,en-US,es-ES,de-DE,it-IT,ar-AR')
            return audio, text
    except Exception as e:
        print(f"Error recording audio: {e}")
        return None, None

# Function to preprocess audio for the model
def preprocess_audio(audio_data, sample_rate=16000, n_mels=64, hop_length=256, win_length=400):
    try:
        if audio_data is None:
            return None

        # Convert audio data to numpy array
        audio_np = np.frombuffer(audio_data.frame_data, dtype=np.int16)

        # Ensure audio is 1D
        audio_np = audio_np.squeeze()

        # Transform audio to tensor
        audio_tensor = torch.tensor(audio_np, dtype=torch.float32)

        # Add batch dimension and channel dimension
        audio_tensor = audio_tensor.unsqueeze(0)

        # Apply transformations
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=n_mels,
            hop_length=hop_length, win_length=win_length
        )(audio_tensor)
        mel_spectrogram_db = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)

        # Replicate the single channel to have three channels
        mel_spectrogram_db = mel_spectrogram_db.repeat(3, 1, 1)

        # Add batch dimension
        mel_spectrogram_db = mel_spectrogram_db.unsqueeze(0)

        return mel_spectrogram_db

    except Exception as e:
        print(f"Error preprocessing audio: {e}")
        return None

# Function to check for fear words in speech
def word_in_speech(text):
    fear_words = [
        "help", "help me", "save me", "emergency", "danger",
        "au secours", "s'il vous plaît", "socorro", "ayuda", "emergenza",
        "النجدة", "المساعدة", "خطر", "طوارئ", "مساعدة",
        "scream", "screaming", "cri", "grito", "grido", "صيحة", "صرخة",
        "aaaaaa", "aaaaaaa", "ahhh", "ahhhh", "aaaaah", "aaaaaah",
        "hilfe", "schrei", "gefahr", "notfall", "bitte", "ayuda", "peligro",
        "socorro", "emergencia", "ayúdame", "auxilio", "aua",
        "au secours", "aidez-moi", "aide", "secours", "danger", "urgence",
        "helfen", "gefahr", "notfall", "bitte", "ayuda", "peligro",
        "socorro", "emergencia", "ayúdame", "auxilio", "aua"
    ]

    for word in fear_words:
        if word in text.lower():
            return True

    return False

@app.route('/detect', methods=['POST'])
def detect():
    T = True
    try:
        # Record audio from microphone
        audio, text = record_audio()

        if audio is None and text is None:
            T = False
        else:
            # Preprocess the recorded audio for the model if audio is available
            features_tensor = preprocess_audio(audio) if audio is not None else None

        if features_tensor is not None:
            # Load the trained model (replace with your model loading code)
            model = torch.load('Emergency_Detector.pt', map_location=torch.device('cpu'))
            model.eval()

            with torch.no_grad():
                prediction = model(features_tensor)

            scream_score = prediction[0][0].item()  # Get the first value from the prediction tensor

            if 50 > scream_score > 0 or (text is not None and word_in_speech(text)):  # Adjust threshold as needed
                return jsonify({"result": True})
            else:
                T = False
        else:
            T = False
    except Exception as e:
        print(f"Error during detection: {e}")
        T = False

    if T == False:
        return jsonify({"result": False})

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=8094)
