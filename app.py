from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename
from ai_derma_bot.utils import detect_and_respond
from gtts import gTTS
import numpy as np
import librosa
import speech_recognition as sr
import tempfile
from flask_cors import CORS
import os
app = Flask(__name__)

# Home Page
@app.route("/")
def index():
    return render_template("index.html")

# Chat Page (BotPress Chat)
@app.route("/chat")
def chat_page():
    return render_template("chat.html")

# Chat with the DermaBot
@app.route('/dermabot')
def chat():
    return render_template('chat.html')

def transcribe_audio(audio_path):
    r = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = r.record(source)
    return r.recognize_google(audio)

def analyze_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    
    # Enhanced features
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    harmonic = librosa.effects.harmonic(y)
    
    analysis = {
        'energy': float(np.sqrt(np.mean(y**2))),
        'pitch': float(np.median(librosa.piptrack(y=y, sr=sr)[0][librosa.piptrack(y=y, sr=sr)[0] > 0])),
        'zcr': float(np.mean(librosa.feature.zero_crossing_rate(y))),
        'mfcc_mean': float(np.mean(mfcc)),
        'spectral_centroid': float(np.mean(spectral_centroid)),
        'harmonics': float(np.mean(harmonic))
    }
    
    # Enhanced classification logic
    if (analysis['energy'] < 0.02 and analysis['pitch'] < 100 and 
        analysis['zcr'] < 0.05 and analysis['spectral_centroid'] < 1200):
        analysis['status'] = "Likely Sick (vocal fatigue detected)"
    else:
        analysis['status'] = "Likely Healthy"
    
    return analysis

@app.route('/voice')
def voice_page():
    return render_template('voice.html')

@app.route('/voice-check', methods=['POST'])
def voice_analysis():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file received'}), 400
            
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        # Ensure WAV format
        if not audio_file.filename.lower().endswith('.wav'):
            return jsonify({'error': 'Only WAV files supported'}), 415

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            audio_file.save(tmp.name)
            try:
                transcript = transcribe_audio(tmp.name)
                analysis = analyze_audio(tmp.name)
            except Exception as e:
                app.logger.error(f"Analysis failed: {str(e)}")
                return jsonify({'error': 'Audio processing failed'}), 500
            finally:
                os.unlink(tmp.name)

        return jsonify({'transcript': transcript, **analysis})

    except Exception as e:
        app.logger.error(f"Server error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

# Skin disease detection page
@app.route("/derma_bot")
def skin_disease_page():
    return render_template("derma_bot.html")

UPLOAD_FOLDER = "temp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/dermabot-detect", methods=["POST"])
def dermabot_detect():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    filename = secure_filename(file.filename)
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(image_path)

    prediction_data = detect_and_respond(image_path)  # This returns dict

    return jsonify(prediction_data)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)

