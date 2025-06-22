import numpy as np
import librosa

def analyze_audio_features(audio_file):
    """Analyze audio features for sickness detection."""
    try:
        # Load audio file
        audio_data, sr = librosa.load(audio_file, sr=16000)

        # Energy calculation (Root Mean Square - RMS)
        energy = np.sqrt(np.mean(audio_data**2))

        # Pitch extraction (only positive values considered)
        pitch, _ = librosa.piptrack(y=audio_data, sr=sr)
        avg_pitch = np.median(pitch[pitch > 0])

        # Speech rate (Zero Crossing Rate - ZCR)
        zcr = librosa.feature.zero_crossing_rate(audio_data)
        avg_zcr = np.mean(zcr)

        # Classification based on audio characteristics
        if energy < 0.02 and avg_pitch < 100 and avg_zcr < 0.05:
            return "Likely Sick (low energy, slow speech, low pitch)"
        elif avg_pitch > 120 and energy > 0.05 and avg_zcr > 0.1:
            return "Likely Healthy (normal pitch and energy)"
        else:
            return "Likely Sick (hoarseness detected or low energy)"
    
    except Exception as e:
        return f"Error analyzing audio: {str(e)}"
