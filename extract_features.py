import librosa
import numpy as np


def extract_music_features(audio_path):

    y, sr = librosa.load(audio_path)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    spectral_centroid = np.mean(
        librosa.feature.spectral_centroid(y=y, sr=sr)
    )

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    pitch_var = np.var(chroma)

    rms = np.mean(librosa.feature.rms(y=y))

    features = {

        "tempo": float(tempo),

        "pitch_var": float(pitch_var),

        "spectral_centroid": float(spectral_centroid),

        "energy": float(rms),

        "harmony_complexity": float(np.mean(chroma)),

        "rhythm_stability": 0.7,

        "section_count": 4,

        "structure_variation": 0.6,

        "valence": 0.5,

        "novelty": 0.5,

        "genre_fusion": 0.4,

        "clarity": 0.7,

        "feature_consistency": 0.7
    }

    return features