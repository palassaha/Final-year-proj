
import parselmouth
import numpy as np
import scipy.stats

def extract_parkinsons_features(audio_path):
    snd = parselmouth.Sound(audio_path)
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    voiced_frames = pitch_values[pitch_values > 0]

    if len(voiced_frames) < 10:
        raise ValueError("Audio is too short or unvoiced. Please use a 3–5 sec sustained vowel (e.g. 'ah').")

    try:
        point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
    except Exception as e:
        raise ValueError("Failed to create PointProcess: " + str(e))

    def safe_praat_call_jitter(command, *args, default=0.0):
        """For jitter commands that need only PointProcess"""
        try:
            return parselmouth.praat.call(point_process, command, *args)
        except:
            return default

    def safe_praat_call_shimmer(command, *args, default=0.0):
        """For shimmer commands that need both Sound and PointProcess"""
        try:
            return parselmouth.praat.call([snd, point_process], command, *args)
        except:
            return default

    # Jitter - these commands need only the PointProcess object
    jitter_abs = safe_praat_call_jitter("Get jitter (absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rap = safe_praat_call_jitter("Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq = safe_praat_call_jitter("Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddp = 3 * rap

    # Shimmer - these commands need both Sound and PointProcess objects
    shimmer_local = safe_praat_call_shimmer("Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_db = safe_praat_call_shimmer("Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3 = safe_praat_call_shimmer("Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq5 = safe_praat_call_shimmer("Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq = safe_praat_call_shimmer("Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    dda = 3 * apq3

    # HNR
    try:
        harmonicity = snd.to_harmonicity_cc()
        hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
        nhr = 1 / (10**(hnr / 10)) if hnr > 0 else 1
    except:
        hnr = 0.0
        nhr = 1.0

    # PPE
    if len(voiced_frames) > 1:
        try:
            hist = np.histogram(voiced_frames, bins=20, density=True)[0]
            ppe = scipy.stats.entropy(hist + 1e-6)
        except:
            ppe = 0.0
    else:
        ppe = 0.0

    return {
        'MDVP:Jitter(Abs)': jitter_abs,
        'MDVP:RAP': rap,
        'MDVP:PPQ': ppq,
        'Jitter:DDP': ddp,
        'MDVP:Shimmer': shimmer_local,
        'MDVP:Shimmer(dB)': shimmer_db,
        'Shimmer:APQ3': apq3,
        'Shimmer:APQ5': apq5,
        'MDVP:APQ': apq,
        'Shimmer:DDA': dda,
        'NHR': nhr,
        'HNR': hnr,
        'PPE': ppe
    }

if __name__ == "__main__":
    import os
    audio_file = "sustained_vowel_20250626_012819.wav"
    if os.path.exists(audio_file):
        try:
            features = extract_parkinsons_features(audio_file)
            for k, v in features.items():
                print(f"{k}: {v:.6f}")
        except Exception as e:
            print("Error during feature extraction:", e)
    else:
        print("Audio file not found.")