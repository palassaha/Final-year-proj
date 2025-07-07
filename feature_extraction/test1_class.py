import parselmouth
import numpy as np
import scipy.stats
import warnings

def extract_parkinsons_features(audio_path, verbose=False):
    """
    Extract Parkinson's disease-related voice features from audio.
    
    Args:
        audio_path (str): Path to the audio file
        verbose (bool): Print debug information
    
    Returns:
        dict: Dictionary of extracted features
    """
    
    if verbose:
        print(f"Processing: {audio_path}")
    
    try:
        snd = parselmouth.Sound(audio_path)
    except Exception as e:
        raise ValueError(f"Could not load audio file: {e}")
    
    if verbose:
        print(f"Audio duration: {snd.duration:.2f}s, Sample rate: {snd.sampling_frequency}Hz")
    
    
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    voiced_frames = pitch_values[pitch_values > 0]
    
    if verbose:
        print(f"Voiced frames: {len(voiced_frames)}/{len(pitch_values)} ({len(voiced_frames)/len(pitch_values)*100:.1f}%)")
    
    if len(voiced_frames) < 10:
        raise ValueError("Insufficient voiced content. Use a 3-5 second sustained vowel recording.")
    
    
    try:
        
        point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
        num_points = parselmouth.praat.call(point_process, "Get number of points")
        
        
        if num_points < 20:
            point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 50, 800)
            num_points = parselmouth.praat.call(point_process, "Get number of points")
            
        if verbose:
            print(f"PointProcess created with {num_points} points")
            
    except Exception as e:
        raise ValueError(f"Failed to create PointProcess: {e}")
    
    if num_points < 10:
        raise ValueError("Too few pitch periods detected for reliable jitter/shimmer analysis")
    
    
    def safe_jitter_call(command, *args, default=0.0):
        """Safe call for jitter commands (PointProcess only)"""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return parselmouth.praat.call(point_process, command, *args)
        except:
            return default
    
    def safe_shimmer_call(command, *args, default=0.0):
        """Safe call for shimmer commands (Sound + PointProcess)"""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return parselmouth.praat.call([snd, point_process], command, *args)
        except:
            return default
    
    def calculate_manual_jitter_abs():
        """Manual calculation of absolute jitter if Praat command fails"""
        try:
            periods = []
            for i in range(2, min(num_points + 1, 500)):
                try:
                    time1 = parselmouth.praat.call(point_process, "Get time from index", i)
                    time2 = parselmouth.praat.call(point_process, "Get time from index", i-1)
                    period = time1 - time2
                    if 0.002 < period < 0.02: 
                        periods.append(period)
                except:
                    continue
            
            if len(periods) > 2:
                periods = np.array(periods)
                return np.mean(np.abs(np.diff(periods)))
            return 0.0
        except:
            return 0.0
    
    
    jitter_abs = safe_jitter_call("Get jitter (absolute)", 0, 0, 0.0001, 0.02, 1.3)
    if jitter_abs == 0.0:
        jitter_abs = calculate_manual_jitter_abs()
        if verbose and jitter_abs > 0:
            print("Used manual jitter calculation")
    
    rap = safe_jitter_call("Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq = safe_jitter_call("Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddp = 3 * rap  
    
    
    shimmer_local = safe_shimmer_call("Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_db = safe_shimmer_call("Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3 = safe_shimmer_call("Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq5 = safe_shimmer_call("Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq = safe_shimmer_call("Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    dda = 3 * apq3 
    
    
    try:
        harmonicity = snd.to_harmonicity_cc()
        hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)
        
        nhr = 1 / (10**(hnr / 10)) if hnr > -10 else 1.0
    except:
        hnr = 0.0
        nhr = 1.0
    
    if len(voiced_frames) > 5:
        try:
        
            hist, _ = np.histogram(voiced_frames, bins=min(20, len(voiced_frames)//2), density=True)
            
            ppe = scipy.stats.entropy(hist + 1e-10)
        except:
            ppe = 0.0
    else:
        ppe = 0.0
    
    
    features = {
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
    
    if verbose:
        print("Feature extraction completed successfully")
        for name, value in features.items():
            print(f"{name}: {value:.6f}")
    
    return features

def validate_features(features):
    """
    Validate extracted features for reasonable ranges
    
    Args:
        features (dict): Extracted features
    
    Returns:
        list: List of validation warnings
    """
    warnings = []
    
    
    if features['MDVP:Jitter(Abs)'] > 0.01:
        warnings.append("High jitter detected (>1%)")
    
    if features['MDVP:Shimmer'] > 0.1:
        warnings.append("High shimmer detected (>10%)")
    
    if features['HNR'] < 10:
        warnings.append("Low HNR detected (<10 dB)")
    
    if features['NHR'] > 0.1:
        warnings.append("High NHR detected (>0.1)")
    
    return warnings

if __name__ == "__main__":
    import os
    
    audio_file = "code\sustained_vowel_20250626_012731.wav"
    
    if os.path.exists(audio_file):
        try:
            print("=== PARKINSON'S VOICE FEATURE EXTRACTION ===")
            features = extract_parkinsons_features(audio_file, verbose=True)
            
            print("\n=== VALIDATION ===")
            warnings = validate_features(features)
            if warnings:
                for warning in warnings:
                    print(f"  {warning}")
            else:
                print(" All features within normal ranges")
                
        except Exception as e:
            print(f" Error: {e}")
    else:
        print(f" Audio file '{audio_file}' not found")