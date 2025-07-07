
import parselmouth
import numpy as np
import matplotlib.pyplot as plt

def diagnose_jitter_calculation(audio_path):
    """
    Diagnostic tool to understand jitter calculation issues
    """
    print("=== JITTER CALCULATION DIAGNOSIS ===")
    
    snd = parselmouth.Sound(audio_path)
    print(f"Audio: {snd.duration:.2f}s, {snd.sampling_frequency}Hz")
    
    
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    voiced_frames = pitch_values[pitch_values > 0]
    
    print(f"Pitch stats: mean={np.mean(voiced_frames):.1f}Hz, std={np.std(voiced_frames):.2f}Hz")


    point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
    num_points = parselmouth.praat.call(point_process, "Get number of points")
    print(f"PointProcess points: {num_points}")
    
    
    jitter_commands = [
        "Get jitter (local)",
        "Get jitter (local, absolute)", 
        "Get jitter (absolute)",
        "Get jitter (rap)",
        "Get jitter (ppq5)"
    ]
    
    print("\n--- Testing Jitter Commands ---")
    for cmd in jitter_commands:
        try:
            result = parselmouth.praat.call(point_process, cmd, 0, 0, 0.0001, 0.02, 1.3)
            print(f"✓ {cmd}: {result:.6f}")
        except Exception as e:
            print(f"✗ {cmd}: {str(e)[:50]}")
    
    
    print("\n--- Manual Calculations ---")

    
    times = []
    for i in range(1, min(num_points + 1, 100)): 
        try:
            time = parselmouth.praat.call(point_process, "Get time from index", i)
            times.append(time)
        except:
            break
    
    times = np.array(times)
    periods = np.diff(times)
    
    print(f"First 10 periods (ms): {periods[:10] * 1000}")
    print(f"Period stats: mean={np.mean(periods)*1000:.3f}ms, std={np.std(periods)*1000:.3f}ms")
    
    
    if len(periods) > 2:
        
        abs_jitter = np.mean(np.abs(np.diff(periods)))
        print(f"Manual absolute jitter: {abs_jitter*1000:.6f}ms = {abs_jitter:.6f}s")
        
        
        local_jitter = (np.mean(np.abs(np.diff(periods))) / np.mean(periods)) * 100
        print(f"Manual local jitter: {local_jitter:.6f}%")
        
        
        expected_range = np.mean(periods) * 0.01  
        print(f"Expected jitter range for 1%: {expected_range*1000:.6f}ms")
        
        
        pitch_periods = 1.0 / voiced_frames
        pitch_jitter = np.mean(np.abs(np.diff(pitch_periods)))
        print(f"Pitch-based jitter: {pitch_jitter*1000:.6f}ms")
    
    return {
        'periods': periods,
        'times': times,
        'voiced_frames': voiced_frames,
        'manual_abs_jitter': abs_jitter if 'abs_jitter' in locals() else 0,
        'manual_local_jitter': local_jitter if 'local_jitter' in locals() else 0
    }

def compare_with_typical_values():
    """Show typical jitter values for reference"""
    print("\n=== TYPICAL JITTER VALUES ===")
    print("Healthy voice:")
    print("  - Absolute jitter: 0.02-0.05 ms (0.00002-0.00005 s)")
    print("  - Local jitter: 0.1-1.0%")
    print("  - RAP: 0.1-0.5%")
    print("  - PPQ: 0.1-0.5%")
    print()
    print("Parkinson's voice:")
    print("  - Absolute jitter: 0.05-0.2+ ms (0.00005-0.0002+ s)")
    print("  - Local jitter: 1.0-5.0+%")
    print("  - RAP: 0.5-2.0+%")
    print("  - PPQ: 0.5-2.0+%")

if __name__ == "__main__":
    audio_file = "sustained_vowel_20250626_012819.wav"
    
    import os
    if os.path.exists(audio_file):
        results = diagnose_jitter_calculation(audio_file)
        compare_with_typical_values()
    else:
        print(f"Audio file '{audio_file}' not found")