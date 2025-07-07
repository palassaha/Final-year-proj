import pyaudio
import wave
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def record_audio(filename="sustained_vowel.wav", duration=5, sample_rate=44100, channels=1):
    """
    Record audio from the microphone and save it as a WAV file.
    
    Parameters:
        filename (str): Name of the output file
        duration (int): Recording duration in seconds
        sample_rate (int): Sampling rate in Hz
        channels (int): Number of audio channels (1 for mono, 2 for stereo)
    """
    # Initialize PyAudio
    audio = pyaudio.PyAudio()
    
    # Set recording parameters
    format = pyaudio.paInt16  # 16-bit resolution
    chunk = 1024  # Record in chunks of 1024 samples
    
    print("=== Sustained Vowel Recording for Parkinson's Voice Analysis ===")
    print("Prepare to say 'ahhh' in a steady tone for", duration, "seconds")
    print("Recording will start in:")
    
    # Countdown
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    
    print("Recording NOW! Say 'ahhh'...")
    
    # Open recording stream
    stream = audio.open(format=format,
                       channels=channels,
                       rate=sample_rate,
                       input=True,
                       frames_per_buffer=chunk)
    
    frames = []
    
    # Record for the specified duration
    for _ in range(0, int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)
    
    print("Recording complete!")
    
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    # Create timestamp for unique filename if not specified
    if filename == "sustained_vowel.wav":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sustained_vowel_{timestamp}.wav"
    
    # Save the recorded data as a WAV file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(format))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
    
    print(f"Audio saved to {filename}")
    return filename

def visualize_audio(filename):
    """
    Visualize the recorded audio waveform and spectrogram.
    """
    # Open the WAV file
    with wave.open(filename, 'rb') as wf:
        # Get basic info
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        n_channels = wf.getnchannels()
        
        # Read all frames
        signal = wf.readframes(n_frames)
        signal = np.frombuffer(signal, dtype=np.int16)
        
        # For stereo audio, just take one channel
        if n_channels == 2:
            signal = signal[::2]
        
        # Convert to seconds
        duration = n_frames / sample_rate
        time_axis = np.linspace(0, duration, num=len(signal))
        
    # Create a figure with two subplots
    plt.figure(figsize=(12, 8))
    
    # Plot the waveform
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, signal)
    plt.title("Waveform")
    plt.ylabel("Amplitude")
    plt.xlabel("Time (s)")
    plt.grid(True)
    
    # Plot the spectrogram
    plt.subplot(2, 1, 2)
    plt.specgram(signal, Fs=sample_rate, cmap='viridis')
    plt.title("Spectrogram")
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    
    plt.tight_layout()
    
    # Save the figure
    plot_filename = os.path.splitext(filename)[0] + "_analysis.png"
    plt.savefig(plot_filename)
    plt.close()
    
    print(f"Audio visualization saved to {plot_filename}")
    return plot_filename

def record_and_visualize(duration=5):
    """Record audio and then visualize it."""
    filename = record_audio(duration=duration)
    visualize_audio(filename)
    print("\nRecording complete! You can now use this file with the Parkinson's feature extraction code.")
    print("Recommended next step: Run your feature extraction on:", filename)
    return filename

if __name__ == "__main__":
    # Allow user to specify duration
    try:
        user_duration = input("Enter recording duration in seconds (default: 5): ")
        if user_duration.strip():
            duration = float(user_duration)
        else:
            duration = 5
    except ValueError:
        print("Invalid input. Using default 5 seconds.")
        duration = 5
    
    record_and_visualize(duration)