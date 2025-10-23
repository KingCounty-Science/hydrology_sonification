import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.io import wavfile
def plot_mp3_to_timeseries(file_path, start_time=None):
    """Modified version of your function that returns a pandas DataFrame"""
    # Load the MP3 file
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    
    # If audio data is less than zero make it equal zero
    audio_data[audio_data < 0] = 0
    print(audio_data)
    # Create time axis
    duration = len(audio_data) / sample_rate
    time_seconds = np.linspace(0, duration, len(audio_data))
    wavfile.write("data/sound_files/modified test.wav", sample_rate, audio_data)
    # Plot waveform (same as before)
    plt.figure(figsize=(12, 6))
    plt.plot(time_seconds, audio_data)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title("MP3 File Waveform (Librosa)")
    plt.grid(True)
    plt.savefig('mp3_waveform_librosa.png', dpi=300, bbox_inches='tight')
    #plt.show()
    
    # Convert to pandas DataFrame with datetime index
    if start_time is None:
        start_time = pd.Timestamp.now()
    
    # Create proper time index
    time_deltas = pd.to_timedelta(time_seconds, unit='s')
    datetime_index = start_time + time_deltas
    
    # Create DataFrame
    df = pd.DataFrame({
        'amplitude': audio_data
    }, index=datetime_index)
    print(df)
    return df, audio_data, sample_rate, time_seconds
file_path = "data/sound_files/06 Savana Dance.mp3"
plot_mp3_to_timeseries(file_path, start_time=None)