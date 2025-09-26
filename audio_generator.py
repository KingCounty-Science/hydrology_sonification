import pandas as pd
import numpy as np
from scipy.io.wavfile import write
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
# Example DataFrame with datetime and data columns
df = pd.read_csv("data/raw_hydrological_data/58a_raw_data.csv", header=0, parse_dates=[0], names=["datetime", "data"])
df["data"] = pd.to_numeric(df["data"], errors='coerce')

# filter by like just this year
df = df[(df["datetime"] >= "2024-10-01") & (df["datetime"] <= "2025-10-01")]

# Resample or interpolate the data to a fixed sampling rate
# Here, we'll resample the data to 44100 Hz (typical for audio)
# resampe 1S is seconds 5T is minutes
#df = df.set_index('datetime').resample('10s').ffill() # higher resample rate will lower the pitch and make audio file longer
df = df.set_index('datetime').resample('10s').interpolate(method='linear', limit=6)  # Only interpolate up to 6 consecutive NaN values
#df = df.set_index('datetime').resample('300s').interpolate(method='linear', limit=6) 
# 1 day centered rolling average meh
#df['data'] = df['data'].rolling(window='300s', center=True).mean()  # 1D

#df = df.set_index('datetime').resample('15m').ffill() 


# try two channels
df["chan1"] = df['data'].rolling(window='4D', center=True).mean()  # 1D
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["data"], label="data")
plt.plot(df.index, df["chan1"], label="chan1")
plt.xlabel("DateTime")
plt.ylabel("Values")
plt.title("Data and Chan1 over Time")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('data.png')


# Normalize the data to a range between -1 and 1
#normalized_data = 2 * (df['data'] - df['data'].min()) / (df['data'].max() - df['data'].min()) - 1
#df['data'] = df['chan1']
#normalized_data = (df['data'] - df['data'].min()) / (df['data'].max() - df['data'].min()) # 0-1
normalized_data = 2 * (df['data'] - df['data'].min()) / (df['data'].max() - df['data'].min()) - 1 # -1 to q
normalized_data = normalized_data.dropna()
plt.clf()  # Clear the current figure
plt.plot(normalized_data.index, normalized_data.values)
plt.savefig('normalized.png')

#plt.plot(normalized_data['datetime'], normalized_data['data'])
#plt.show()
# not sure how well this works
#smoothed_normalized = savgol_filter(normalized_data, window_length=11, polyorder=3)
#
# Scale the data to a range suitable for audio (e.g., between -32768 and 32767 for 16-bit audio)
#scaled_data = (normalized_data * 32767).astype(np.int16)

scaled_data = (normalized_data * 32767 * 0.8).astype(np.int16)  # 80% of max range
plt.clf()  # Clear the current figure
plt.plot(scaled_data.index, scaled_data.values)
plt.savefig('scaled_data.png')

#plt.plot(scaled_data.index, scaled_data.values)
#plt.show()
# Change the pitch by resampling with a different rate
# For example, to decrease pitch by a factor of 2, resample at half the rate
#new_sample_rate = 44100 // 3  # Change this value to adjust pitch
#old_index = np.arange(len(normalized_data))
#new_index = np.linspace(0, len(normalized_data) - 1, int(len(normalized_data) * (44100 / new_sample_rate)))
#interpolated_data = interp1d(old_index, normalized_data, kind='linear')(new_index)

# Scale the data to a range suitable for audio (e.g., between -32768 and 32767 for 16-bit audio)
#scaled_data = (interpolated_data * 32767).astype(np.int16)


# Write the audio file
write("data/sound_files/58a_audio_raw.wav", 44100, scaled_data.values)

import librosa
import matplotlib.pyplot as plt
import numpy as np

# Load the WAV file
audio_data, sample_rate = librosa.load("data/sound_files/58a_audio_raw.wav", sr=None)

# Create time axis
time = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))

# Plot waveform
plt.figure(figsize=(12, 6))
plt.plot(time, audio_data)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("WAV File Waveform")
plt.grid(True)
plt.savefig('waveform.png')
plt.show()

# something evil sounding
"""# Load audio file
y, sr = librosa.load("data/sound_files/58a_audio_raw.wav", sr=None)

# Slow down by factor (0.5 = half speed, 2.0 = double speed)
y_slow = librosa.effects.time_stretch(y, rate=0.1)

# Save the result
sf.write("data/sound_files/58a_audio_2.wav", y_slow, sr)"""