import pandas as pd
import numpy as np
from scipy.io.wavfile import write

from scipy.interpolate import interp1d
# Example DataFrame with datetime and data columns
df = pd.read_csv("W:/STS/hydro/GAUGE/Temp/Ian's Temp/raw_audio_data.csv", header=0, parse_dates=[0], names=["datetime", "data"])

# Resample or interpolate the data to a fixed sampling rate
# Here, we'll resample the data to 44100 Hz (typical for audio)
# resampe 1S is seconds 5T is minutes
df = df.set_index('datetime').resample('10S').ffill() # higher resample rate will lower the pitch and make audio file longer


# Normalize the data to a range between -1 and 1
normalized_data = df['data'] / df['data'].abs().max()


# Scale the data to a range suitable for audio (e.g., between -32768 and 32767 for 16-bit audio)
scaled_data = (normalized_data * 32767).astype(np.int16)

# Change the pitch by resampling with a different rate
# For example, to decrease pitch by a factor of 2, resample at half the rate
#new_sample_rate = 44100 // 3  # Change this value to adjust pitch
#old_index = np.arange(len(normalized_data))
#new_index = np.linspace(0, len(normalized_data) - 1, int(len(normalized_data) * (44100 / new_sample_rate)))
#interpolated_data = interp1d(old_index, normalized_data, kind='linear')(new_index)

# Scale the data to a range suitable for audio (e.g., between -32768 and 32767 for 16-bit audio)
#scaled_data = (interpolated_data * 32767).astype(np.int16)

# Write the audio file
#write("W:/STS/hydro/GAUGE/Temp/Ian's Temp/audio_pitch_adjusted.wav", new_sample_rate, scaled_data)

# Write the audio file
write("W:/STS/hydro/GAUGE/Temp/Ian's Temp/audio.wav", 44100, scaled_data.values)