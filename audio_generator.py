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
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
# looks at row to row change
# more pronounced/destinct sound aka "heart-beat" is a rapid increase such as the arrival of a peak
# how loud the sound get is how big the peak is
# might add some filtering as baseflow is kinda noisy every little change creates noise and we are not too concerned about little changes
# Example DataFrame with datetime and data columns
df = pd.read_csv("data/raw_hydrological_data/58a_raw_data.csv", header=0, parse_dates=[0], names=["datetime", "data"])
df["data"] = pd.to_numeric(df["data"], errors='coerce')
q_95 = df['data'].quantile(.95)

# filter by like just this year
df = df[(df["datetime"] >= "2024-10-01") & (df["datetime"] <= "2025-10-01")]

# Resample or interpolate the data to a fixed sampling rate
# Here, we'll resample the data to 44100 Hz (typical for audio)
# resampe 1S is seconds 5T is minutes
#df = df.set_index('datetime').resample('10s').ffill() # higher resample rate will lower the pitch and make audio file longer

# 5 minutes , resample to a day, calculate difference, then downsample back to 5 minutes
# inital downsampling (ie 1 minute) will create less record by record change

# 
#df["data"] = 2 * (df['data'] - df['data'].min()) / (df['data'].max() - df['data'].min()) - 1 # -1 to q
df = df.set_index('datetime').resample('5T').interpolate(method='linear', limit=6)  # Only interpolate up to 6 consecutive NaN values

#df = df.asfreq('H')

#df = df.set_index('datetime').resample('300s').interpolate(method='linear', limit=6) 
# 1 day centered rolling average this is like smoothing
#df['1h_data'] = df['data'].rolling(window='1h', center=True).mean()
df['1h_data'] = df['data'].rolling(window='30T', center=True).mean()
#df = df.resample('1D').mean()
#print("upsampled")
#print(df)
### down sample to 10 seconds
# set min to zero

# this is nice
#df['diff'] = df['1h_data'].diff()
df['diff'] = df['1h_data'].diff()

#start_date = df.index.min()
#end_date = df.index.max()
#complete_10s_index = pd.date_range(start=start_date, end=end_date, freq='10s')
    
# Step 3: Reindex to 10-second intervals and interpolate
#df = df.reindex(complete_10s_index)#.interpolate(method= 'time') # method = 'time' #
#df['diff'] = df['data'] - q_95
# Step 2: Create a complete 10-second time index for interpolation

df = df.dropna()
df['off_mean'] = df['data'] - df['data'].mean()
#print(df.head(100))
## try two channels
#df["chan1"] = df['data'].rolling(window='4D', center=True).mean()  # 1D
plt.figure(figsize=(12, 6))
#plt.plot(df.index, df["data"], label="data")
plt.plot(df.index, df["data"], label="data")
plt.plot(df.index, df["1h_data"], label="1d data")
plt.plot(df.index, df["off_mean"], label="off mean")
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


#normalized_data = 2 * (df['data'] - df['data'].min()) / (df['data'].max() - df['data'].min()) - 1 # -1 to q
#normalized_data = normalized_data.dropna()
#df['diff'] = 2 * (df['diff'] - df['diff'].min()) / (df['diff'].max() - df['diff'].min()) - 1 # -1 to q
normalized_data = 2 * (df['diff'] - df['diff'].min()) / (df['diff'].max() - df['diff'].min()) - 1 # -1 to q
normalized_data = normalized_data - normalized_data.mean() # centers around zero

normalized_data = normalized_data.where(normalized_data.abs() >= 0.03, 0)
#from scipy.stats import zscore
#normalized_data = zscore(df['diff'])
#print(normalized_data)
#normalized_neg = 2 * (df['neg'] - df['neg'].min()) / (df['neg'].max() - df['neg'].min()) - 1 # -1 to q
### pos and neg normalization
#plt.clf()  # Clear the current figure
plt.plot(normalized_data.index, normalized_data.values)
#plt.plot(normalized_neg.index, normalized_neg.values)
plt.savefig('normalized.png')

#plt.plot(normalized_data['datetime'], normalized_data['data'])
#plt.show()
# not sure how well this works
#smoothed_normalized = savgol_filter(normalized_data, window_length=11, polyorder=3)
#

# Scale the data to a range suitable for audio (e.g., between -32768 and 32767 for 16-bit audio)
scaled_data = (normalized_data * 32767).astype(np.int16)


#scaled_data = (normalized_data * 20000).astype(np.int16)
#scaled_data = (normalized_data * 32767 * 0.8).astype(np.int16)  # 80% of max range

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
#write("data/sound_files/58a_audio_raw.wav", 44100, scaled_data.values)

#write("data/sound_files/58a_audio_raw.wav", 22050, scaled_data.values) # lowered sample rate will slow it down and lower pitch
write("data/sound_files/58a_audio_raw.wav",  16537, scaled_data.values) 
print("write")
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