import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


def create_sine_wave(amplitude, frequency, num_samples, sample_rate, datetime):
    '''Create a sine wave that starts and ends at zero amplitude.
    
    amplitude : Peak amplitude of the sine wave (default: 1)
    frequency : Frequency in Hz (default: 1)
    num_samples : Number of data points/rows (default: 1000)
    sample_rate : Number of samples per second (default: 100)'''
    
    # Calculate duration from number of samples
    duration = num_samples / sample_rate
    period = 1 / frequency
    cycles = duration / period
    desired_cycles = round(cycles, 0)
    duration = desired_cycles * period
    # Create time array - use endpoint=False to exclude the last point
    # This ensures the wave ends at zero and doesn't overlap with next cycle
    t = np.linspace(0, duration, num_samples, endpoint=False) 
    # Create sine wave
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * t)
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': t,
        'amplitude': sine_wave,
        'frequency': frequency,
        'datetime': datetime
    })
    
    return df, sine_wave

def get_data(site, resample_interval, sample_rate, hertz):
    raw = pd.read_csv(f"data/raw_hydrological_data/{site}_raw_data.csv", header=0, parse_dates=[0], names=["datetime", "data"])
    raw["data"] = pd.to_numeric(raw["data"], errors='coerce')
    raw = raw[(raw["datetime"] >= "2024-10-01") & (raw["datetime"] <= "2025-10-01")]
    
    raw = raw.set_index('datetime').resample(resample_interval).mean()
    #raw = raw.set_index('datetime').resample('6h').mean()
    raw['data_log'] = raw['data'].copy()
    raw["data_log"] = np.log1p(raw["data_log"]) # log transform handles zero and negative

    raw["data_offset"] = raw["data"]

    
    offset = hertz - raw["data_offset"].mean()
    raw["data_offset"] = raw["data_offset"] + offset

    raw["data_offset"] = raw["data_offset"] ** 2 # squared
   
    offset = hertz - raw["data_offset"].mean()
    raw["data_offset"] = raw["data_offset"] + offset
    raw["data_offset"] = raw["data_offset"].round(0)
    
    

    all_dfs = []

    #raw = raw[325:330]
    for index, row in raw.iterrows():
        #sample_rate = 600 dont need to define it is set in function call
        num_samples = 200
        
        df, sine_wave = create_sine_wave(
            amplitude=row['data_log'], 
            #amplitude=row["data"], 
            frequency=row["data_offset"], 
            num_samples=num_samples, 
            sample_rate=sample_rate,
            datetime = index
        )
        all_dfs.append(df)

    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Replace all zeros with NaN
    combined_df["amplitude"] = combined_df['amplitude'].replace(0, np.nan)
    combined_df["amplitude"] = combined_df["amplitude"].interpolate(method='linear', limit_direction='both')
    combined_df["amplitude"] = combined_df["amplitude"] / np.max(np.abs(combined_df["amplitude"]))
    #print(combined_df)
    import matplotlib.pyplot as plt
    #print(combined_df)
    sine_wave = combined_df["amplitude"].values
    # If that's still too quiet, amplify BEFORE converting:
    amplification = 5  # Try 1.5x, 2x, etc.
    audio_data = np.int16(sine_wave / np.max(np.abs(sine_wave)) * 32767) #32767# higher sample rate will speed it up 32767
    wavfile.write(f"data/sound_files/{site}_soundfile_resample interval {resample_interval} sample rate_{sample_rate}_hertz_{hertz}.wav", sample_rate, audio_data)
    #save as mp3
    # First write as WAV
    temp_wav = "temp.wav"
    wavfile.write(temp_wav, sample_rate, audio_data)

        # Convert to MP3
    # Write temp WAV first
    temp_wav = "temp.wav"
    wavfile.write(temp_wav, sample_rate, audio_data)

    # Convert to MP3
    from pydub import AudioSegment
    import imageio_ffmpeg as ffmpeg
   

    # Set ffmpeg path
    AudioSegment.converter = ffmpeg.get_ffmpeg_exe()

    # Convert your numpy audio_data to AudioSegment
    audio_segment = AudioSegment(
        audio_data.tobytes(),
        frame_rate=sample_rate,
        sample_width=audio_data.dtype.itemsize,
        channels=1  # use 2 if stereo
    )

    # Export as MP3
    audio_segment.export(
        f"data/sound_files/{site}_soundfile_resample_interval_{resample_interval}_sample_rate_{sample_rate}_hertz_{hertz}.mp3",
        format="mp3"
    )

    
                
    combined_df['mean_aplitude'] = combined_df.groupby('datetime')['amplitude'].transform('mean')


    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot mean_amplitude on the first y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Index', fontsize=12)
    ax1.set_ylabel('Mean Amplitude', color=color, fontsize=12)
    ax1.plot(combined_df.index, combined_df['amplitude'], color=color, linewidth=2, label='Mean Amplitude')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Create second y-axis sharing the same x-axis
    ax2 = ax1.twinx()

    # Plot frequency on the second y-axis
    color = 'tab:orange'
    ax2.set_ylabel('Frequency', color=color, fontsize=12)
    ax2.plot(combined_df.index, combined_df["frequency"], color=color, linewidth=2, label='Frequency')
    ax2.tick_params(axis='y', labelcolor=color)

    # Add title
    plt.title('Mean Amplitude and Frequency vs Index', fontsize=14, fontweight='bold')

    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    #lt.show()
 
    plt.savefig(f'data/figures/{site}_mean_amplitude_interval_{resample_interval}_sample rate_{sample_rate}_hertz_{hertz}.png')
    plt.plot(raw.index, raw["data_offset"])

#58a, 02a, 11u_solar_radiation  data\raw_hydrological_data\11u_solar_radiation_raw_data.csv
#"11u_solar_radiation"
site = "02a" #"11u_solar_radiation_day" # f"data/raw_hydrological_data/{site}_raw_data.csv"
resample_interval = '3D' #'15T' # '1D' '1H'

sample_rate = 800 # higher sample rate will speed it up

# convert to frequency 
#hertz = 261.625565
# d4:293.6648
# c4: 261.625565
# b3: 246.9417
# a3: 220.0000
# c3 130.81
get_data(site = site, resample_interval = resample_interval, sample_rate = sample_rate, hertz = 261.625565)