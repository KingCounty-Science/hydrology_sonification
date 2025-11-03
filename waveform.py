import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from pydub import AudioSegment
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.io import wavfile
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg
import imageio
import io
import matplotlib.animation as animation
# Using FFMpegWriter with more control
from matplotlib.animation import FFMpegWriter
import imageio_ffmpeg

ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

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
    
    #print(combined_df)
    sine_wave = combined_df["amplitude"].values
    # If that's still too quiet, amplify BEFORE converting:
    amplification = 5  # Try 1.5x, 2x, etc.
    audio_data = np.int16(sine_wave / np.max(np.abs(sine_wave)) * 32767) #32767# higher sample rate will speed it up 32767
    wavfile.write(f"data/sound_files/{site}_soundfile_resample interval {resample_interval} sample rate_{sample_rate}_hertz_{hertz}.wav", sample_rate, audio_data)
    #save as mp3
    # Write temp WAV first
    temp_wav = "temp.wav"
    wavfile.write(temp_wav, sample_rate, audio_data)

    
            
    
    return combined_df

def make_video(site, resample_interval, sample_rate, hertz, combined_df):
    """
    Create an animated video visualization with audio overlay.
    
    Parameters:
    -----------
    site : str
        Site identifier for file naming
    resample_interval : str
        Resample interval used in data processing
    sample_rate : int
        Audio sample rate
    hertz : float
        Base frequency in hertz
    combined_df : pd.DataFrame
        DataFrame containing 'amplitude' and 'frequency' columns with datetime index
    """
    
    # Calculate mean amplitude by datetime
    combined_df['mean_amplitude'] = combined_df.groupby('datetime')['amplitude'].transform('mean')
    
    # Get audio duration to determine video length
    audio_path = f"data/sound_files/{site}_soundfile_resample interval {resample_interval} sample rate_{sample_rate}_hertz_{hertz}.wav"
    
    import wave
    with wave.open(audio_path, 'r') as audio_file:
        frames_audio = audio_file.getnframes()
        rate = audio_file.getframerate()
        audio_duration = frames_audio / float(rate)
    
    print(f"Audio duration: {audio_duration:.2f} seconds")
    
    # Calculate number of video frames needed
    fps = 30
    n_frames = int(audio_duration * fps)
    print(f"Generating {n_frames} frames at {fps} fps")
    
    # Create figure with dimensions divisible by 16 (for video encoding)
    dpi = 100
    width_inches = 1200 / dpi  # = 12 inches = 1200 pixels
    height_inches = 608 / dpi  # = 6.08 inches = 608 pixels
    
    fig, ax1 = plt.subplots(1, 1, figsize=(width_inches, height_inches), dpi=dpi)
    
    # Configure primary axis (amplitude)
    color = 'tab:blue'
    ax1.set_xlabel('Index', fontsize=12)
    ax1.set_ylabel('Mean Amplitude', color=color, fontsize=12)
    line1, = ax1.plot([], [], color=color, linewidth=2, label='Mean Amplitude')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(combined_df.index.min(), combined_df.index.max())
    ax1.set_ylim(combined_df['amplitude'].min(), combined_df['amplitude'].max())
    
    # Configure secondary axis (frequency)
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Frequency', color=color, fontsize=12)
    line2, = ax2.plot([], [], color=color, linewidth=2, label='Frequency')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(combined_df['frequency'].min(), combined_df['frequency'].max())
    
    # Add vertical line marker and time display
    vline = ax1.axvline(x=combined_df.index[0], color='red', linewidth=2, 
                        linestyle='--', label='Current Position')
    # adds a time tracker
    #time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, 
    #                    fontsize=12, verticalalignment='top',
    #                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    def update(frame):
        """Update animation for given frame number"""
        n_points = len(combined_df)
        current_idx = int((frame / n_frames) * n_points)
        current_idx = min(current_idx, n_points - 1)  # Prevent index out of bounds
        
        # Update data lines
        line1.set_data(combined_df.index[:current_idx], combined_df['amplitude'][:current_idx])
        line2.set_data(combined_df.index[:current_idx], combined_df['frequency'][:current_idx])
        
        # Update vertical line position
        if current_idx > 0:
            vline.set_xdata([combined_df.index[current_idx]])
        
        # Update frame info text
        #time_text.set_text(f'Frame: {frame}\nIndex: {current_idx}')
        
        return line1, line2, vline, #time_text
    
    # Generate video frames
    frames = []
    canvas = FigureCanvasAgg(fig)
    
    print("Generating frames...")
    for frame_num in range(n_frames):
        update(frame_num)
        canvas.draw()
        img = np.asarray(canvas.buffer_rgba()).copy()
        frames.append(img[:, :, :3])  # Remove alpha channel
        
        if frame_num % 100 == 0:
            print(f"Frame {frame_num}/{n_frames}")
    
    # Save video without audio
    video_no_audio_path = f"data/figures/{site}_animation_no_audio.mp4"
    print("Saving video...")
    imageio.mimsave(video_no_audio_path, frames, fps=fps)
    
    # Combine video with audio using ffmpeg
    import subprocess
    
    ffmpeg_path = r"C:\Users\ianrh\AppData\Local\Programs\Python\Python312\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe"
    output_path = f"data/figures/{site}_animation_{resample_interval}_sample_rate_{sample_rate}_hertz_{hertz}.mp4"
    
    print(f"Combining video and audio...")
    print(f"  Video: {video_no_audio_path}")
    print(f"  Audio: {audio_path}")
    print(f"  Output: {output_path}")
    
    subprocess.run([
        ffmpeg_path, '-y',
        '-i', video_no_audio_path,
        '-i', audio_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-shortest',
        output_path
    ], check=True, capture_output=True, text=True)
    
    print(f"✓ Video with audio saved to {output_path}")
    
    """ # Optional: Save as GIF
    gif_path = f"data/figures/{site}_animation.gif"
    print("Saving GIF...")
    imageio.mimsave(gif_path, frames, fps=20, loop=0)
    print(f"✓ GIF saved to {gif_path}")"""
    
    # Close figure to free memory
    plt.close(fig)
    
    return output_path
    # Save final frame as static image
    #plt.savefig(f"data/figures/{site}_animation_{resample_interval}_sample_rate_{sample_rate}_hertz_{hertz}.png")
    #58a, 02a, 11u_solar_radiation  data\raw_hydrological_data\11u_solar_radiation_raw_data.csv
#"11u_solar_radiation"
site = "58d" #"58a" #"11u_solar_radiation_day" # f"data/raw_hydrological_data/{site}_raw_data.csv"
resample_interval =  '1D' #'3D'#'15T' # '1D' '1H'
hertz = 130.81
sample_rate = 2000 # higher sample rate will speed it up
# 899 is pretty good

# convert to frequency 
#hertz = 261.625565
# d4:293.6648
# c4: 261.625565
# b3: 246.9417
# a3: 220.0000
# c3 130.81
combined_df = get_data(site, resample_interval, sample_rate, hertz)
make_video(site, resample_interval, sample_rate, hertz, combined_df)