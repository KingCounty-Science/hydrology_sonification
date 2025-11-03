import subprocess

try:
    result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
    print("✓ ffmpeg is accessible!")
    print(result.stdout[:100])
except FileNotFoundError:
    print("✗ ffmpeg not found in PATH")
    print("Use the full path instead")