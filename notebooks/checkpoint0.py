import soundfile as sf

path = 'data/clean/2902-9008-0000.flac'

data, sr = sf.read(path)
print('Sample rate:', sr)
print('Duration (s):', len(data) / sr)
print('Shape:', data.shape)

assert sr == 16000, f"Expected 16000 Hz, got {sr}"
print("PASS: file loads and is 16 kHz")