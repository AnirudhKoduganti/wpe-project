import os
import json
import numpy as np
import pyroomacoustics as pra
import soundfile as sf
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt

# ---- Reproducibility ----
SEED = 42
np.random.seed(SEED)

# ---- Paths ----
CLEAN_SPEECH_PATH = 'data/clean/dev/5808-48608-0017.flac'  # pick one of your 18 dev files
RIR_DIR = 'data/rir_simulated'
REVERB_DIR = 'data/reverberant'
FIG_DIR = 'results/figures'
AUDIO_DIR = 'results/audio'
os.makedirs(RIR_DIR, exist_ok=True)
os.makedirs(REVERB_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

# ---- Room settings (from the guide's recommended first room) ----
room_dim = [5.0, 4.0, 2.8]      # meters
fs = 16000
rt60_target = 0.4               # seconds
source_distance = 1.0           # meters
mic_spacing = 0.08              # meters (8 cm)

# ---- Convert RT60 -> wall absorption + image order via Sabine's formula ----
e_absorption, max_order = pra.inverse_sabine(rt60_target, room_dim)

room = pra.ShoeBox(
    room_dim,
    fs=fs,
    materials=pra.Material(e_absorption),
    max_order=max_order,
)

# ---- Place microphone array at room center ----
room_center = np.array(room_dim) / 2
d = mic_spacing / 2
mic_positions = np.array([
    [room_center[0]-d, room_center[1]-d, room_center[2]],
    [room_center[0]+d, room_center[1]-d, room_center[2]],
    [room_center[0]-d, room_center[1]+d, room_center[2]],
    [room_center[0]+d, room_center[1]+d, room_center[2]],
]).T  # pyroomacoustics wants shape (3, n_mics)

mic_array = pra.MicrophoneArray(mic_positions, fs)
room.add_microphone_array(mic_array)

# ---- Place source 1 m from array center ----
source_pos = room_center + np.array([source_distance, 0, 0])
room.add_source(source_pos.tolist())

# ---- Compute RIRs (one per microphone) ----
room.compute_rir()

# ---- Verify achieved RT60 (Sabine's formula is approximate) ----
achieved_rt60 = room.measure_rt60()
print("Target RT60:", rt60_target)
print("Achieved RT60 per mic:", achieved_rt60.flatten())

# ---- Load clean speech ----
clean, sr = sf.read(CLEAN_SPEECH_PATH)
assert sr == fs, f"Clean speech sample rate {sr} != room fs {fs}"

# ---- Convolve clean speech with each mic's RIR ----
n_mics = mic_positions.shape[1]
reverberant_channels = []
for m in range(n_mics):
    rir = room.rir[m][0]  # RIR from source 0 to mic m
    reverb = fftconvolve(clean, rir, mode='full')
    reverberant_channels.append(reverb)

# Trim/pad to equal length across channels
max_len = max(len(ch) for ch in reverberant_channels)
reverberant = np.zeros((n_mics, max_len))
for m, ch in enumerate(reverberant_channels):
    reverberant[m, :len(ch)] = ch

# Normalize to avoid clipping (peak normalize)
peak = np.max(np.abs(reverberant))
if peak > 0:
    reverberant = reverberant / peak * 0.95

# ---- Save reverberant multichannel signal ----
reverb_path = os.path.join(REVERB_DIR, 'room01_utt01_reverberant.wav')
sf.write(reverb_path, reverberant.T, fs)  # soundfile wants (samples, channels)

# ---- Save one channel for easy listening ----
sf.write(os.path.join(AUDIO_DIR, 'room01_utt01_reverberant_ch0.wav'), reverberant[0], fs)
sf.write(os.path.join(AUDIO_DIR, 'room01_utt01_clean.wav'), clean, fs)

# ---- Save RIR (channel 0) as audio + all channels as npy ----
# Pad RIRs to equal length before stacking (mic RIRs can differ slightly in length)
rir_list = [room.rir[m][0] for m in range(n_mics)]
max_rir_len = max(len(r) for r in rir_list)
rir_array = np.zeros((n_mics, max_rir_len))
for m, r in enumerate(rir_list):
    rir_array[m, :len(r)] = r

np.save(os.path.join(RIR_DIR, 'room01_rir.npy'), rir_array)
sf.write(os.path.join(AUDIO_DIR, 'room01_rir_ch0.wav'), room.rir[0][0] / np.max(np.abs(room.rir[0][0])), fs)

# ---- Save room metadata + seed ----
metadata = {
    'seed': SEED,
    'room_dim_m': room_dim,
    'sampling_rate': fs,
    'rt60_target_s': rt60_target,
    'rt60_achieved_s': achieved_rt60.flatten().tolist(),
    'source_distance_m': source_distance,
    'source_position_m': source_pos.tolist(),
    'mic_array_center_m': room_center.tolist(),
    'mic_spacing_m': mic_spacing,
    'mic_positions_m': mic_positions.T.tolist(),
    'n_mics': n_mics,
    'max_image_order': max_order,
    'wall_absorption': e_absorption,
    'clean_speech_source': CLEAN_SPEECH_PATH,
}
with open(os.path.join(RIR_DIR, 'room01_metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

print("Saved metadata:", metadata)

# ==========================================================
# INSPECTION PLOTS (required by the checkpoint)
# ==========================================================

t_clean = np.arange(len(clean)) / fs
t_reverb = np.arange(reverberant.shape[1]) / fs
t_rir = np.arange(len(room.rir[0][0])) / fs

# 1. Clean speech waveform
plt.figure(figsize=(10, 3))
plt.plot(t_clean, clean)
plt.title('Clean Speech Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'checkpoint2_clean_waveform.png'), dpi=150)
plt.close()

# 2. RIR waveform (channel 0) -- should show direct peak then decay
plt.figure(figsize=(10, 3))
plt.plot(t_rir, room.rir[0][0])
plt.title('Room Impulse Response (Mic 0)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'checkpoint2_rir_waveform.png'), dpi=150)
plt.close()

# 3. Reverberant waveform (channel 0)
plt.figure(figsize=(10, 3))
plt.plot(t_reverb, reverberant[0])
plt.title('Reverberant Speech Waveform (Mic 0)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'checkpoint2_reverberant_waveform.png'), dpi=150)
plt.close()

# 4. Spectrograms: clean vs reverberant
fig, axes = plt.subplots(2, 1, figsize=(10, 6))
axes[0].specgram(clean, Fs=fs, NFFT=512, noverlap=256, cmap='magma')
axes[0].set_title('Clean Speech Spectrogram')
axes[0].set_ylabel('Frequency (Hz)')

axes[1].specgram(reverberant[0], Fs=fs, NFFT=512, noverlap=256, cmap='magma')
axes[1].set_title('Reverberant Speech Spectrogram (Mic 0)')
axes[1].set_ylabel('Frequency (Hz)')
axes[1].set_xlabel('Time (s)')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'checkpoint2_spectrograms.png'), dpi=150)
plt.close()

print("All plots saved to", FIG_DIR)
print("Checkpoint 2 pipeline complete.")