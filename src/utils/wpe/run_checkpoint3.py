import os
import json
import time
import warnings
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from nara_wpe.wpe import wpe
from nara_wpe.utils import stft, istft

# ==========================================================
# CONFIG (exactly as specified by the guide for Checkpoint 3)
# ==========================================================
CONFIG = {
    'prediction_delay_frames': 3,
    'filter_length_taps': 20,
    'iterations': 3,
    'stft_hop_ms': 8,
    'stft_size': 512,
    'stft_shift': 128,       # 128 samples / 16kHz = 8ms
    'processing_mode': 'offline',
    'sampling_rate': 16000,
    'input_file': 'data/reverberant/room01_utt01_reverberant.wav',
}

CONFIG_DIR = 'configs'
AUDIO_IN_DIR = 'results/audio'
AUDIO_OUT_DIR = 'results/audio'
FIG_DIR = 'results/figures'
LOG_DIR = 'results/tables'
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Save the config file (required artifact #1)
config_path = os.path.join(CONFIG_DIR, 'checkpoint3_wpe_config.json')
with open(config_path, 'w') as f:
    json.dump(CONFIG, f, indent=2)
print(f"Saved config to {config_path}")

# Log the exact command used (required artifact: "the command or notebook used")
log_path = os.path.join(LOG_DIR, 'checkpoint3_run_log.txt')
log_lines = []
log_lines.append("Command used: python src/wpe/run_checkpoint3.py")
log_lines.append(f"Config: {json.dumps(CONFIG)}")


def run_wpe_on_channels(y_multichannel, label):
    """
    y_multichannel: shape (n_channels, n_samples)
    label: 'single_mic' or 'four_mic', used for filenames
    Returns dict of results including runtime, warnings, output signal, checks.
    """
    stft_options = dict(size=CONFIG['stft_size'], shift=CONFIG['stft_shift'])

    # STFT: (n_channels, n_samples) -> (n_channels, n_frames, n_freq) -> (n_freq, n_channels, n_frames)
    Y = stft(y_multichannel, **stft_options).transpose(2, 0, 1)

    caught_warnings = []
    start_time = time.perf_counter()
    error_message = None
    Z = None

    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Z = wpe(
                Y,
                taps=CONFIG['filter_length_taps'],
                delay=CONFIG['prediction_delay_frames'],
                iterations=CONFIG['iterations'],
            )
            caught_warnings = [str(warning.message) for warning in w]
    except Exception as e:
        error_message = str(e)

    runtime_s = time.perf_counter() - start_time

    result = {
        'label': label,
        'runtime_s': runtime_s,
        'warnings': caught_warnings,
        'error': error_message,
    }

    if error_message is not None:
        print(f"[{label}] ERROR during WPE: {error_message}")
        return result, None

    # Back to time domain: (n_freq, n_channels, n_frames) -> (n_channels, n_frames, n_freq)
    z = istft(Z.transpose(1, 2, 0), size=stft_options['size'], shift=stft_options['shift'])

    # ---- Numerical/signal health checks (required for pass condition) ----
    has_nan = bool(np.isnan(z).any())
    has_inf = bool(np.isinf(z).any())
    peak_abs = float(np.max(np.abs(z)))
    is_clipping = peak_abs >= 0.999
    rms = float(np.sqrt(np.mean(z ** 2)))
    is_silent = rms < 1e-5

    result.update({
        'has_nan': has_nan,
        'has_inf': has_inf,
        'peak_abs': peak_abs,
        'is_clipping': is_clipping,
        'rms': rms,
        'is_silent': is_silent,
        'output_shape': z.shape,
    })

    print(f"[{label}] Runtime: {runtime_s:.3f}s | NaN: {has_nan} | Inf: {has_inf} | "
          f"Peak: {peak_abs:.4f} | Clipping: {is_clipping} | RMS: {rms:.6f} | Silent: {is_silent}")
    if caught_warnings:
        print(f"[{label}] Warnings: {caught_warnings}")

    return result, z


# ==========================================================
# LOAD INPUT (required artifact #2: "the input signal")
# ==========================================================
y_full, sr = sf.read(CONFIG['input_file'])
assert sr == CONFIG['sampling_rate'], f"Sample rate mismatch: {sr} vs {CONFIG['sampling_rate']}"
y_full = y_full.T  # soundfile gives (samples, channels) -> we want (channels, samples)
n_channels_available = y_full.shape[0]
print(f"Loaded input: {CONFIG['input_file']}, shape {y_full.shape}, sr {sr}")

# Save a copy of the exact input used, for the record
sf.write(os.path.join(AUDIO_IN_DIR, 'checkpoint3_input_used.wav'), y_full.T, sr)

all_results = {}

# ==========================================================
# RUN 1: single microphone (channel 0 only)
# ==========================================================
print("\n--- Running WPE: single microphone ---")
y_single = y_full[0:1, :]  # shape (1, n_samples)
result_single, z_single = run_wpe_on_channels(y_single, 'single_mic')
all_results['single_mic'] = result_single

single_ok = z_single is not None and not result_single['has_nan'] and not result_single['has_inf'] \
            and not result_single['is_clipping'] and not result_single['is_silent']

if z_single is not None:
    out_path_single = os.path.join(AUDIO_OUT_DIR, 'checkpoint3_wpe_output_single_mic.wav')
    sf.write(out_path_single, z_single[0], sr)
    print(f"Saved single-mic WPE output to {out_path_single}")

# ==========================================================
# RUN 2: four microphones (only if single-mic worked, per the guide)
# ==========================================================
z_four = None
if single_ok:
    print("\n--- Single-mic passed. Running WPE: four microphones ---")
    result_four, z_four = run_wpe_on_channels(y_full, 'four_mic')
    all_results['four_mic'] = result_four

    if z_four is not None:
        out_path_four = os.path.join(AUDIO_OUT_DIR, 'checkpoint3_wpe_output_four_mic.wav')
        sf.write(out_path_four, z_four[0], sr)  # save reference channel for listening
        # also save full 4-channel output
        sf.write(os.path.join(AUDIO_OUT_DIR, 'checkpoint3_wpe_output_four_mic_allch.wav'), z_four.T, sr)
        print(f"Saved four-mic WPE output to {out_path_four}")
else:
    print("\nSingle-mic run failed pass checks — skipping four-mic run per the guide's instructions.")

# ==========================================================
# SAVE RUNTIME + WARNINGS/ERRORS LOG (required artifacts)
# ==========================================================
with open(os.path.join(LOG_DIR, 'checkpoint3_results.json'), 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

log_lines.append(f"Results: {json.dumps(all_results, default=str)}")
with open(log_path, 'w') as f:
    f.write('\n'.join(log_lines))

# ==========================================================
# INSPECTION: waveform + spectrogram comparisons
# ==========================================================

def plot_waveform_compare(input_sig, output_sig, sr, title_suffix, filename):
    t_in = np.arange(len(input_sig)) / sr
    t_out = np.arange(len(output_sig)) / sr
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharey=True)
    axes[0].plot(t_in, input_sig)
    axes[0].set_title(f'Input (Reverberant) Waveform {title_suffix}')
    axes[0].set_ylabel('Amplitude')
    axes[1].plot(t_out, output_sig)
    axes[1].set_title(f'WPE Output Waveform {title_suffix}')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=150)
    plt.close()


def plot_spectrogram_compare(input_sig, output_sig, sr, title_suffix, filename):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].specgram(input_sig, Fs=sr, NFFT=512, noverlap=256, cmap='magma')
    axes[0].set_title(f'Input (Reverberant) Spectrogram {title_suffix}')
    axes[0].set_ylabel('Frequency (Hz)')
    axes[1].specgram(output_sig, Fs=sr, NFFT=512, noverlap=256, cmap='magma')
    axes[1].set_title(f'WPE Output Spectrogram {title_suffix}')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=150)
    plt.close()


if z_single is not None:
    plot_waveform_compare(y_single[0], z_single[0], sr, '(Single Mic)',
                           'checkpoint3_waveform_single_mic.png')
    plot_spectrogram_compare(y_single[0], z_single[0], sr, '(Single Mic)',
                              'checkpoint3_spectrogram_single_mic.png')

if z_four is not None:
    plot_waveform_compare(y_full[0], z_four[0], sr, '(Four Mic, ref ch 0)',
                           'checkpoint3_waveform_four_mic.png')
    plot_spectrogram_compare(y_full[0], z_four[0], sr, '(Four Mic, ref ch 0)',
                              'checkpoint3_spectrogram_four_mic.png')

print("\nCheckpoint 3 pipeline complete. Check results/figures and results/audio.")