import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import fftconvolve
from pystoi import stoi
from nara_wpe.wpe import wpe
from nara_wpe.utils import stft, istft

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
simulation_dir = os.path.join(current_dir, '..', 'simulation')
sys.path.append(simulation_dir)

from generate_room import generate_room_rir, convolve_multichannel

# ==========================================================
# FIXED SETTINGS (kept constant across this whole comparison, per the guide)
# ==========================================================
FS = 16000
STFT_SIZE = 512
STFT_SHIFT = 128          # 8 ms hop
ITERATIONS = 3
REFERENCE_CHANNEL = 0
PROCESSING_MODE = 'offline'
EARLY_MS = 50
TAPER_MS = 5
NORM_PEAK = 0.95

# ==========================================================
# THE NINE CONFIGURATIONS (C1-C9, from the guide's table)
# ==========================================================
CONFIGS = [
    {'id': 'C1', 'delay': 2, 'taps': 10},
    {'id': 'C2', 'delay': 2, 'taps': 20},
    {'id': 'C3', 'delay': 2, 'taps': 30},
    {'id': 'C4', 'delay': 3, 'taps': 10},
    {'id': 'C5', 'delay': 3, 'taps': 20},
    {'id': 'C6', 'delay': 3, 'taps': 30},
    {'id': 'C7', 'delay': 4, 'taps': 10},
    {'id': 'C8', 'delay': 4, 'taps': 20},
    {'id': 'C9', 'delay': 4, 'taps': 30},
]

# ==========================================================
# THREE ROOM CONDITIONS (small pilot per Checkpoint 5 — not the full matrix)
# ==========================================================
ROOMS = [
    {'room_id': 'P01_mild_near',   'room_dim': [5.0, 4.0, 2.8], 'rt60_target': 0.2, 'source_distance': 1.0, 'seed': 101},
    {'room_id': 'P02_moderate_near', 'room_dim': [5.0, 4.0, 2.8], 'rt60_target': 0.4, 'source_distance': 1.0, 'seed': 102},
    {'room_id': 'P03_strong_far',  'room_dim': [5.0, 4.0, 2.8], 'rt60_target': 0.6, 'source_distance': 2.5, 'seed': 103},
]

CLEAN_SPEECH_PATH = 'data/clean/dev/5808-48608-0017.flac'
MIC_SPACING = 0.08

RESULTS_DIR = 'results/tables'
CONFIG_DIR = 'configs'
AUDIO_DIR = 'results/audio'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

# Which configs to save audio for, per room, so you have something concrete
# to listen to when filling in listening_notes (worst / mid / best-ish spread)
CONFIGS_TO_SAVE = ['C1', 'C3', 'C7', 'C8', 'C9']

# ==========================================================
# Reused target-construction logic (same rule as Checkpoint 4)
# ==========================================================
def detect_direct_arrival(rir):
    return int(np.argmax(np.abs(rir)))

def build_target_rir(rir, direct_idx, fs, early_ms, taper_ms):
    early_samples = int(early_ms / 1000 * fs)
    taper_samples = int(taper_ms / 1000 * fs)
    end_idx = direct_idx + early_samples
    target_rir = np.zeros_like(rir)
    target_rir[:end_idx] = rir[:end_idx]
    if taper_samples > 0 and end_idx - taper_samples >= 0:
        taper_window = 0.5 * (1 + np.cos(np.linspace(0, np.pi, taper_samples)))
        target_rir[end_idx - taper_samples:end_idx] *= taper_window
    return target_rir

def peak_normalize(sig, peak=NORM_PEAK):
    m = np.max(np.abs(sig))
    return sig / m * peak if m > 0 else sig

def si_sdr(reference, estimate, eps=1e-8):
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)
    alpha = np.dot(estimate, reference) / (np.dot(reference, reference) + eps)
    projection = alpha * reference
    noise = estimate - projection
    ratio = np.sum(projection ** 2) / (np.sum(noise ** 2) + eps)
    return 10 * np.log10(ratio + eps)

def align_crop(a, b):
    min_len = min(len(a), len(b))
    return a[:min_len], b[:min_len]

def run_wpe(y_multichannel, delay, taps, iterations):
    stft_options = dict(size=STFT_SIZE, shift=STFT_SHIFT)
    Y = stft(y_multichannel, **stft_options).transpose(2, 0, 1)
    start = time.perf_counter()
    error = None
    caught = []
    Z = None
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Z = wpe(Y, taps=taps, delay=delay, iterations=iterations)
            caught = [str(x.message) for x in w]
    except Exception as e:
        error = str(e)
    runtime = time.perf_counter() - start
    if error is not None:
        return None, runtime, caught, error
    z = istft(Z.transpose(1, 2, 0), size=STFT_SIZE, shift=STFT_SHIFT)
    return z, runtime, caught, None

# ==========================================================
# MAIN LOOP
# ==========================================================
clean, sr = sf.read(CLEAN_SPEECH_PATH)
assert sr == FS

results = []
room_metadata_log = {}

for room_cfg in ROOMS:
    print(f"\n=== Room: {room_cfg['room_id']} (RT60 target={room_cfg['rt60_target']}, dist={room_cfg['source_distance']}m) ===")
    rir_array, room_meta = generate_room_rir(
        room_cfg['room_dim'], FS, room_cfg['rt60_target'],
        room_cfg['source_distance'], MIC_SPACING, room_cfg['seed']
    )
    room_metadata_log[room_cfg['room_id']] = room_meta
    print(f"Achieved RT60: {room_meta['rt60_achieved_s']}")

    reverberant = convolve_multichannel(clean, rir_array)

    # ---- Save the reverberant baseline audio for this room (needed for listening) ----
    baseline_audio_path = os.path.join(AUDIO_DIR, f"checkpoint5_{room_cfg['room_id']}_baseline.wav")
    sf.write(baseline_audio_path, reverberant[REFERENCE_CHANNEL], FS)

    # Build target (fixed target definition, per predeclaration)
    direct_idx = [detect_direct_arrival(rir_array[m]) for m in range(4)]
    target_rirs = np.array([
        build_target_rir(rir_array[m], direct_idx[m], FS, EARLY_MS, TAPER_MS)
        for m in range(4)
    ])
    target_channels = [fftconvolve(clean, target_rirs[m], mode='full') for m in range(4)]
    max_len = max(len(t) for t in target_channels)
    target = np.zeros((4, max_len))
    for m, t in enumerate(target_channels):
        target[m, :len(t)] = t
    target_ref = target[REFERENCE_CHANNEL]

    # Baseline score for this room (no processing)
    t_al, b_al = align_crop(target_ref, reverberant[REFERENCE_CHANNEL])
    baseline_stoi = stoi(peak_normalize(t_al), peak_normalize(b_al), FS, extended=False)
    baseline_sisdr = si_sdr(peak_normalize(t_al), peak_normalize(b_al))
    results.append({
        'room_id': room_cfg['room_id'], 'config_id': 'baseline', 'delay': None, 'taps': None,
        'stoi': baseline_stoi, 'si_sdr_db': baseline_sisdr, 'runtime_s': None,
        'numerical_failure': False, 'signal_failure': False, 'warnings': '', 'listening_notes': ''
    })
    print(f"  Baseline: STOI={baseline_stoi:.4f}, SI-SDR={baseline_sisdr:.2f}dB")

    for cfg in CONFIGS:
        z, runtime, warns, error = run_wpe(reverberant, cfg['delay'], cfg['taps'], ITERATIONS)

        numerical_failure = error is not None
        signal_failure = False
        row = {
            'room_id': room_cfg['room_id'], 'config_id': cfg['id'],
            'delay': cfg['delay'], 'taps': cfg['taps'],
            'runtime_s': round(runtime, 3),
            'numerical_failure': numerical_failure,
            'warnings': '; '.join(warns) if warns else '',
            'listening_notes': '',  # fill in manually after listening
        }

        if numerical_failure:
            row.update({'stoi': None, 'si_sdr_db': None, 'signal_failure': None})
            print(f"  [{cfg['id']}] NUMERICAL FAILURE: {error}")
        else:
            out_ref = z[REFERENCE_CHANNEL]
            has_nan = bool(np.isnan(out_ref).any())
            has_inf = bool(np.isinf(out_ref).any())
            peak = float(np.max(np.abs(out_ref)))
            rms = float(np.sqrt(np.mean(out_ref ** 2)))
            signal_failure = has_nan or has_inf or peak >= 0.999 or rms < 1e-5

            if signal_failure:
                row.update({'stoi': None, 'si_sdr_db': None, 'signal_failure': True})
                print(f"  [{cfg['id']}] SIGNAL FAILURE (nan={has_nan}, inf={has_inf}, clip={peak>=0.999}, silent={rms<1e-5})")
            else:
                t_al, o_al = align_crop(target_ref, out_ref)
                s = stoi(peak_normalize(t_al), peak_normalize(o_al), FS, extended=False)
                sd = si_sdr(peak_normalize(t_al), peak_normalize(o_al))
                row.update({'stoi': s, 'si_sdr_db': sd, 'signal_failure': False})
                print(f"  [{cfg['id']}] delay={cfg['delay']}, taps={cfg['taps']} -> STOI={s:.4f}, SI-SDR={sd:.2f}dB, runtime={runtime:.2f}s")

                # ---- Save audio for select configs, for the listening-notes step ----
                if cfg['id'] in CONFIGS_TO_SAVE:
                    cfg_audio_path = os.path.join(
                        AUDIO_DIR, f"checkpoint5_{room_cfg['room_id']}_{cfg['id']}.wav"
                    )
                    sf.write(cfg_audio_path, out_ref, FS)

        results.append(row)

# ==========================================================
# SAVE RESULTS TABLE
# ==========================================================
df = pd.DataFrame(results)
df.to_csv(os.path.join(RESULTS_DIR, 'checkpoint5_parameter_study.csv'), index=False)

with open(os.path.join(RESULTS_DIR, 'checkpoint5_room_metadata.json'), 'w') as f:
    json.dump(room_metadata_log, f, indent=2)

fixed_settings = {
    'stft_size': STFT_SIZE, 'stft_shift': STFT_SHIFT, 'iterations': ITERATIONS,
    'reference_channel': REFERENCE_CHANNEL, 'processing_mode': PROCESSING_MODE,
    'normalization_peak': NORM_PEAK, 'target_definition': f'direct + {EARLY_MS}ms, {TAPER_MS}ms taper',
}
with open(os.path.join(CONFIG_DIR, 'checkpoint5_fixed_settings.json'), 'w') as f:
    json.dump(fixed_settings, f, indent=2)

print(f"\nSaved {len(df)} rows to {RESULTS_DIR}/checkpoint5_parameter_study.csv")
print(f"Saved listening audio (baseline + {CONFIGS_TO_SAVE}) per room to {AUDIO_DIR}")
print("Checkpoint 5 pipeline complete.")