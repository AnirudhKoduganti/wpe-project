import os
import sys
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

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..', 'simulation'))
from generate_room import generate_room_rir, convolve_multichannel
from noise import add_noise_to_snr

# ==========================================================
# FIXED SETTINGS (unchanged from Checkpoints 3-5 — kept constant)
# ==========================================================
FS = 16000
STFT_SIZE = 512
STFT_SHIFT = 128
ITERATIONS = 3
REFERENCE_CHANNEL = 0
EARLY_MS = 50
TAPER_MS = 5
NORM_PEAK = 0.95
MIC_SPACING = 0.08
ROOM_DIM = [5.0, 4.0, 2.8]  # Geometry G-A, matches Checkpoints 2-5

CONFIGS = [
    {'id': 'C1', 'delay': 2, 'taps': 10}, {'id': 'C2', 'delay': 2, 'taps': 20}, {'id': 'C3', 'delay': 2, 'taps': 30},
    {'id': 'C4', 'delay': 3, 'taps': 10}, {'id': 'C5', 'delay': 3, 'taps': 20}, {'id': 'C6', 'delay': 3, 'taps': 30},
    {'id': 'C7', 'delay': 4, 'taps': 10}, {'id': 'C8', 'delay': 4, 'taps': 20}, {'id': 'C9', 'delay': 4, 'taps': 30},
]

RT60_TARGETS = [0.2, 0.4, 0.6]
DISTANCES = [1.0, 2.5]
N_REPLICATES = 2
SNR_CONDITIONS = [None, 10]   # None = clean

# ---- SCOPE DECISION (documented): 1 utterance/speaker = 6 total for this run.
# Raise to 3 for the full 18-utterance matrix once this passes and time allows.
UTTERANCES_PER_SPEAKER = 1

MANIFEST_PATH = 'data/manifests/manifest.csv'
RESULTS_CSV = 'results/tables/checkpoint6_development_results.csv'
os.makedirs('results/tables', exist_ok=True)

CSV_COLUMNS = [
    'room_id', 'rt60_target', 'rt60_achieved', 'distance_m', 'replicate', 'seed',
    'snr_condition', 'utterance', 'config_id', 'delay', 'taps',
    'stoi', 'si_sdr_db', 'runtime_s', 'numerical_failure', 'signal_failure', 'warnings'
]

# ==========================================================
# Shared helper functions (same rules as Checkpoints 4-5)
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
        w = 0.5 * (1 + np.cos(np.linspace(0, np.pi, taper_samples)))
        target_rir[end_idx - taper_samples:end_idx] *= w
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
    m = min(len(a), len(b))
    return a[:m], b[:m]

def run_wpe(y, delay, taps, iterations):
    Y = stft(y, size=STFT_SIZE, shift=STFT_SHIFT).transpose(2, 0, 1)
    start = time.perf_counter()
    error, caught, Z = None, [], None
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

def score_and_check(target_ref, out_ref):
    has_nan = bool(np.isnan(out_ref).any())
    has_inf = bool(np.isinf(out_ref).any())
    peak = float(np.max(np.abs(out_ref)))
    rms = float(np.sqrt(np.mean(out_ref ** 2)))
    signal_failure = has_nan or has_inf or peak >= 0.999 or rms < 1e-5
    if signal_failure:
        return None, None, True
    t_al, o_al = align_crop(target_ref, out_ref)
    s = stoi(peak_normalize(t_al), peak_normalize(o_al), FS, extended=False)
    sd = si_sdr(peak_normalize(t_al), peak_normalize(o_al))
    return s, sd, False

# ==========================================================
# RESUME SUPPORT: load already-completed rows, skip them
# ==========================================================
if os.path.exists(RESULTS_CSV):
    existing = pd.read_csv(RESULTS_CSV)
    done_keys = set(zip(existing['room_id'], existing['snr_condition'],
                         existing['utterance'], existing['config_id']))
    print(f"Resuming: {len(existing)} rows already completed, will skip those.")
else:
    pd.DataFrame(columns=CSV_COLUMNS).to_csv(RESULTS_CSV, index=False)
    done_keys = set()

def append_row(row):
    pd.DataFrame([row], columns=CSV_COLUMNS).to_csv(RESULTS_CSV, mode='a', header=False, index=False)

# ==========================================================
# Load development utterances (subset per speaker)
# ==========================================================
manifest = pd.read_csv(MANIFEST_PATH)
dev = manifest[manifest['split'] == 'dev']
utt_rows = dev.groupby('speaker_id').head(UTTERANCES_PER_SPEAKER)
UTTERANCES = list(utt_rows['path'])
print(f"Using {len(UTTERANCES)} development utterances (of 18 available): {UTTERANCES}")

total_runs = len(RT60_TARGETS) * len(DISTANCES) * N_REPLICATES * len(SNR_CONDITIONS) * len(UTTERANCES) * (1 + len(CONFIGS))
print(f"Estimated total scoring rows this run will produce: {total_runs}")

room_metadata_log = {}
seed_base = 200
combo_index = 0

for rt60_target in RT60_TARGETS:
    for distance in DISTANCES:
        for replicate in range(N_REPLICATES):
            combo_index += 1
            seed = seed_base + combo_index
            room_id = f"D_rt{rt60_target}_d{distance}_rep{replicate}"

            rir_array, room_meta = generate_room_rir(ROOM_DIM, FS, rt60_target, distance, MIC_SPACING, seed)
            room_meta['seed'] = seed
            room_metadata_log[room_id] = room_meta
            print(f"\n=== {room_id} (seed={seed}) | achieved RT60: {room_meta['rt60_achieved_s']} ===")

            direct_idx = [detect_direct_arrival(rir_array[m]) for m in range(4)]
            target_rirs = np.array([build_target_rir(rir_array[m], direct_idx[m], FS, EARLY_MS, TAPER_MS) for m in range(4)])

            for snr_db in SNR_CONDITIONS:
                snr_label = 'clean' if snr_db is None else f'{snr_db}dB'

                for utt_path in UTTERANCES:
                    clean, sr = sf.read(utt_path)
                    assert sr == FS
                    utt_id = os.path.splitext(os.path.basename(utt_path))[0]

                    reverberant_clean = convolve_multichannel(clean, rir_array)
                    reverberant = add_noise_to_snr(reverberant_clean, snr_db, seed=seed)

                    target_channels = [fftconvolve(clean, target_rirs[m], mode='full') for m in range(4)]
                    max_len = max(len(t) for t in target_channels)
                    target = np.zeros((4, max_len))
                    for m, t in enumerate(target_channels):
                        target[m, :len(t)] = t
                    target_ref = target[REFERENCE_CHANNEL]

                    # ---- baseline (no processing) ----
                    key = (room_id, snr_label, utt_id, 'baseline')
                    if key not in done_keys:
                        t_al, b_al = align_crop(target_ref, reverberant[REFERENCE_CHANNEL])
                        b_stoi = stoi(peak_normalize(t_al), peak_normalize(b_al), FS, extended=False)
                        b_sisdr = si_sdr(peak_normalize(t_al), peak_normalize(b_al))
                        append_row({
                            'room_id': room_id, 'rt60_target': rt60_target,
                            'rt60_achieved': room_meta['rt60_achieved_s'][REFERENCE_CHANNEL],
                            'distance_m': distance, 'replicate': replicate, 'seed': seed,
                            'snr_condition': snr_label, 'utterance': utt_id, 'config_id': 'baseline',
                            'delay': None, 'taps': None, 'stoi': b_stoi, 'si_sdr_db': b_sisdr,
                            'runtime_s': None, 'numerical_failure': False, 'signal_failure': False, 'warnings': ''
                        })
                        done_keys.add(key)

                    # ---- all nine configs ----
                    for cfg in CONFIGS:
                        key = (room_id, snr_label, utt_id, cfg['id'])
                        if key in done_keys:
                            continue
                        z, runtime, warns, error = run_wpe(reverberant, cfg['delay'], cfg['taps'], ITERATIONS)
                        base_row = {
                            'room_id': room_id, 'rt60_target': rt60_target,
                            'rt60_achieved': room_meta['rt60_achieved_s'][REFERENCE_CHANNEL],
                            'distance_m': distance, 'replicate': replicate, 'seed': seed,
                            'snr_condition': snr_label, 'utterance': utt_id, 'config_id': cfg['id'],
                            'delay': cfg['delay'], 'taps': cfg['taps'], 'runtime_s': round(runtime, 3),
                            'warnings': '; '.join(warns) if warns else '',
                        }
                        if error is not None:
                            base_row.update({'stoi': None, 'si_sdr_db': None,
                                              'numerical_failure': True, 'signal_failure': None})
                        else:
                            s, sd, sig_fail = score_and_check(target_ref, z[REFERENCE_CHANNEL])
                            base_row.update({'stoi': s, 'si_sdr_db': sd,
                                              'numerical_failure': False, 'signal_failure': sig_fail})
                        append_row(base_row)
                        done_keys.add(key)

                print(f"  [{room_id} | {snr_label}] utterance {utt_id}: done")

with open('results/tables/checkpoint6_room_metadata.json', 'w') as f:
    json.dump(room_metadata_log, f, indent=2)

print(f"\nCheckpoint 6 data collection complete. {len(done_keys)} total rows in {RESULTS_CSV}")