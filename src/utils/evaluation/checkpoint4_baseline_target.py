import os
import json
import numpy as np
import soundfile as sf
from pystoi import stoi

# ==========================================================
# Load predeclaration (documents the rules — not re-derived here)
# ==========================================================
with open('configs/checkpoint4_predeclaration.json') as f:
    PREDECLARATION = json.load(f)

EARLY_MS = PREDECLARATION['early_reflection_window_ms']
TAPER_MS = 5
NORM_PEAK = 0.95

FIG_DIR = 'results/figures'
TABLE_DIR = 'results/tables'
AUDIO_DIR = 'results/audio'
os.makedirs(TABLE_DIR, exist_ok=True)

# ==========================================================
# Load room01 artifacts from Checkpoints 2-3
# ==========================================================
clean, sr = sf.read('data/clean/dev/5808-48608-0017.flac')
rir_array = np.load('data/rir_simulated/room01_rir.npy')  # shape (4, rir_len)
reverberant, sr_r = sf.read('data/reverberant/room01_utt01_reverberant.wav')
reverberant = reverberant.T  # (4, samples)
assert sr == sr_r == 16000

# ==========================================================
# STEP 1: detect direct arrival per channel (peak-based rule)
# ==========================================================
def detect_direct_arrival(rir):
    return int(np.argmax(np.abs(rir)))

direct_idx_per_ch = [detect_direct_arrival(rir_array[m]) for m in range(rir_array.shape[0])]
print("Detected direct-arrival sample index per channel:", direct_idx_per_ch)

# ==========================================================
# STEP 2: build the tapered target impulse response per channel
# ==========================================================
def build_target_rir(rir, direct_idx, fs, early_ms, taper_ms):
    early_samples = int(early_ms / 1000 * fs)
    taper_samples = int(taper_ms / 1000 * fs)
    end_idx = direct_idx + early_samples

    target_rir = np.zeros_like(rir)
    target_rir[:end_idx] = rir[:end_idx]

    # Raised-cosine taper over the last `taper_samples` before the cutoff
    if taper_samples > 0 and end_idx - taper_samples >= 0:
        taper_window = 0.5 * (1 + np.cos(np.linspace(0, np.pi, taper_samples)))
        target_rir[end_idx - taper_samples:end_idx] *= taper_window

    return target_rir

target_rirs = np.array([
    build_target_rir(rir_array[m], direct_idx_per_ch[m], sr, EARLY_MS, TAPER_MS)
    for m in range(rir_array.shape[0])
])

# ==========================================================
# STEP 3: build the target signal (clean speech convolved with target RIR)
# NOTE: clean speech is used ONLY here — never given to WPE
# ==========================================================
from scipy.signal import fftconvolve

target_channels = [fftconvolve(clean, target_rirs[m], mode='full') for m in range(target_rirs.shape[0])]
max_len = max(len(t) for t in target_channels)
target = np.zeros((len(target_channels), max_len))
for m, t in enumerate(target_channels):
    target[m, :len(t)] = t

# ==========================================================
# STEP 4: alignment — crop to minimum common length, log residual lag
# ==========================================================
def align_and_crop(sig_a, sig_b):
    """Both assumed to start at sample 0 by construction. Crop to shared length,
    and log residual cross-correlation lag as a sanity check (not used to shift)."""
    min_len = min(len(sig_a), len(sig_b))
    a = sig_a[:min_len]
    b = sig_b[:min_len]

    # Sanity-check lag via cross-correlation on a short window (first 2 seconds)
    check_len = min(min_len, sr * 2)
    corr = np.correlate(a[:check_len] - a[:check_len].mean(),
                         b[:check_len] - b[:check_len].mean(), mode='full')
    lag = np.argmax(np.abs(corr)) - (check_len - 1)
    return a, b, lag

# ==========================================================
# STEP 5: gain normalization (peak-normalize to 0.95, per predeclaration)
# ==========================================================
def peak_normalize(sig, peak=NORM_PEAK):
    m = np.max(np.abs(sig))
    return sig / m * peak if m > 0 else sig

# ==========================================================
# STEP 6: SI-SDR implementation (Le Roux et al. formulation)
# ==========================================================
def si_sdr(reference, estimate, eps=1e-8):
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)
    alpha = np.dot(estimate, reference) / (np.dot(reference, reference) + eps)
    projection = alpha * reference
    noise = estimate - projection
    ratio = np.sum(projection ** 2) / (np.sum(noise ** 2) + eps)
    return 10 * np.log10(ratio + eps)

# ==========================================================
# STEP 7: score the no-processing baseline (reference channel = 0)
# ==========================================================
results = {'predeclaration_file': 'configs/checkpoint4_predeclaration.json'}

ref_ch = 0
target_ref = target[ref_ch]
baseline_ref = reverberant[ref_ch]

t_al, b_al, lag = align_and_crop(target_ref, baseline_ref)
t_norm = peak_normalize(t_al)
b_norm = peak_normalize(b_al)

baseline_stoi = stoi(t_norm, b_norm, sr, extended=False)
baseline_sisdr = si_sdr(t_norm, b_norm)

results['baseline'] = {
    'stoi': float(baseline_stoi),
    'si_sdr_db': float(baseline_sisdr),
    'alignment_residual_lag_samples': int(lag),
    'target_length': len(t_al),
    'baseline_length_before_crop': len(baseline_ref),
}
print(f"\n[Baseline, no processing] STOI: {baseline_stoi:.4f} | SI-SDR: {baseline_sisdr:.2f} dB | "
      f"residual lag: {lag} samples")

# ==========================================================
# BONUS sanity check: score Checkpoint 3's WPE outputs too
# (not required by Checkpoint 4, but validates the scoring pipeline end-to-end)
# ==========================================================
for label, path in [
    ('wpe_single_mic', 'results/audio/checkpoint3_wpe_output_single_mic.wav'),
    ('wpe_four_mic', 'results/audio/checkpoint3_wpe_output_four_mic.wav'),
]:
    if os.path.exists(path):
        wpe_out, sr_w = sf.read(path)
        assert sr_w == sr
        t_al2, w_al2, lag2 = align_and_crop(target_ref, wpe_out)
        t_norm2 = peak_normalize(t_al2)
        w_norm2 = peak_normalize(w_al2)
        s = stoi(t_norm2, w_norm2, sr, extended=False)
        sd = si_sdr(t_norm2, w_norm2)
        results[label] = {
            'stoi': float(s),
            'si_sdr_db': float(sd),
            'alignment_residual_lag_samples': int(lag2),
        }
        print(f"[{label}] STOI: {s:.4f} | SI-SDR: {sd:.2f} dB | residual lag: {lag2} samples")

# ==========================================================
# Save results
# ==========================================================
with open(os.path.join(TABLE_DIR, 'checkpoint4_scores.json'), 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved scores to {os.path.join(TABLE_DIR, 'checkpoint4_scores.json')}")
print("Checkpoint 4 pipeline complete.")