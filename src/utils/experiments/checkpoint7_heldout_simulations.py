import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
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
# LOAD THE FROZEN EXPERIMENT (Checkpoint 6) — do not deviate from this
# ==========================================================
with open('configs/checkpoint6_frozen_experiment.json') as f:
    FROZEN = json.load(f)

CONFIGS = FROZEN['all_nine_configurations']
TOP3_DEV = FROZEN['top_three_shortlist']
DEV_RANKING = pd.DataFrame(FROZEN['full_ranking'])  # has config_id, macro_stoi_improvement, rank

# ==========================================================
# FIXED SETTINGS — identical to Checkpoint 6 (same array, simulator, target rule)
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
ROOM_DIM = [5.0, 4.0, 2.8]  # same geometry G-A — this is a HELD-OUT test, not a new-geometry test

# SAME broad ranges as development (per the guide's requirement)
RT60_TARGETS = [0.2, 0.4, 0.6]
DISTANCES = [1.0, 2.5]
N_REPLICATES = 2
SNR_CONDITIONS = [None, 10]

UTTERANCES_PER_SPEAKER = 1  # matches Checkpoint 6's scope decision

MANIFEST_PATH = 'data/manifests/manifest.csv'
RESULTS_CSV = 'results/tables/checkpoint7_heldout_results.csv'
os.makedirs('results/tables', exist_ok=True)

# NEW seed range — must not overlap with Checkpoint 6's seeds (201-212)
SEED_BASE = 300

CSV_COLUMNS = [
    'room_id', 'rt60_target', 'rt60_achieved', 'distance_m', 'replicate', 'seed',
    'snr_condition', 'utterance', 'config_id', 'delay', 'taps',
    'stoi', 'si_sdr_db', 'runtime_s', 'numerical_failure', 'signal_failure', 'warnings'
]

# ==========================================================
# Shared helper functions — IDENTICAL logic to Checkpoints 4-6
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
# RESUME SUPPORT
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
# Load TEST utterances — must be disjoint from dev speakers (manifest already enforces this)
# ==========================================================
manifest = pd.read_csv(MANIFEST_PATH)
test = manifest[manifest['split'] == 'test']
utt_rows = test.groupby('speaker_id').head(UTTERANCES_PER_SPEAKER)
UTTERANCES = list(utt_rows['path'])
print(f"Using {len(UTTERANCES)} TEST utterances (held-out speakers, never used in development): {UTTERANCES}")

# ==========================================================
# MAIN LOOP — same room-condition structure as Checkpoint 6, NEW seeds
# ==========================================================
room_metadata_log = {}
combo_index = 0

for rt60_target in RT60_TARGETS:
    for distance in DISTANCES:
        for replicate in range(N_REPLICATES):
            combo_index += 1
            seed = SEED_BASE + combo_index  # 301, 302, ... never overlaps dev seeds 201-212
            room_id = f"M_rt{rt60_target}_d{distance}_rep{replicate}"

            rir_array, room_meta = generate_room_rir(ROOM_DIM, FS, rt60_target, distance, MIC_SPACING, seed)
            room_meta['seed'] = seed
            room_metadata_log[room_id] = room_meta
            print(f"\n=== {room_id} (seed={seed}, NEW RIR) | achieved RT60: {room_meta['rt60_achieved_s']} ===")

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

                    # ---- baseline ----
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

                    # ---- frozen nine configs, no retuning ----
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

with open('results/tables/checkpoint7_room_metadata.json', 'w') as f:
    json.dump(room_metadata_log, f, indent=2)

print(f"\nCheckpoint 7 data collection complete. {len(done_keys)} total rows in {RESULTS_CSV}")

# ==========================================================
# ANALYSIS: ranking, rank concordance, top-3 overlap, selection regret
# ==========================================================
df = pd.read_csv(RESULTS_CSV)
baseline = df[df['config_id'] == 'baseline'][
    ['room_id', 'snr_condition', 'utterance', 'stoi', 'rt60_target', 'distance_m']
].rename(columns={'stoi': 'baseline_stoi'})

configs_df = df[df['config_id'] != 'baseline'].copy()
merged = configs_df.merge(baseline, on=['room_id', 'snr_condition', 'utterance', 'rt60_target', 'distance_m'])
merged['stoi_improvement'] = merged['stoi'] - merged['baseline_stoi']

cell_avg = merged.groupby(['config_id', 'rt60_target', 'distance_m'])['stoi_improvement'].mean().reset_index()
cell_avg.rename(columns={'stoi_improvement': 'cell_stoi_improvement'}, inplace=True)
cell_avg.to_csv('results/tables/checkpoint7_cell_level.csv', index=False)

heldout_macro = cell_avg.groupby('config_id')['cell_stoi_improvement'].mean().reset_index()
heldout_macro.rename(columns={'cell_stoi_improvement': 'macro_stoi_improvement'}, inplace=True)
heldout_macro = heldout_macro.sort_values('macro_stoi_improvement', ascending=False).reset_index(drop=True)
heldout_macro['rank'] = heldout_macro.index + 1
heldout_macro.to_csv('results/tables/checkpoint7_ranking.csv', index=False)

print("\n=== Held-out ranking (macro STOI improvement) ===")
print(heldout_macro.to_string(index=False))

# ---- Rank concordance with development ranking ----
merged_ranks = DEV_RANKING[['config_id', 'rank']].merge(
    heldout_macro[['config_id', 'rank']], on='config_id', suffixes=('_dev', '_heldout')
)
tau, p_value = kendalltau(merged_ranks['rank_dev'], merged_ranks['rank_heldout'])

# Pairwise-order agreement (of the 36 config pairs)
pairs_total, pairs_agree = 0, 0
configs_list = merged_ranks['config_id'].tolist()
for i in range(len(configs_list)):
    for j in range(i + 1, len(configs_list)):
        c1, c2 = configs_list[i], configs_list[j]
        dev1 = merged_ranks.loc[merged_ranks.config_id == c1, 'rank_dev'].iloc[0]
        dev2 = merged_ranks.loc[merged_ranks.config_id == c2, 'rank_dev'].iloc[0]
        ho1 = merged_ranks.loc[merged_ranks.config_id == c1, 'rank_heldout'].iloc[0]
        ho2 = merged_ranks.loc[merged_ranks.config_id == c2, 'rank_heldout'].iloc[0]
        pairs_total += 1
        if (dev1 < dev2) == (ho1 < ho2):
            pairs_agree += 1
pairwise_agreement = pairs_agree / pairs_total

# ---- Top-3 overlap ----
top3_heldout = heldout_macro.head(3)['config_id'].tolist()
top3_overlap = len(set(TOP3_DEV) & set(top3_heldout))

# ---- Selection regret ----
best_of_9 = heldout_macro['macro_stoi_improvement'].max()
best_of_top3_dev = heldout_macro[heldout_macro['config_id'].isin(TOP3_DEV)]['macro_stoi_improvement'].max()
selection_regret = best_of_9 - best_of_top3_dev

transfer_summary = {
    'development_top3': TOP3_DEV,
    'heldout_top3': top3_heldout,
    'top3_overlap_count': top3_overlap,
    'kendalls_tau': float(tau),
    'kendalls_tau_p_value': float(p_value),
    'pairwise_order_agreement': pairwise_agreement,
    'pairwise_agreement_fraction': f"{pairs_agree}/{pairs_total}",
    'best_of_nine_heldout': float(best_of_9),
    'best_of_dev_top3_on_heldout': float(best_of_top3_dev),
    'selection_regret_stoi': float(selection_regret),
    'heldout_ranking': heldout_macro.to_dict(orient='records'),
    'numerical_failures': int(df['numerical_failure'].sum()),
    'signal_failures': int(df['signal_failure'].fillna(False).sum()),
    'total_runs': len(df),
}

with open('results/tables/checkpoint7_transfer_summary.json', 'w') as f:
    json.dump(transfer_summary, f, indent=2)

print("\n=== TRANSFER SUMMARY ===")
print(f"Development top-3: {TOP3_DEV}")
print(f"Held-out top-3:    {top3_heldout}")
print(f"Top-3 overlap:     {top3_overlap}/3")
print(f"Kendall's tau:     {tau:.3f} (p={p_value:.4f})")
print(f"Pairwise agreement: {pairs_agree}/{pairs_total} ({pairwise_agreement:.1%})")
print(f"Selection regret (STOI): {selection_regret:.4f}")
print(f"Numerical failures: {transfer_summary['numerical_failures']}, Signal failures: {transfer_summary['signal_failures']}")
print("\nCheckpoint 7 complete.")