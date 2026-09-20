import os
import json
import subprocess
import pandas as pd

RESULTS_CSV = 'results/tables/checkpoint6_development_results.csv'
RANKING_CSV = 'results/tables/checkpoint6_ranking.csv'
FREEZE_JSON = 'configs/checkpoint6_frozen_experiment.json'

df = pd.read_csv(RESULTS_CSV)

baseline = df[df['config_id'] == 'baseline'][
    ['room_id', 'snr_condition', 'utterance', 'stoi', 'rt60_target', 'distance_m']
].rename(columns={'stoi': 'baseline_stoi'})

configs_df = df[df['config_id'] != 'baseline'].copy()
merged = configs_df.merge(baseline, on=['room_id', 'snr_condition', 'utterance', 'rt60_target', 'distance_m'])
merged['stoi_improvement'] = merged['stoi'] - merged['baseline_stoi']

# Step 1-2: average across utterances/replicates/SNR within each (config, rt60, distance) cell
cell_avg = merged.groupby(['config_id', 'rt60_target', 'distance_m'])['stoi_improvement'].mean().reset_index()
cell_avg.rename(columns={'stoi_improvement': 'cell_stoi_improvement'}, inplace=True)

# Step 3-4: average equally across the 6 (RT60 x distance) cells -> macro score per config
macro = cell_avg.groupby('config_id')['cell_stoi_improvement'].mean().reset_index()
macro.rename(columns={'cell_stoi_improvement': 'macro_stoi_improvement'}, inplace=True)
macro = macro.sort_values('macro_stoi_improvement', ascending=False).reset_index(drop=True)
macro['rank'] = macro.index + 1

print("=== Development ranking (macro STOI improvement over baseline, equal-weighted across cells) ===")
print(macro.to_string(index=False))

macro.to_csv(RANKING_CSV, index=False)
cell_avg.to_csv('results/tables/checkpoint6_cell_level.csv', index=False)

top3 = macro.head(3)['config_id'].tolist()
print(f"\nTop-3 shortlist: {top3}")

# ---- Get git commit hash for a hard freeze marker ----
try:
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
except Exception:
    commit = 'unavailable'

freeze = {
    'all_nine_configurations': [
        {'id': 'C1', 'delay': 2, 'taps': 10}, {'id': 'C2', 'delay': 2, 'taps': 20}, {'id': 'C3', 'delay': 2, 'taps': 30},
        {'id': 'C4', 'delay': 3, 'taps': 10}, {'id': 'C5', 'delay': 3, 'taps': 20}, {'id': 'C6', 'delay': 3, 'taps': 30},
        {'id': 'C7', 'delay': 4, 'taps': 10}, {'id': 'C8', 'delay': 4, 'taps': 20}, {'id': 'C9', 'delay': 4, 'taps': 30},
    ],
    'top_three_shortlist': top3,
    'selection_metric': 'Mean STOI improvement over no-processing baseline, averaged first within each (RT60, distance) cell, then averaged equally across the 6 cells (macro-average).',
    'full_ranking': macro.to_dict(orient='records'),
    'target_definition_reference': 'configs/checkpoint4_predeclaration.json',
    'manifest_reference': 'data/manifests/manifest.csv',
    'dataset_scope_note': 'This development run used 1 utterance per dev speaker (6 total) rather than the full 18, for compute tractability. See run script UTTERANCES_PER_SPEAKER constant.',
    'git_commit_at_freeze': commit,
    'environment_reference': 'results/tables/environment.txt',
    'room_geometry': 'G-A: 5.0 x 4.0 x 2.8 m',
    'rt60_targets': [0.2, 0.4, 0.6],
    'distances_m': [1.0, 2.5],
    'snr_conditions': ['clean', '10dB'],
    'replicates_per_cell': 2,
}

os.makedirs('configs', exist_ok=True)
with open(FREEZE_JSON, 'w') as f:
    json.dump(freeze, f, indent=2)

print(f"\nFROZEN. No configuration may be added or changed after this point.")
print(f"Freeze record saved to {FREEZE_JSON} at git commit {commit}")