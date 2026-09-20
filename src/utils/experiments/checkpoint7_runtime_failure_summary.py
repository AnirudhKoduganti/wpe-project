import pandas as pd
import json

df = pd.read_csv('results/tables/checkpoint7_heldout_results.csv')

# ==========================================================
# RUNTIME SUMMARY (per configuration, and per RT60 condition)
# ==========================================================
runtime_df = df[df['config_id'] != 'baseline'].copy()

runtime_by_config = runtime_df.groupby('config_id')['runtime_s'].agg(
    ['mean', 'std', 'min', 'max', 'count']
).round(3).reset_index()
runtime_by_config = runtime_by_config.sort_values('mean')

runtime_by_rt60 = runtime_df.groupby('rt60_target')['runtime_s'].agg(
    ['mean', 'std']
).round(3).reset_index()

print("=== Runtime by configuration (seconds per run) ===")
print(runtime_by_config.to_string(index=False))

print("\n=== Runtime by RT60 condition (seconds per run) ===")
print(runtime_by_rt60.to_string(index=False))

total_runtime_s = runtime_df['runtime_s'].sum()
print(f"\nTotal WPE compute time across all Checkpoint 7 runs: {total_runtime_s:.1f}s ({total_runtime_s/60:.1f} min)")

runtime_by_config.to_csv('results/tables/checkpoint7_runtime_summary.csv', index=False)

# ==========================================================
# FAILURE BREAKDOWN — where exactly did the 11 signal failures occur?
# ==========================================================
failures = df[df['signal_failure'] == True].copy()

print(f"\n=== Signal failure breakdown ({len(failures)} total) ===")
print(failures[['room_id', 'rt60_target', 'distance_m', 'snr_condition',
                 'utterance', 'config_id', 'delay', 'taps']].to_string(index=False))

print("\n--- Failures by RT60 condition ---")
print(failures['rt60_target'].value_counts().to_string())

print("\n--- Failures by configuration ---")
print(failures['config_id'].value_counts().to_string())

print("\n--- Failures by SNR condition ---")
print(failures['snr_condition'].value_counts().to_string())

failures.to_csv('results/tables/checkpoint7_failure_breakdown.csv', index=False)

# ==========================================================
# Update the transfer summary JSON with runtime + failure detail
# ==========================================================
with open('results/tables/checkpoint7_transfer_summary.json') as f:
    summary = json.load(f)

summary['runtime_summary'] = {
    'mean_runtime_s_by_config': runtime_by_config.set_index('config_id')['mean'].to_dict(),
    'total_runtime_s': float(total_runtime_s),
    'mean_runtime_s_by_rt60': runtime_by_rt60.set_index('rt60_target')['mean'].to_dict(),
}
summary['failure_breakdown'] = {
    'by_rt60': failures['rt60_target'].value_counts().to_dict(),
    'by_config': failures['config_id'].value_counts().to_dict(),
    'by_snr': failures['snr_condition'].value_counts().to_dict(),
}

with open('results/tables/checkpoint7_transfer_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\nUpdated checkpoint7_transfer_summary.json with runtime and failure breakdowns.")
print("Checkpoint 7 fully complete per all report requirements.")