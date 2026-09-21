import pandas as pd
import json

df = pd.read_csv('results/tables/checkpoint8_geometry_heldout_results.csv')

# ==========================================================
# RUNTIME SUMMARY
# ==========================================================
runtime_df = df[df['config_id'] != 'baseline'].copy()

runtime_by_config = runtime_df.groupby('config_id')['runtime_s'].agg(
    ['mean', 'std', 'min', 'max', 'count']
).round(3).reset_index().sort_values('mean')

runtime_by_geometry = runtime_df.groupby('geometry')['runtime_s'].agg(['mean', 'std']).round(3).reset_index()

print("=== Runtime by configuration (seconds per run) ===")
print(runtime_by_config.to_string(index=False))

print("\n=== Runtime by geometry ===")
print(runtime_by_geometry.to_string(index=False))

total_runtime_s = runtime_df['runtime_s'].sum()
print(f"\nTotal WPE compute time: {total_runtime_s:.1f}s ({total_runtime_s/60:.1f} min)")

runtime_by_config.to_csv('results/tables/checkpoint8_runtime_summary.csv', index=False)

# ==========================================================
# FAILURE BREAKDOWN
# ==========================================================
failures = df[df['signal_failure'] == True].copy()

print(f"\n=== Signal failure breakdown ({len(failures)} total) ===")
print(failures[['room_id', 'geometry', 'rt60_target', 'distance_m', 'snr_condition',
                 'utterance', 'config_id']].to_string(index=False))

print("\n--- Failures by geometry ---")
print(failures['geometry'].value_counts().to_string())

print("\n--- Failures by RT60 condition ---")
print(failures['rt60_target'].value_counts().to_string())

print("\n--- Failures by distance ---")
print(failures['distance_m'].value_counts().to_string())

print("\n--- Failures by configuration ---")
print(failures['config_id'].value_counts().to_string())

print("\n--- Failures by SNR ---")
print(failures['snr_condition'].value_counts().to_string())

failures.to_csv('results/tables/checkpoint8_failure_breakdown.csv', index=False)

# ==========================================================
# Merge into transfer summary
# ==========================================================
with open('results/tables/checkpoint8_transfer_summary.json') as f:
    summary = json.load(f)

summary['runtime_summary'] = {
    'mean_runtime_s_by_config': runtime_by_config.set_index('config_id')['mean'].to_dict(),
    'total_runtime_s': float(total_runtime_s),
    'mean_runtime_s_by_geometry': runtime_by_geometry.set_index('geometry')['mean'].to_dict(),
}
summary['failure_breakdown_detail'] = {
    'by_geometry': failures['geometry'].value_counts().to_dict(),
    'by_rt60': failures['rt60_target'].value_counts().to_dict(),
    'by_distance': failures['distance_m'].value_counts().to_dict(),
    'by_config': failures['config_id'].value_counts().to_dict(),
    'by_snr': failures['snr_condition'].value_counts().to_dict(),
}

with open('results/tables/checkpoint8_transfer_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\nCheckpoint 8 fully complete per all report requirements.")