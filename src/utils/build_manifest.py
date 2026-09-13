import os
import shutil
import random
import csv
import soundfile as sf

random.seed(42)  # fixed seed = reproducible speaker/utterance selection

def collect_speaker_files(root_dir, n_speakers, n_utts_per_speaker, exclude_speakers=None):
    """Randomly pick n_speakers from root_dir, and n_utts_per_speaker .flac files from each."""
    exclude_speakers = exclude_speakers or set()
    all_speakers = [s for s in os.listdir(root_dir)
                    if os.path.isdir(os.path.join(root_dir, s)) and s not in exclude_speakers]
    chosen_speakers = random.sample(all_speakers, n_speakers)

    selected = []
    for speaker in chosen_speakers:
        speaker_dir = os.path.join(root_dir, speaker)
        flac_files = []
        for chapter in os.listdir(speaker_dir):
            chapter_dir = os.path.join(speaker_dir, chapter)
            if os.path.isdir(chapter_dir):
                for f in os.listdir(chapter_dir):
                    if f.endswith('.flac'):
                        flac_files.append(os.path.join(chapter_dir, f))
        chosen_utts = random.sample(flac_files, n_utts_per_speaker)
        for utt_path in chosen_utts:
            selected.append((speaker, utt_path))
    return selected, chosen_speakers


def copy_and_log(selected, dest_dir, split_name, manifest_rows):
    for speaker_id, src_path in selected:
        filename = os.path.basename(src_path)
        utterance_id = os.path.splitext(filename)[0]
        dest_path = os.path.join(dest_dir, filename)
        shutil.copy(src_path, dest_path)

        data, sr = sf.read(dest_path)
        duration_s = len(data) / sr

        manifest_rows.append({
            'speaker_id': speaker_id,
            'utterance_id': utterance_id,
            'split': split_name,
            'path': dest_path.replace('\\', '/'),
            'sample_rate': sr,
            'duration_s': round(duration_s, 3),
        })


# ---- EDIT THESE PATHS ----
TRAIN_CLEAN_100 = r'C:\Users\Anirudh\Downloads\train-clean-100\LibriSpeech\train-clean-100'
TEST_CLEAN = r'C:\Users\Anirudh\Downloads\test-clean\LibriSpeech\test-clean'
DEV_DEST = 'data/clean/dev'
TEST_DEST = 'data/clean/test'
MANIFEST_PATH = 'data/manifests/manifest.csv'
# ---------------------------

manifest_rows = []

dev_selected, dev_speakers = collect_speaker_files(TRAIN_CLEAN_100, n_speakers=6, n_utts_per_speaker=3)
copy_and_log(dev_selected, DEV_DEST, 'dev', manifest_rows)

test_selected, test_speakers = collect_speaker_files(TEST_CLEAN, n_speakers=6, n_utts_per_speaker=3,
                                                        exclude_speakers=set(dev_speakers))
copy_and_log(test_selected, TEST_DEST, 'test', manifest_rows)

os.makedirs('data/manifests', exist_ok=True)
with open(MANIFEST_PATH, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['speaker_id', 'utterance_id', 'split', 'path', 'sample_rate', 'duration_s'])
    writer.writeheader()
    writer.writerows(manifest_rows)

print(f"Dev speakers: {dev_speakers}")
print(f"Test speakers: {test_speakers}")
print(f"Manifest written with {len(manifest_rows)} rows to {MANIFEST_PATH}")