# Does Echo Removal Tuned in Simulated Rooms Hold Up in New Rooms?

**A reproducible audit of the WPE dereverberation algorithm under room-model mismatch**

Anirudh Koduganti · South Brunswick High School, NJ
Research mentor · Dr. Nicoleta Roman, The Ohio State University

> **Paper.** *A Reproducible Audit of Weighted Prediction Error Under Room-Model Mismatch.* Sole-author paper accepted to the IEEE BigData 2026 High School Symposium, to be published in the IEEE BigData 2026 conference proceedings.

---

## What this project is about (no background needed)

When someone talks in a room, a microphone records their voice plus every echo off the walls, floor, and ceiling. Those late echoes make speech sound blurry and far away, which hurts hearing aids, video calls, smart speakers, and speech recognition.

**Weighted Prediction Error (WPE)** is one of the most widely used algorithms for removing those echoes. Before anyone can use it, they have to choose two settings.

| Setting | What it controls |
|---|---|
| **Prediction delay** | How far back in time WPE starts looking for echoes. Too short, and it deletes parts of the real voice. |
| **Filter length** | How much of the echo "tail" WPE tries to model. Longer is slower and not always better. |

Researchers usually pick these settings by testing in **computer-simulated rooms**, because simulated rooms are cheap and easy to control. That raises a simple question, which is what this project tests.

**If I pick the best WPE settings in simulated rooms, does that choice still hold in rooms I never tuned on?**

---

## Short answer

I ranked all 9 combinations of the two settings in simulated "development" rooms. Then I **froze that ranking in a saved, version-controlled file before generating any test data**, so I could not adjust it after seeing the answer. Finally, I tested the frozen ranking in rooms it had never seen.

| Test domain | Runs | Same top 3? | Ranking agreement (Kendall's τ) | Loss from trusting my top 3 |
|---|---:|:---:|:---:|:---:|
| Development rooms (where I tuned) | 1,440 | – | – | – |
| New simulated rooms, new speakers | 1,440 | **3 of 3** | **0.944** (35 of 36 pairs agree) | **0.0000** |
| New simulated room shapes and sizes | 2,880 | **3 of 3** | **0.944** (35 of 36 pairs agree) | **0.0000** |
| Real measured rooms (in the paper) | 240 | **3 of 3** | **0.611** | **0.0000** |

**In plain words.** The same three settings stayed on top everywhere, and picking from that shortlist never cost any performance. The finer ordering below the top held almost perfectly in new simulated rooms and slipped in real rooms. Across all experiments, the frozen grid ran **6,030 times with zero numerical failures**.

Three more findings came out of the data.

- **The prediction delay mattered far more than the filter length.** The three settings with the longest delay (4 frames, 32 ms) took ranks 1 to 3 in every domain. The shortest delay (2 frames) made speech *less* clear than doing nothing.
- **The best setting was also the cheapest.** The top development setting (C7) used the shortest filter and ran in about 1.1 seconds per clip, compared with about 6.5 seconds for the longest filter.
- **The same sentences kept failing.** The clips where WPE made things worse were mostly the same specific utterances, especially with the speaker 2.5 m away. That points to the speech itself, along with the room, as a cause of instability.

---

## How the experiment works

```text
LibriSpeech clean speech ──► simulated room (Pyroomacoustics) ──► 4-microphone reverberant recording
                                                                          │
                                                                          ▼
                                     9 WPE settings (NARA-WPE)  ◄── same input for every setting
                                                                          │
                                                                          ▼
                     score vs. "direct sound + first 50 ms" target (STOI, SI-SDR) ──► rank ──► FREEZE ──► test in new rooms
```

| Piece | Details |
|---|---|
| **Speech** | 36 LibriSpeech utterances. 18 from 6 development speakers (`train-clean-100`) and 18 from 6 different test speakers (`test-clean`). No speaker appears in both. |
| **Rooms** | Shoebox rooms simulated with Pyroomacoustics. Echo times (RT60) of 0.2, 0.4, and 0.6 s, speaker distances of 1.0 and 2.5 m, and a clean or 10 dB noise condition. |
| **Room shapes** | Development rooms are 5.0 × 4.0 × 2.8 m (G-A). The shape test uses 6.0 × 5.0 × 3.0 m (G-C) and 10.0 × 6.0 × 3.5 m (G-D). |
| **Microphones** | 4 microphones in an 8 cm square at the center of the room. |
| **The 9 settings** | Prediction delay of 2, 3, or 4 frames × filter length of 10, 20, or 30 frames (C1 to C9). |
| **Score** | STOI (speech intelligibility) improvement over the unprocessed recording, averaged within each room condition and then across conditions. SI-SDR is also recorded. |
| **"Clean" target** | The direct sound plus the first 50 ms of reflections, with a 5 ms fade, so WPE is judged on removing late echoes only. |

### Why the ranking was frozen first

The rules for scoring were written down in [`configs/checkpoint4_predeclaration.json`](configs/checkpoint4_predeclaration.json) before any experiment ran. The development ranking and the top-3 shortlist were saved in [`configs/checkpoint6_frozen_experiment.json`](configs/checkpoint6_frozen_experiment.json) at commit [`a3d5d3b`](https://github.com/AnirudhKoduganti/wpe-project/commit/a3d5d3bf50155a9acf654fc8424e1286bdf6e712), before any test data existed. Every later checkpoint reads that file and never retunes. Anyone can check the commit history to confirm the order.

---

## Reproduce it

### 1. Requirements

The results were produced on **Windows 11 with Python 3.14.6**. The exact package versions are in [`results/tables/environment.txt`](results/tables/environment.txt). The main ones are listed below.

```bash
git clone https://github.com/AnirudhKoduganti/wpe-project.git
cd wpe-project

python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
```

If you are on a different Python version and the pinned versions fail, install the main packages directly.

```bash
pip install numpy scipy pandas matplotlib soundfile tqdm pyroomacoustics nara-wpe pystoi pesq
```

> [!IMPORTANT]
> Run every command from the **repository root**. All scripts use paths like `data/...` and `configs/...` relative to the root.

### 2. Data

The 36 clean speech files are already in [`data/clean/`](data/clean), and the speaker split is in [`data/manifests/manifest.csv`](data/manifests/manifest.csv), so **no download is needed**.

<details>
<summary>Rebuilding the speech set from scratch (optional)</summary>

1. Download `train-clean-100` and `test-clean` from [LibriSpeech (OpenSLR 12)](https://www.openslr.org/12).
2. Edit `TRAIN_CLEAN_100` and `TEST_CLEAN` near the top of [`src/utils/build_manifest.py`](src/utils/build_manifest.py) to point to your download.
3. Run `python src/utils/build_manifest.py`. It uses `random.seed(42)`, so it picks the same speakers.

</details>

### 3. Run the checkpoints in order

Each checkpoint matches one commit in the history, and each one checks something before the next step builds on it.

| # | What it checks | Command | Main output |
|:---:|---|---|---|
| 0 | Clean speech loads at 16 kHz | `python notebooks/checkpoint0.py` | printed PASS |
| 2 | Simulated room and echo look correct | `python src/utils/simulation/room.py` | `data/rir_simulated/`, `results/figures/checkpoint2_*.png` |
| 3 | WPE runs cleanly on 1 and 4 microphones | `python src/utils/wpe/run_checkpoint3.py` | `results/tables/checkpoint3_results.json` |
| 4 | Target and baseline scores follow the predeclared rules | `python src/utils/evaluation/checkpoint4_baseline_target.py` | `results/tables/checkpoint4_scores.json` |
| 5 | All 9 settings on 3 sample rooms, with audio to listen to | `python src/utils/experiments/checkpoint5_parameter_study.py` | `results/tables/checkpoint5_parameter_study.csv`, `results/audio/checkpoint5_*.wav` |
| 6 | Development experiment (1,440 runs) | `python src/utils/experiments/checkpoint6_development_experiment.py` | `results/tables/checkpoint6_development_results.csv` |
| 6 | Rank the 9 settings and freeze the shortlist | `python src/utils/experiments/checkpoint6_rank_and_freeze.py` | `configs/checkpoint6_frozen_experiment.json` |
| 7 | New simulated rooms and new speakers (1,440 runs) | `python src/utils/experiments/checkpoint7_heldout_simulations.py` | `results/tables/checkpoint7_transfer_summary.json` |
| 7 | Runtime and failure summary | `python src/utils/experiments/checkpoint7_runtime_failure_summary.py` | `results/tables/checkpoint7_*` |
| 8 | New room shapes and sizes (2,880 runs) | `python src/utils/experiments/checkpoint8_geometry_heldout.py` | `results/tables/checkpoint8_transfer_summary.json` |
| 8 | Runtime and failure summary | `python src/utils/experiments/checkpoint8_runtime_failure_summary.py` | `results/tables/checkpoint8_*` |

Checkpoint 1 is the manifest from step 2, which is already committed.

> [!NOTE]
> On a laptop, WPE alone took about 70 minutes for Checkpoint 6, about 70 minutes for Checkpoint 7, and about 2.4 hours for Checkpoint 8. Results are appended row by row, so a stopped run keeps its progress in the CSV.

> [!WARNING]
> `checkpoint6_rank_and_freeze.py` **overwrites** `configs/checkpoint6_frozen_experiment.json` and records *your* current commit. To check my original freeze, compare your ranking with the committed file instead of re-running the freeze step.

### 4. Check your results against mine

Open `results/tables/checkpoint7_transfer_summary.json` and `results/tables/checkpoint8_transfer_summary.json` and look for these values.

```json
"top3_overlap_count": 3,
"kendalls_tau": 0.9444444444444445,
"pairwise_agreement_fraction": "35/36",
"selection_regret_stoi": 0.0,
"numerical_failures": 0
```

All random seeds are fixed in the scripts (`SEED_BASE` = 200, 300, and 400 for Checkpoints 6, 7, and 8), so the same environment should give the same numbers. Small differences can appear across operating systems or package versions.

---

## Repository map

```text
wpe-project/
├── configs/                 # predeclared rules, fixed settings, and the frozen ranking
├── data/
│   ├── clean/               # 36 LibriSpeech utterances (dev/ and test/)
│   ├── manifests/           # manifest.csv with the speaker-disjoint split
│   ├── rir_simulated/       # example simulated room impulse response + metadata
│   └── reverberant/         # example reverberant recording
├── notebooks/
│   └── checkpoint0.py       # audio loading check
├── results/
│   ├── audio/               # before/after clips you can listen to
│   ├── figures/             # waveforms and spectrograms
│   └── tables/              # every result CSV/JSON + environment.txt
└── src/utils/
    ├── build_manifest.py    # builds the speaker split
    ├── simulation/          # room simulation, convolution, noise
    ├── wpe/                 # first WPE run (Checkpoint 3)
    ├── evaluation/          # target definition and baseline scoring
    └── experiments/         # Checkpoints 5 to 8
```

**Want to hear it?** Compare [`checkpoint5_P03_strong_far_baseline.wav`](results/audio/checkpoint5_P03_strong_far_baseline.wav) (echoey room, speaker 2.5 m away) with [`checkpoint5_P03_strong_far_C7.wav`](results/audio/checkpoint5_P03_strong_far_C7.wav) (after the top setting) and [`checkpoint5_P03_strong_far_C1.wav`](results/audio/checkpoint5_P03_strong_far_C1.wav) (after a short-delay setting).

---

## Limitations and open questions

- **Scale.** The experiments use 6 utterances per split and 2 replicates per room condition, which is smaller than a full-scale study, so per-condition numbers have wider error bars.
- **Real rooms.** The measured-room stage uses real impulse responses from [BUT ReverbDB](https://speech.fit.vutbr.cz/software/but-speech-fit-reverb-database), mixed with clean speech. It is not live recordings, and its code is not in this repository yet.
- **No listening study.** All scores are computed metrics. Listening notes from Checkpoint 5 are informal.
- **The four-microphone puzzle.** In Checkpoint 4, WPE with all four microphones scored *worse* than one microphone and worse than doing nothing (STOI 0.884 vs. 0.935 and 0.904). That is still not explained, and it is the next thing I want to figure out.

---

## Acknowledgments

Thank you to Dr. Nicoleta Roman at The Ohio State University for mentoring this project. Speech comes from [LibriSpeech](https://www.openslr.org/12) (CC BY 4.0). WPE uses [NARA-WPE](https://github.com/fgnt/nara_wpe), rooms use [Pyroomacoustics](https://github.com/LCAV/pyroomacoustics), and STOI uses [pystoi](https://github.com/mpariente/pystoi).
