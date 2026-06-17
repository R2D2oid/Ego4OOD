# FFCV Generation

Scripts for re-generating the Ego4OOD `.beton` files from Ego4OOD SlowFast `.npy` features.
**You do not need these scripts to use the dataset** — they are only needed if you want to rebuild the FFCV files from scratch.

## Files

| File | Description |
|---|---|
| `encode.py` | Main script: reads `.npy` features and writes a `.beton` file |
| `feature_dataset.py` | PyTorch Dataset used internally by `encode.py` |
| `config.yaml` | Configuration (paths, feature dim, splits, etc.) |
| `generate.sh` | Runs encoding for a single domain/split |
| `generate_all.sh` | Runs encoding for all domains and splits |
| `requirements.txt` | Python dependencies |

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Edit `config.yaml` and set:
   - `repo_root` — path to the root of this repo (used to locate `annotations/` and `betons/`)
   - `feat_root` — path to the directory containing per-split SlowFast feature folders (`train/`, `val/`)

## Usage

Generate a single domain/split:
```bash
bash generate.sh <holdout_domain> <split>
```

**Example** — generate all splits with `iiith` held out:
```bash
bash generate.sh iiith train   # in-distribution train split
bash generate.sh iiith val     # in-distribution val split
bash generate.sh iiith test    # out-of-distribution test split (iiith only)
```

To generate all domains and splits at once:
```bash
bash generate_all.sh
```

Valid holdout domains: `cmu`, `iiith`, `bristol`, `unict`, `kaust`, `frl_track_1_public`, `utokyo`, `minnesota`
