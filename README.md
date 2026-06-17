# [Ego4OOD: A Domain Generalization Benchmark for Egocentric Video](https://github.com/R2D2oid/Ego4OOD)

**Ego4OOD** is a domain generalization benchmark constructed from moment-level annotations in the [Ego4D](https://ego4d-data.org/) dataset. It provides pre-extracted SlowFast features and FFCV-format files for fast training and evaluation of domain generalization methods on egocentric action recognition.

> **Note:** Access to the raw Ego4D video clips and features requires accepting the [Ego4D license agreement](https://ego4d-data.org/docs/start-here/). The annotation files in this repository are freely available; the `.npy` feature files and `.beton` FFCV files require Ego4D access. To request the `.beton` files, open a [GitHub issue](https://github.com/R2D2oid/Ego4OOD/issues) or email the authors.

---

## Dataset Overview

Ego4OOD leverages the fine-grained, temporally localized moment annotations in Ego4D, covering **203 atomic action classes** grouped into **9 semantically coherent categories**. This categorization provides precise semantic boundaries and reduces ambiguity compared to video-level scenario labels.

| Stat | Value |
|------|-------|
| Total clips | 20,096 |
| Training clips | 15,490 |
| Validation clips | 4,923 |
| Action classes (moments) | 203 |
| Semantic categories | 9 |
| Domains (regions) | 8 |

---

## Domains

Each domain corresponds to a distinct geographic region where Ego4D data was collected.

| Domain key | Region | Index |
|---|---|---|
| `bristol` | UK (University of Bristol) | 2 |
| `iiith` | India (IIIT Hyderabad) | 1 |
| `unict` | Italy (University of Catania) | 3 |
| `kaust` | Saudi Arabia (KAUST) | 4 |
| `frl_track_1_public` | FRL (Facebook Reality Labs) | 5 |
| `cmu` | USA — CMU | 0 |
| `minnesota` | USA — Minnesota | 7 |
| `utokyo` | Japan (University of Tokyo) | 6 |

The standard domain generalization protocol holds out one domain as the out-of-distribution (OOD) test set while training on the remaining seven.

---

## Semantic Categories

The 203 Ego4D moment classes are grouped into 9 categories:

| Category | Index | Example moments |
|---|---|---|
| `construction` | 0 | drill into wall, smooth wood using sandpaper, fix pipe/plumbing |
| `dressing` | 1 | try-out/wear clothing items, fold clothes, iron clothes |
| `foodprep` | 2 | cut/chop/slice a vegetable, fry egg, knead dough |
| `gardening` | 3 | trim hedges, water plants, dig soil with hoe |
| `housechores` | 4 | mop floor, hang clothes to dry, load washing machine |
| `leisure` | 5 | converse/interact with someone, watch television, play video game |
| `personalcare` | 6 | wash hands, clean/arrange shoes |
| `shopping` | 7 | browse groceries on shelf, pay at billing counter, place items in cart |
| `useelectronics` | 8 | use phone, use a laptop/computer, charge electronic device |

The full moment-to-category mapping is in [`annotations/moment_to_idx.json`](annotations/moment_to_idx.json) and [`annotations/category_to_idx.json`](annotations/category_to_idx.json).

---

## Repository Structure

```
Ego4OOD/
├── annotations/
│   ├── train.csv               # training split annotations
│   ├── val.csv                 # validation split annotations
│   ├── domain_to_idx.json      # domain name → integer index
│   ├── category_to_idx.json    # category name → integer index
│   └── moment_to_idx.json      # moment class name → integer index
├── betons/                     # pre-built FFCV .beton files (one per domain/split)
├── usage/
│   ├── load_ffcv.py            # example: load a .beton file with FFCV
│   └── ffcv_config.yaml        # config for the loading example
└── ffcv_generation/            # scripts to rebuild .beton files from scratch (optional)
    ├── README.md
    ├── encode.py
    ├── feature_dataset.py
    ├── config.yaml
    ├── generate.sh             # run encoding for a single domain/split
    ├── generate_all.sh         # run encoding for all domains and splits
    └── requirements.txt
```

---

## Annotation Format

`annotations/train.csv` and `annotations/val.csv` are enriched versions of the official Ego4D moment-level annotations (`moments_train.json` / `moments_val.json`). Each record consolidates the original clip metadata with two additional fields derived from Ego4D video-level metadata: the **geographic domain** of the recording and the **semantic category** grouping of the action class. Each row provides a complete characterization of a sample — clip identity, temporal extent, fine-grained action class, semantic category, and geographic domain — without requiring any additional Ego4D metadata file.

Files use `::` as a field separator to avoid conflicts with action class names:

```
domain::feature_filename.npy::category
```

The feature filename encodes the clip identity and temporal extent:

```
{video_uid}_{start_frame}_{end_frame}_{moment_class}.npy
```

**Example:**
```
cmu::08cd0e23-308f-4120-ae63-42e05d323450_27514_27900_smooth_other_surface_eg_using_sandpaper.npy::construction
```

The base directory for feature files is set via `feat_root` in `ffcv_generation/config.yaml`.

| Field | Source | Description |
|---|---|---|
| `domain` | Ego4D | Geographic region where the video was recorded |
| `video_uid` | Ego4D | Unique video identifier |
| `start_frame` / `end_frame` | Ego4D | Temporal extent of the moment annotation |
| `moment_class` | Ego4D | Fine-grained atomic action label (203 classes) |
| `category` | Ego4OOD | Semantic grouping of the moment class (9 categories) |

---

## Features

Per-moment SlowFast features are sliced from Ego4D's pre-extracted per-video `slowfast8x8_r101_k400` features (available via the [Ego4D download tool](https://ego4d-data.org/docs/start-here/)). Each `.npy` file contains a sequence of 2304-dimensional feature vectors (one per temporal stride of 16 frames). The dataset loader samples 3 evenly-spaced frames from this sequence.

| Property | Value |
|---|---|
| Feature extractor | SlowFast 8×8 R101 (Ego4D, pretrained on Kinetics-400) |
| Feature dimension | 2304 |
| Temporal stride | 16 frames |
| Window length | 8 seconds (240 frames, capped) |
| Samples per clip | 3 (evenly spaced from the window) |
| File format | NumPy `.npy` |

> **Note:** In addition to SlowFast features, we also provide VideoMAE (ViT-B, `MCG-NJU/videomae-base`) features extracted directly from raw Ego4D video frames. These are available on request alongside the `.beton` files.

---

## FFCV Files

Pre-built [FFCV](https://ffcv.io/) `.beton` files enable fast dataloading. One file is generated per **(holdout domain, split)** combination:

```
{holdout_domain}_Ego4OOD_{split}.beton
```

**Splits:** `train` (in-distribution domains), `val` (in-distribution validation), `test` (holdout domain only).

Each sample in a `.beton` file contains three fields:

| Field | Type | Shape | Description |
|---|---|---|---|
| `info` | JSON | — | `(uid, start_frame, end_frame)` clip metadata |
| `feat` | float32 NDArray | `[3, 2304]` | SlowFast features (3 temporal samples) |
| `label` | int32 NDArray | `[3]` | `[domain_idx, moment_idx, category_idx]` |

### Downloading FFCV Files

> The `.beton` files are available upon request. Please open a [GitHub issue](https://github.com/R2D2oid/Ego4OOD/issues) or email the authors directly. Access requires prior acceptance of the [Ego4D license](https://ego4d-data.org/docs/start-here/).

---

## Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate FFCV files from SlowFast features

Set `repo_root` and `feat_root` in `ffcv_generation/config.yaml`, then run:

```bash
# single domain/split
bash ffcv_generation/generate.sh iiith train

# all domains and splits at once
bash ffcv_generation/generate_all.sh
```

This generates `iiith_Ego4OOD_train.beton` (training on all domains except `iiith`) and `iiith_Ego4OOD_test.beton` (OOD test set = `iiith` only) inside `betons/`.

### 3. Load FFCV files in Python

See [`usage/load_ffcv.py`](usage/load_ffcv.py) for a working example, and [`usage/ffcv_config.yaml`](usage/ffcv_config.yaml) for the corresponding config.

---

## Citation

If you use Ego4OOD in your research, please cite:

```bibtex
@inproceedings{ego4ood2025,
  title     = {Ego4OOD: Rethinking Egocentric Video Domain Generalization},
  author    = {Vaseqi, Zahrah and others},
  booktitle = {TODO: add venue},
  year      = {2025}
}
```

Please also cite the original Ego4D dataset:

```bibtex
@inproceedings{grauman2022ego4d,
  title     = {Ego4D: Around the World in 3,000 Hours of Egocentric Video},
  author    = {Grauman, Kristen and others},
  booktitle = {CVPR},
  year      = {2022}
}
```

---

## License

The underlying Ego4D video data and SlowFast features are subject to the [Ego4D Terms of Service](https://ego4d-data.org/docs/start-here/). Access requires registration through the Ego4D consortium.
