# NSSC 2026 — HiRISE unsupervised anomaly detection

IIT Kharagpur · Data Analytics 

Conv autoencoder (from scratch) → Isolation Forest on latents → MAD threshold → reconstruction heatmaps + metadata join.

---

## What judges need (Unstop checklist)

| Required | Path |
|----------|------|
| Notebook with code + outputs | `HiRISE_Anomaly_Detection.ipynb` |
| PDF report | `REPORT.pdf` |
| Changelog (≥3 versions, symptom → diagnosis → fix) | `MODEL_CHANGELOG.md` |

Everything below also lives in this repo so the work is reproducible.

---

## Repo layout

```
HiRISE_Anomaly_Detection.ipynb   # main submission notebook
REPORT.pdf
MODEL_CHANGELOG.md
README.md
requirements.txt
data/
  crop_metadata_index.csv
  source_image_metadata.csv
  DA PS 26'.docx                 # problem statement
  DATA_DESCRIPTION.docx
  RULES AND REGULATIONS.docx
# images/ + images.zip are local-only (.gitignore) — unzip locally to train
src/
  model.py                       # ConvAE + MSE/SSIM/grad losses
  dataset.py                     # crop loader
  anomaly.py                     # Isolation Forest + threshold + fusion
  plots.py                       # figure export
  train_and_analyze.py           # train / score / save outputs
submission/
  checkpoints/cae_v4.pt          # trained weights (float16; cae_v3.pt mirrored)
  figures/                       # all report figures
  outputs/                       # scored CSV, threshold JSON, latents
```

Problem-statement PDFs/DOCX from the organisers can sit in the repo root if you want them for reference.

---

## Results (v4)

- Latent dim **512**
- Loss **0.40·MSE + 0.35·(1−SSIM) + 0.25·∇L1** (v4 continue-train from v3)
- Novelty **−IF.score_samples** (higher = more anomalous); IF **600** trees
- Threshold / flagged: see `submission/outputs/threshold_meta.json` (**MAD**, currently **161 / 10422**)
- Optional: recon hybrid + metadata fusion (comparative)

---

## How to run

```bash
pip install -r requirements.txt
# unzip competition crops locally (not in git): images/images/*.jpg
# full train
python src/train_and_analyze.py --epochs 21 --batch-size 24 --latent-dim 512
# or continue-train from checkpoint (v4 path)
python src/train_and_analyze.py --finetune --epochs 8 --batch-size 24 --lr 5e-5
# or reuse checkpoint only
python src/train_and_analyze.py --skip-train
```

Then open `HiRISE_Anomaly_Detection.ipynb` (loads the checkpoint; cells already have outputs).

Checkpoint: `submission/checkpoints/cae_v4.pt` (also mirrored as `cae_v3.pt` for older notebook cells).

---

### Size notes

| File | Approx size | Notes |
|------|-------------|--------|
| `cae_v3.pt` | ~55 MB | float16 weights — under 100 MB limit |
| notebook / PDF / figures | smaller | normal git |
| `images/`, `images.zip` | ~138 MB | **not pushed** (`.gitignore`) |

---

## Team IDs (nssc.in)

- Participant IDs: `26-985995,26-711062,26-505811,26-435653`
- Team ID: `T-0637108` 
