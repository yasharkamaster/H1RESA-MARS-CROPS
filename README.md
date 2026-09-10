# NSSC 2026 — HiRISE unsupervised anomaly detection

IIT Kharagpur · Data Analytics (Unstop online round).

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
crop_metadata_index.csv
source_image_metadata.csv
# images/ + images.zip are local-only (.gitignore) — unzip locally to train
src/
  model.py                       # ConvAE + MSE/SSIM/grad losses
  dataset.py                     # crop loader
  anomaly.py                     # Isolation Forest + threshold + fusion
  plots.py                       # figure export
  train_and_analyze.py           # train / score / save outputs
submission/
  checkpoints/cae_v3.pt          # trained weights (float16)
  figures/                       # all report figures
  outputs/                       # scored CSV, threshold JSON, latents
```

Problem-statement PDFs/DOCX from the organisers can sit in the repo root if you want them for reference.

---

## Results (v3)

- Latent dim **512**
- Loss **0.40·MSE + 0.35·(1−SSIM) + 0.25·∇L1**
- Novelty **−IF.score_samples** (higher = more anomalous)
- Threshold / flagged count: see `submission/outputs/threshold_meta.json`

---

## How to run

```bash
pip install -r requirements.txt
# unzip competition crops locally (not in git): images/images/*.jpg
# optional full retrain
python src/train_and_analyze.py --epochs 16 --batch-size 24 --latent-dim 512
# or reuse checkpoint
python src/train_and_analyze.py --skip-train
```

Then open `HiRISE_Anomaly_Detection.ipynb` (loads the checkpoint; cells already have outputs).

---

## Push to GitHub (important)

1. Create a **private** repo.
2. Add files and push (`images/`, `images.zip`, and `submission/figures/_eq/` are gitignored):

```bash
git add -A
git status
git commit -m "NSSC 2026 HiRISE anomaly detection submission"
git branch -M main
git remote add origin https://github.com/<USER>/<REPO>.git
git push -u origin main
```

3. Add organisers as collaborators:  
   `SarthakXSingh09`, `R15HV`, `Shivam3473`, `aditohates-bugs`, `Dhairya646`

4. Submit the **GitHub link** on Unstop (latest entry counts). Enter your **Team ID** (`T-xxxxxxx`) on Unstop.

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
