"""
Train the ConvAE, extract latents, score anomalies, write figures/CSVs.

  python src/train_and_analyze.py --epochs 16 --batch-size 24 --latent-dim 512
  python src/train_and_analyze.py --skip-train
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from anomaly import metadata_fusion, novelty_pipeline
from dataset import HiRISEDataset
from model import ConvAutoencoder, combined_loss
from plots import make_figures

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "images" / "images"
CROP_META = ROOT / "crop_metadata_index.csv"
SRC_META = ROOT / "source_image_metadata.csv"
OUT = ROOT / "submission"
FIG = OUT / "figures"
CKPT = OUT / "checkpoints"
RES = OUT / "outputs"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_autoencoder(
    latent_dim: int = 512,
    epochs: int = 18,
    batch_size: int = 32,
    lr: float = 2e-4,
    device: str = "cuda",
):
    ds = HiRISEDataset(IMG_DIR, CROP_META, preload=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    model = ConvAutoencoder(latent_dim=latent_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history = []
    model.train()
    for epoch in range(1, epochs + 1):
        if epoch <= 3:
            alpha, beta, gamma = 1.0, 0.0, 0.0
        elif epoch <= 8:
            alpha, beta, gamma = 0.7, 0.2, 0.1
        else:
            alpha, beta, gamma = 0.55, 0.30, 0.15

        losses, mses = [], []
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
        for x, _, _ in pbar:
            x = x.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            recon, _ = model(x)
            mse = F.mse_loss(recon, x)
            loss = combined_loss(recon, x, alpha=alpha, beta=beta, gamma=gamma)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
            mses.append(float(mse.detach()))
            pbar.set_postfix(loss=float(np.mean(losses[-20:])), mse=float(np.mean(mses[-20:])))
        sched.step()
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "mse": float(np.mean(mses)),
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
            }
        )
        print(f"Epoch {epoch}: loss={history[-1]['loss']:.6f} mse={history[-1]['mse']:.6f}")

    CKPT.mkdir(parents=True, exist_ok=True)
    # float16 weights keep the file under GitHub's 100 MB limit
    sd = {
        k: (v.half() if v.is_floating_point() else v).cpu()
        for k, v in model.state_dict().items()
    }
    torch.save(
        {
            "model_state": sd,
            "latent_dim": latent_dim,
            "history": history,
            "dtype": "float16",
            "alpha": 0.40,
            "beta": 0.35,
            "gamma": 0.25,
        },
        CKPT / "cae_v3.pt",
    )
    RES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(RES / "train_history.csv", index=False)
    return model, history


def load_checkpoint(path: Path, device: str) -> ConvAutoencoder:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ConvAutoencoder(latent_dim=ckpt["latent_dim"]).to(device)
    sd = ckpt["model_state"]
    if ckpt.get("dtype") == "float16":
        sd = {k: v.float() if v.is_floating_point() else v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    return model


@torch.no_grad()
def extract_latents(model, device, batch_size=128):
    ds = HiRISEDataset(IMG_DIR, CROP_META, preload=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    zs, names, srcs, recon_errs = [], [], [], []
    for x, name, src in tqdm(loader, desc="Extract latents"):
        x = x.to(device)
        recon, z = model(x)
        err = ((recon - x) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
        zs.append(z.cpu().numpy())
        names.extend(name)
        srcs.extend(src)
        recon_errs.append(err)
    return np.concatenate(zs), names, srcs, np.concatenate(recon_errs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    set_seed(42)
    for p in (OUT, FIG, CKPT, RES):
        p.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    if args.skip_train and (CKPT / "cae_v3.pt").exists():
        model = load_checkpoint(CKPT / "cae_v3.pt", device)
    else:
        model, _ = train_autoencoder(
            latent_dim=args.latent_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        )

    Z, names, srcs, recon_errs = extract_latents(model, device)
    np.save(RES / "latents.npy", Z)

    df, meta, _, _ = novelty_pipeline(Z, names, srcs, recon_errs)
    fused_df, _ = metadata_fusion(Z, df, SRC_META)

    nov_f = fused_df["novelty_score_fused"].values
    thr_f = float(nov_f.mean() + 2.5 * nov_f.std())
    fused_df["is_anomaly_fused"] = nov_f >= thr_f
    meta["n_flagged_fused"] = int(fused_df["is_anomaly_fused"].sum())
    meta["fused_threshold_gauss_2p5"] = thr_f
    meta["flag_overlap_image_vs_fused"] = int(
        ((fused_df["is_anomaly"]) & (fused_df["is_anomaly_fused"])).sum()
    )

    src = pd.read_csv(SRC_META)
    out_df = fused_df.merge(src, on="source_image_id", how="left", suffixes=("", "_dup"))
    out_df = out_df.loc[:, ~out_df.columns.str.endswith("_dup")]
    out_df.to_csv(RES / "scored_crops.csv", index=False)
    (RES / "threshold_meta.json").write_text(json.dumps(meta, indent=2))

    top = make_figures(
        model, df, meta, Z, device,
        img_dir=IMG_DIR, crop_meta=CROP_META, src_meta=SRC_META,
        fig_dir=FIG, res_dir=RES,
    )
    print("Flagged:", meta["n_flagged"], "threshold:", meta["final_threshold"], "rule:", meta["threshold_rule"])
    print(top[["filename", "novelty_score", "source_image_id", "recon_mse"]])
    print("Done.", OUT)


if __name__ == "__main__":
    main()
