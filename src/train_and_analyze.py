"""
Train the ConvAE, extract latents, score anomalies, write figures/CSVs.

  python src/train_and_analyze.py --epochs 21 --batch-size 24 --latent-dim 512
  python src/train_and_analyze.py --finetune --epochs 8 --lr 5e-5
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

from anomaly import apply_threshold, metadata_fusion, novelty_pipeline, recon_hybrid_scores
from dataset import HiRISEDataset
from model import ConvAutoencoder, combined_loss
from plots import make_figures

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "images" / "images"
CROP_META = ROOT / "data" / "crop_metadata_index.csv"
SRC_META = ROOT / "data" / "source_image_metadata.csv"
OUT = ROOT / "submission"
FIG = OUT / "figures"
CKPT = OUT / "checkpoints"
RES = OUT / "outputs"
CKPT_NAME = "cae_v4.pt"
CKPT_FALLBACK = "cae_v3.pt"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def loss_weights(epoch: int, total_epochs: int) -> tuple[float, float, float]:
    """Curriculum matching the documented v3/v4 schedule."""
    if epoch <= 3:
        return 1.0, 0.0, 0.0
    if epoch <= 8:
        return 0.7, 0.2, 0.1
    # last 5 epochs (or remaining) = edge fine-tune
    if epoch > max(8, total_epochs - 5):
        return 0.40, 0.35, 0.25
    return 0.55, 0.30, 0.15


def _save_checkpoint(model, latent_dim, history, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
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
            "version": "v4",
        },
        path,
    )


def train_autoencoder(
    latent_dim: int = 512,
    epochs: int = 21,
    batch_size: int = 24,
    lr: float = 2e-4,
    device: str = "cuda",
    model: ConvAutoencoder | None = None,
    start_epoch: int = 1,
    finetune_only: bool = False,
    history_prev: list | None = None,
):
    ds = HiRISEDataset(IMG_DIR, CROP_META, preload=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    if model is None:
        model = ConvAutoencoder(latent_dim=latent_dim).to(device)
    else:
        model = model.to(device)
        latent_dim = model.latent_dim

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    end_epoch = start_epoch + epochs - 1
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    history = list(history_prev or [])
    model.train()
    for epoch in range(start_epoch, end_epoch + 1):
        if finetune_only:
            alpha, beta, gamma = 0.40, 0.35, 0.25
        else:
            alpha, beta, gamma = loss_weights(epoch, end_epoch)

        losses, mses = [], []
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{end_epoch}")
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
        print(
            f"Epoch {epoch}: loss={history[-1]['loss']:.6f} "
            f"mse={history[-1]['mse']:.6f} "
            f"w=({alpha:.2f},{beta:.2f},{gamma:.2f})"
        )

    _save_checkpoint(model, latent_dim, history, CKPT / CKPT_NAME)
    # keep a copy at the name the notebook already knows
    _save_checkpoint(model, latent_dim, history, CKPT / CKPT_FALLBACK)
    RES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(RES / "train_history.csv", index=False)
    return model, history


def load_checkpoint(path: Path, device: str) -> tuple[ConvAutoencoder, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ConvAutoencoder(latent_dim=ckpt["latent_dim"]).to(device)
    sd = ckpt["model_state"]
    if ckpt.get("dtype") == "float16":
        sd = {k: v.float() if v.is_floating_point() else v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    return model, ckpt


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
    parser.add_argument("--epochs", type=int, default=21)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Continue from cae_v3/v4 with edge-heavy loss (0.40/0.35/0.25)",
    )
    args = parser.parse_args()

    set_seed(42)
    for p in (OUT, FIG, CKPT, RES):
        p.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    prefer = CKPT / CKPT_NAME if (CKPT / CKPT_NAME).exists() else CKPT / CKPT_FALLBACK

    if args.skip_train and prefer.exists():
        model, _ = load_checkpoint(prefer, device)
    elif args.finetune:
        src_ckpt = CKPT / CKPT_FALLBACK if (CKPT / CKPT_FALLBACK).exists() else prefer
        if not src_ckpt.exists():
            raise FileNotFoundError(f"No checkpoint to fine-tune at {src_ckpt}")
        model, ckpt = load_checkpoint(src_ckpt, device)
        hist_prev = ckpt.get("history") or []
        start = int(hist_prev[-1]["epoch"]) + 1 if hist_prev else 22
        print(f"Fine-tuning from {src_ckpt.name} starting epoch {start}")
        model, _ = train_autoencoder(
            latent_dim=args.latent_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            model=model,
            start_epoch=start,
            finetune_only=True,
            history_prev=hist_prev,
        )
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

    # Optional recon fusion (comparative + can become primary if MAD band OK)
    hybrid = recon_hybrid_scores(df["novelty_score"].values, df["recon_mse"].values)
    df["novelty_score_hybrid"] = hybrid
    df_h, meta_h = apply_threshold(df.copy(), score_col="novelty_score_hybrid")
    meta["hybrid_threshold"] = meta_h["final_threshold"]
    meta["hybrid_threshold_rule"] = meta_h["threshold_rule"]
    meta["n_flagged_hybrid"] = meta_h["n_flagged"]
    meta["flag_overlap_if_vs_hybrid"] = int(
        ((df["is_anomaly"]) & (df_h["is_anomaly"])).sum()
    )
    # Keep IF+MAD as official primary; record hybrid for the report
    df["is_anomaly_hybrid"] = df_h["is_anomaly"]

    fused_df, _ = metadata_fusion(Z, df, SRC_META)
    nov_f = fused_df["novelty_score_fused"].values
    thr_f = float(nov_f.mean() + 2.5 * nov_f.std())
    fused_df["is_anomaly_fused"] = nov_f >= thr_f
    meta["n_flagged_fused"] = int(fused_df["is_anomaly_fused"].sum())
    meta["fused_threshold_gauss_2p5"] = thr_f
    meta["flag_overlap_image_vs_fused"] = int(
        ((fused_df["is_anomaly"]) & (fused_df["is_anomaly_fused"])).sum()
    )
    meta["model_version"] = "v4"

    src = pd.read_csv(SRC_META)
    out_df = fused_df.merge(src, on="source_image_id", how="left", suffixes=("", "_dup"))
    out_df = out_df.loc[:, ~out_df.columns.str.endswith("_dup")]
    out_df.to_csv(RES / "scored_crops.csv", index=False)
    (RES / "threshold_meta.json").write_text(json.dumps(meta, indent=2))

    top = make_figures(
        model,
        df,
        meta,
        Z,
        device,
        img_dir=IMG_DIR,
        crop_meta=CROP_META,
        src_meta=SRC_META,
        fig_dir=FIG,
        res_dir=RES,
    )
    print(
        "Flagged:",
        meta["n_flagged"],
        "threshold:",
        meta["final_threshold"],
        "rule:",
        meta["threshold_rule"],
    )
    print("Hybrid flagged:", meta["n_flagged_hybrid"], "overlap:", meta["flag_overlap_if_vs_hybrid"])
    print(top[["filename", "novelty_score", "source_image_id", "recon_mse"]])
    print("Done.", OUT)


if __name__ == "__main__":
    main()
