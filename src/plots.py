"""Plots written to submission/figures/ during / after training."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from dataset import HiRISEDataset


def make_figures(
    model,
    df: pd.DataFrame,
    meta: dict,
    Z: np.ndarray,
    device: str,
    img_dir: Path,
    crop_meta: Path,
    src_meta: Path,
    fig_dir: Path,
    res_dir: Path,
):
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="ticks", context="notebook")

    # 1) training curves
    hist = pd.read_csv(res_dir / "train_history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharex=True)
    axes[0].plot(hist["epoch"], hist["mse"], "o-", color="#1f4e79", ms=4, lw=1.5)
    axes[0].set_ylabel("MSE")
    axes[0].set_xlabel("epoch")
    axes[0].set_title("Reconstruction MSE")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(hist["epoch"], hist["loss"], "s-", color="#8B3A2F", ms=4, lw=1.5, zorder=3)
    ymax = float(hist["loss"].max())
    # Curriculum stages as shaded bands + legend (no rotated text on the curve)
    stages = [
        (0.5, 3.5, "#d9e6f2", "MSE warmup"),
        (3.5, 8.5, "#f3e4d8", "add SSIM+grad"),
        (8.5, 16.5, "#e8d9e6", "raise SSIM/grad"),
        (16.5, float(hist["epoch"].max()) + 0.5, "#dde8dd", "fine-tune edges"),
    ]
    for x0, x1, color, lab in stages:
        axes[1].axvspan(x0, x1, color=color, alpha=0.55, zorder=0, label=lab)
    for x in (3.5, 8.5, 16.5):
        axes[1].axvline(x, color="0.55", ls=":", lw=0.9, zorder=1)
    axes[1].set_ylim(0, ymax * 1.12)
    axes[1].set_ylabel("composite loss")
    axes[1].set_xlabel("epoch")
    axes[1].set_title("Composite loss (shaded = curriculum stage)")
    axes[1].legend(loc="lower right", fontsize=7, frameon=True, fancybox=False, edgecolor="0.7")
    axes[1].grid(True, alpha=0.3, zorder=2)
    fig.suptitle("Autoencoder training (v3)", y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "01_training_curve.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # 2) reconstructions
    ds = HiRISEDataset(img_dir, crop_meta, preload=False)
    idxs = np.random.RandomState(0).choice(len(ds), size=8, replace=False)
    fig, axes = plt.subplots(2, 8, figsize=(14, 4))
    model.eval()
    with torch.no_grad():
        for i, idx in enumerate(idxs):
            x, name, _ = ds[idx]
            recon, _ = model(x.unsqueeze(0).to(device))
            axes[0, i].imshow(x.squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
            axes[0, i].axis("off")
            axes[0, i].set_title(name.replace("sample_", "")[:8], fontsize=7)
            axes[1, i].imshow(recon.squeeze().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Original", fontsize=9)
    axes[1, 0].set_ylabel("Recon", fontsize=9)
    fig.suptitle("Original vs reconstruction")
    fig.tight_layout()
    fig.savefig(fig_dir / "02_reconstructions.png", dpi=160)
    plt.close(fig)

    # 3) t-SNE
    print("Running t-SNE (subsample 3000)...")
    rs = np.random.RandomState(42)
    sel = rs.choice(len(Z), size=min(3000, len(Z)), replace=False)
    emb = TSNE(
        n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=42
    ).fit_transform(StandardScaler().fit_transform(Z[sel]))
    fig, ax = plt.subplots(figsize=(6.5, 5.4))
    sc = ax.scatter(
        emb[:, 0], emb[:, 1], c=df.iloc[sel]["novelty_score"].values,
        s=5, cmap="magma", alpha=0.8, linewidths=0,
    )
    plt.colorbar(sc, ax=ax, fraction=0.046).set_label("novelty")
    ax.set_title("Latent t-SNE (n=3000). Colour = novelty, not class labels.")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()
    fig.savefig(fig_dir / "03_tsne_latent.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # 4) threshold
    nov = df["novelty_score"].values
    sorted_scores = np.sort(nov)[::-1]
    tau = meta["final_threshold"]
    elbow_thr = meta["elbow_threshold"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    axes[0].hist(nov, bins=80, color="#1f4e79", alpha=0.9)
    axes[0].axvline(elbow_thr, color="#2e7d32", ls=":", lw=1.8, label=f"elbow τ={elbow_thr:.3f}")
    axes[0].axvline(tau, color="#c0392b", ls="--", lw=2, label=f"final τ={tau:.3f} (MAD)")
    axes[0].set_xlabel("novelty (-score_samples)")
    axes[0].set_ylabel("count")
    axes[0].set_title("Score distribution")
    axes[0].legend(fontsize=8)
    axes[1].plot(sorted_scores, color="#1f4e79", lw=1.4)
    axes[1].axhline(elbow_thr, color="#2e7d32", ls=":", lw=1.8, label="elbow")
    axes[1].axhline(tau, color="#c0392b", ls="--", lw=2, label="final (MAD k=3.5)")
    axes[1].axvline(meta["elbow_idx"], color="#2e7d32", ls=":", lw=1)
    axes[1].axvline(meta["n_flagged"], color="#c0392b", ls="--", lw=1.2)
    axes[1].text(meta["n_flagged"] + 40, tau + 0.008, f"n={meta['n_flagged']}", color="#c0392b", fontsize=8)
    axes[1].set_xlabel("rank (high -> low novelty)")
    axes[1].set_ylabel("novelty")
    axes[1].set_title("Sorted scores")
    axes[1].legend(fontsize=8)
    fig.suptitle("Threshold selection (final cut is MAD, not elbow count)", y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "04_novelty_threshold.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # 5) metadata panels (lon in release is only 0/180)
    src = pd.read_csv(src_meta)
    m = df.merge(src, on="source_image_id", how="left")
    flg = m[m["is_anomaly"]]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    bins = np.linspace(-90, 90, 19)
    axes[0].hist(m["latitude"], bins=bins, color="0.75", density=True, label="all")
    axes[0].hist(flg["latitude"], bins=bins, color="#c0392b", alpha=0.65, density=True, label="flagged")
    axes[0].set_xlabel("latitude")
    axes[0].set_title("Latitude")
    axes[0].legend(fontsize=8)
    axes[1].hist(m["sun_angle"], bins=30, color="0.75", density=True, label="all")
    axes[1].hist(flg["sun_angle"], bins=30, color="#c0392b", alpha=0.65, density=True, label="flagged")
    axes[1].set_xlabel("sun_angle")
    axes[1].set_title("Sun angle")
    axes[1].legend(fontsize=8)
    seasons = sorted(m["season"].dropna().unique())
    all_c = m["season"].value_counts(normalize=True).reindex(seasons).fillna(0)
    flg_c = flg["season"].value_counts(normalize=True).reindex(seasons).fillna(0)
    x = np.arange(len(seasons))
    w = 0.38
    axes[2].bar(x - w / 2, all_c.values, w, color="0.75", label="all")
    axes[2].bar(x + w / 2, flg_c.values, w, color="#c0392b", alpha=0.85, label="flagged")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(seasons, rotation=25, ha="right", fontsize=8)
    axes[2].set_title("Season share")
    axes[2].legend(fontsize=8)
    fig.suptitle(
        "Acquisition metadata vs flags (source lon in release is only 0/180 deg)",
        y=1.05, fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "05_geo_anomalies.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    rng = np.random.RandomState(0)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.scatter(
        m["longitude"] + rng.uniform(-2.5, 2.5, len(m)), m["latitude"],
        s=8, c="0.7", alpha=0.35, label="all", linewidths=0,
    )
    ax.scatter(
        flg["longitude"] + rng.uniform(-2.5, 2.5, len(flg)), flg["latitude"],
        s=22, c="#c0392b", marker="x", label="flagged",
    )
    ax.set_xlabel("longitude (jittered; raw values are 0 or 180)")
    ax.set_ylabel("latitude")
    ax.set_title("Source coordinates in the provided metadata")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "05b_source_coords.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # 5c) optional metadata fusion vs image-only
    if "novelty_score_fused" in df.columns and "is_anomaly_fused" in df.columns:
        img = df["is_anomaly"].astype(bool)
        fus = df["is_anomaly_fused"].astype(bool)
        both = int((img & fus).sum())
        only_img = int((img & ~fus).sum())
        only_fus = int((~img & fus).sum())
        n_img, n_fus = int(img.sum()), int(fus.sum())
        thr_f = float(meta.get("fused_threshold_gauss_2p5", np.nan))

        fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6))
        vals = [n_img, n_fus, both]
        bars = axes[0].bar(
            ["Image-only\n(MAD τ)", "Fused\n(μ+2.5σ)", "Overlap"],
            vals,
            color=["#1f4e79", "#8B3A2F", "#5a7a5a"],
            width=0.65,
        )
        for b, v in zip(bars, vals):
            axes[0].text(b.get_x() + b.get_width() / 2, v + 4, str(v), ha="center", fontsize=9)
        axes[0].set_ylabel("flagged crops")
        axes[0].set_title("Flag set sizes")
        axes[0].set_ylim(0, max(vals) * 1.18)
        axes[0].grid(True, axis="y", alpha=0.3)

        axes[1].pie(
            [both, only_img, only_fus],
            labels=[f"both ({both})", f"image only ({only_img})", f"fused only ({only_fus})"],
            colors=["#5a7a5a", "#1f4e79", "#8B3A2F"],
            startangle=90,
            textprops={"fontsize": 8},
        )
        axes[1].set_title("Agreement of flags")

        rng2 = np.random.RandomState(0)
        idx = rng2.choice(len(df), size=min(4000, len(df)), replace=False)
        s = df.iloc[idx]
        axes[2].scatter(s["novelty_score"], s["novelty_score_fused"], s=6, c="0.65", alpha=0.35, linewidths=0)
        fl = df[img | fus]
        axes[2].scatter(
            fl["novelty_score"],
            fl["novelty_score_fused"],
            s=14,
            c=np.where(fl["is_anomaly"] & fl["is_anomaly_fused"], "#5a7a5a", "#c45c26"),
            alpha=0.75,
            linewidths=0,
        )
        lim0 = min(df["novelty_score"].min(), df["novelty_score_fused"].min())
        lim1 = max(df["novelty_score"].max(), df["novelty_score_fused"].max())
        axes[2].plot([lim0, lim1], [lim0, lim1], "k--", lw=0.8, alpha=0.5)
        axes[2].axvline(meta["final_threshold"], color="#1f4e79", ls=":", lw=1)
        if np.isfinite(thr_f):
            axes[2].axhline(thr_f, color="#8B3A2F", ls=":", lw=1)
        axes[2].set_xlabel("novelty (image latent)")
        axes[2].set_ylabel("novelty (fused)")
        axes[2].set_title("Score comparison")
        axes[2].grid(True, alpha=0.25)
        fig.suptitle("Optional metadata fusion vs image-latent baseline", fontsize=11, y=1.02)
        fig.tight_layout()
        fig.savefig(fig_dir / "07_fusion_compare.png", dpi=170, bbox_inches="tight")
        plt.close(fig)

    # 6) heatmaps
    top = df[df["is_anomaly"]].sort_values("novelty_score", ascending=False).head(5)
    if len(top) == 0:
        top = df.sort_values("novelty_score", ascending=False).head(5)
    fig, axes = plt.subplots(len(top), 3, figsize=(9, 3 * len(top)))
    if len(top) == 1:
        axes = np.array([axes])
    name_to_idx = {n: i for i, n in enumerate(ds.filenames)}
    with torch.no_grad():
        for row_i, (_, row) in enumerate(top.iterrows()):
            x, _, _ = ds[name_to_idx[row["filename"]]]
            recon, _ = model(x.unsqueeze(0).to(device))
            x_np = x.squeeze().numpy()
            recon_np = recon.squeeze().cpu().numpy()
            err = np.abs(x_np - recon_np)
            axes[row_i, 0].imshow(x_np, cmap="gray", vmin=0, vmax=1)
            axes[row_i, 0].set_title(f"{row['filename']}\nnov={row['novelty_score']:.3f}", fontsize=8)
            axes[row_i, 1].imshow(recon_np, cmap="gray", vmin=0, vmax=1)
            axes[row_i, 1].set_title("Reconstruction", fontsize=8)
            im = axes[row_i, 2].imshow(err, cmap="inferno")
            axes[row_i, 2].set_title("|Error|", fontsize=8)
            for ax_ in axes[row_i]:
                ax_.axis("off")
            fig.colorbar(im, ax=axes[row_i, 2], fraction=0.046)
    fig.suptitle("Top flagged anomalies — pixel |error| heatmaps", y=1.01)
    fig.tight_layout()
    fig.savefig(fig_dir / "06_top5_heatmaps.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    top.to_csv(res_dir / "top5_anomalies.csv", index=False)
    return top
