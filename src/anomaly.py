"""Isolation Forest novelty scoring + thresholding + optional fusions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def apply_threshold(df: pd.DataFrame, score_col: str = "novelty_score"):
    """
    Statistical cutoff on a score column (higher = more anomalous).
    Tries elbow → MAD k=3.5 → μ+2.5σ; contamination never sets the count.
    """
    novelty = df[score_col].to_numpy(dtype=np.float64)
    med = np.median(novelty)
    mad = np.median(np.abs(novelty - med)) + 1e-12
    robust_z = 0.6745 * (novelty - med) / mad
    threshold_robust = med + (3.5 / 0.6745) * mad
    mu, sigma = float(novelty.mean()), float(novelty.std())
    thresh_gauss = mu + 2.5 * sigma

    sorted_scores = np.sort(novelty)[::-1]
    xs = np.arange(len(sorted_scores), dtype=np.float64)
    x1, y1 = 0.0, float(sorted_scores[0])
    x2, y2 = float(len(sorted_scores) - 1), float(sorted_scores[-1])
    denom = np.hypot(x2 - x1, y2 - y1) + 1e-12
    dists = np.abs((y2 - y1) * xs - (x2 - x1) * sorted_scores + x2 * y1 - y2 * x1) / denom
    elbow_idx = int(np.argmax(dists[: max(50, len(sorted_scores) // 20)]))
    elbow_threshold = float(sorted_scores[elbow_idx])

    candidates = [
        ("elbow", elbow_threshold),
        ("robust", float(threshold_robust)),
        ("gauss_2p5", float(thresh_gauss)),
    ]
    final_threshold = float(thresh_gauss)
    chosen = "gauss_2p5"
    for name, thr in candidates:
        n = int((novelty >= thr).sum())
        frac = n / len(novelty)
        if 0.005 <= frac <= 0.03:
            final_threshold, chosen = float(thr), name
            break

    n_flag = int((novelty >= final_threshold).sum())
    if n_flag < 5:
        final_threshold = float(np.quantile(novelty, 0.995))
        chosen = "quantile_99.5"
        n_flag = int((novelty >= final_threshold).sum())
    if n_flag > 0.05 * len(novelty):
        final_threshold = float(np.quantile(novelty, 0.99))
        chosen = "quantile_99"
        n_flag = int((novelty >= final_threshold).sum())

    out = df.copy()
    out["is_anomaly"] = out[score_col] >= final_threshold
    out["robust_z"] = robust_z
    meta = {
        "median": float(med),
        "mad": float(mad),
        "mu": mu,
        "sigma": sigma,
        "elbow_idx": elbow_idx,
        "elbow_threshold": elbow_threshold,
        "robust_threshold": float(threshold_robust),
        "gauss_2p5": float(thresh_gauss),
        "final_threshold": float(final_threshold),
        "threshold_rule": chosen,
        "n_flagged": int(out["is_anomaly"].sum()),
        "k_modified_z": 3.5,
        "isolation_forest_contamination": "auto",
        "score_col": score_col,
    }
    return out, meta


def recon_hybrid_scores(
    novelty: np.ndarray,
    recon_mse: np.ndarray,
    w_if: float = 0.75,
    w_recon: float = 0.25,
) -> np.ndarray:
    """z-score blend of IF novelty and reconstruction MSE (optional fusion)."""
    n = (novelty - novelty.mean()) / (novelty.std() + 1e-12)
    r = (recon_mse - recon_mse.mean()) / (recon_mse.std() + 1e-12)
    return w_if * n + w_recon * r


def novelty_pipeline(Z: np.ndarray, names: list, srcs: list, recon_errs: np.ndarray):
    """
    Fit Isolation Forest on latents.
    Novelty = -score_samples  (higher => more anomalous).
    Final threshold from MAD / elbow / mu+k*sigma — not from contamination.
    """
    scaler = StandardScaler()
    Zs = scaler.fit_transform(Z)

    iso = IsolationForest(
        n_estimators=600,
        max_samples="auto",
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(Zs)
    novelty = -iso.score_samples(Zs)

    df = pd.DataFrame(
        {
            "filename": names,
            "source_image_id": srcs,
            "novelty_score": novelty,
            "recon_mse": recon_errs,
        }
    )
    df, meta = apply_threshold(df, score_col="novelty_score")
    return df, meta, scaler, iso


def metadata_fusion(Z: np.ndarray, df: pd.DataFrame, src_meta_path: str | Path):
    """Concat image latent with acquisition metadata; re-run Isolation Forest."""
    src = pd.read_csv(src_meta_path)
    merged = df.merge(src, on="source_image_id", how="left")
    season_oh = pd.get_dummies(merged["season"].fillna("UNK"), prefix="season")
    meta_feats = np.column_stack(
        [
            (merged["sun_angle"].fillna(merged["sun_angle"].mean()).values - 45) / 45.0,
            merged["latitude"].fillna(0).values / 90.0,
            merged["longitude"].fillna(0).values / 180.0,
            (merged["resolution"].fillna(0.5).values - 0.25) / 0.75,
            season_oh.values,
        ]
    )
    fused = np.concatenate(
        [StandardScaler().fit_transform(Z), StandardScaler().fit_transform(meta_feats)],
        axis=1,
    )
    iso = IsolationForest(n_estimators=600, contamination="auto", random_state=42, n_jobs=-1)
    iso.fit(fused)
    merged["novelty_score_fused"] = -iso.score_samples(fused)
    return merged, season_oh.columns.tolist()
