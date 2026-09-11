# Model changelog

Evolution of the HiRISE anomaly pipeline. Each entry follows **symptom → diagnosis → fix**.

---

## v1 — shallow ConvAE, MSE, latent 128

**Setup:** 3 stride-2 blocks, latent 128, MSE only, ~6 epochs.

**Symptom:** Recons of dunes/crater rims were mush. Error maps were flat. t-SNE looked like one blob.

**Diagnosis:** MSE averages away high-frequency texture. 128-D was too tight for 227² crops. Model underfit the normal class, so everything looked “weird” to the Isolation Forest.

**Fix:** deeper encoder, larger bottleneck, add SSIM.

---

## v2 — deeper net, MSE+SSIM, latent 256

**Setup:** 5 conv stages, reflection pad for 227, latent 256, `0.8 MSE + 0.2 (1-SSIM)`, AdamW.

**Symptom:** Better global structure, still soft on ridgelines. Novelty scores were right-skewed; a naive μ+2σ cutoff felt shaky. A few obviously odd crops sat in the middle of the ranking.

**Diagnosis:** SSIM weight too low. Threshold story leaned on Gaussian tails that the histogram doesn’t really support. Metadata unused.

**Fix:** stronger structure/edge losses, train more carefully, threshold from elbow + MAD as well as μ+kσ, try metadata fusion.

---

## v3 — spatial bottleneck + MAD cut

**Symptom (carried from v2):** AdaptiveAvgPool erased spatial layout → soft decodes; threshold needed a non-top-N rule independent of contamination.

**Diagnosis:** Need a 14×14 feature map before the FC, a larger latent, edge-aware loss without pretrained nets, and a MAD/elbow threshold story.

**Fix:**
- Architecture: keep **14×14** map (squeeze to 128 ch → FC), nearest upsample + double conv decoder, **latent_dim = 512**.
- Loss curriculum: MSE warmup → `0.55/0.30/0.15` → fine-tune `0.40/0.35/0.25`. Grad = finite differences (no VGG/ResNet).
- Anomaly: IF on latents; `novelty = -score_samples`; final cut = **MAD k=3.5**; metadata join + optional fusion.

**Outcome:** usable heatmaps, ~242 / 10422 flags from the score distribution.

---

## v4 — edge fine-tune + recon hybrid (current)

**Symptom:** After the deadline extension we audited the code: the saved v3 *weights* had seen the `0.40/0.35/0.25` fine-tune stage, but the training script had regressed and would stop at `0.55/0.30/0.15` on any retrain. Ridgelines in heatmaps were still softer than ideal; IF alone ignored strong reconstruction failures that looked visually wrong.

**Diagnosis:** Need (1) a real continue-train path at the documented edge weights, (2) optional blend of IF novelty with reconstruction MSE as a comparative fusion (problem allows score fusion), (3) a stabler IF ensemble.

**Fix:**
- Restore full curriculum in code; add `--finetune` to continue from `cae_v3.pt` at **lr=5e-5**, loss **0.40 MSE + 0.35 (1−SSIM) + 0.25 ∇L1**, 8 more epochs → `cae_v4.pt`.
- Isolation Forest **600** trees (was 400).
- Optional hybrid score: `0.75·z(IF) + 0.25·z(recon_mse)` with its own MAD/elbow cut (reported comparatively; **primary remains IF+MAD**).
- Metadata fusion unchanged as a second comparative check.

**Outcome:** continue-train lowered reconstruction MSE further (0.00265 → 0.0023). After re-score with the restored pipeline: primary MAD flags **161 / 10422** (1.5%). Recon hybrid **65**; metadata fusion **266**. Official answers remain IF-latent + MAD.

**Evidence :** `submission/outputs/train_history.csv`, `threshold_meta.json`, figures `01`–`07`, and the PS-numbered notebook.

---

## Short justifications

| Choice | Reason |
|--------|--------|
| ConvAE not VAE | Don’t need a KL term for IF; fewer knobs |
| latent 512 | Fidelity vs separability after 128/256 looked soft |
| MSE+SSIM+grad | structure + edges; no pretrained perceptual net |
| IF on latents | Cheap; already compressed geology |
| elbow / MAD / μ+kσ | Defend the cut without assuming contamination |
| recon hybrid (optional) | Catches high-error crops IF ranks mid; not the official cut |
| Notebook PS headings | Rules require markdown mapped to each question/sub-question |
