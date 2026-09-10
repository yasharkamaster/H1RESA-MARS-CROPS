# Model changelog

Evolution of the HiRISE anomaly pipeline across three versions. Each entry follows **symptom → diagnosis → fix**.

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

## v3 — submitted

**Symptom (carried from v2):** AdaptiveAvgPool(4/8) erased spatial layout → soft decodes; threshold needed a non-top-N rule independent of contamination.

**Diagnosis:** Need a 14×14 feature map before the FC, a larger latent, edge-aware loss without pretrained nets, and a MAD/elbow threshold story.

**Fix:**
- Architecture: keep **14×14** map (squeeze to 128 ch → FC), nearest upsample + double conv decoder, **latent_dim = 512**.
- Loss: MSE warmup → `0.55/0.30/0.15` MSE+SSIM+grad → fine-tune `0.40/0.35/0.25`. Grad = finite differences (no VGG/ResNet).
- Training: 16 epochs + 5 fine-tune, batch 24, AdamW 2e-4, cosine, grad clip 1.0.
- Anomaly: IF on latents, `contamination='auto'`; `novelty = -score_samples`; final cut = **MAD k=3.5** (stricter than elbow ~517 → **242** flags); top-5 heatmaps only after that cut; metadata join + optional fusion; lon in release is only 0°/180° so location plots use lat/sun_angle/season.

**Outcome:** sharper than v1/v2, usable heatmaps, flag count from the score distribution (~2.3% of crops).

---

## Short justifications

| Choice | Reason |
|--------|--------|
| ConvAE not VAE | Don’t need a KL term for IF; fewer knobs under time pressure |
| latent 512 | Fidelity vs separability after 128/256 looked soft |
| MSE+SSIM+grad | structure + edges; no pretrained perceptual net |
| IF on latents | Cheap; already compressed geology |
| elbow / MAD / μ+kσ | Defend the cut without assuming a contamination rate |
