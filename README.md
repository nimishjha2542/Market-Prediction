## Project Overview

This project delivers a production-ready system that decides whether to accept or skip each trading opportunity to maximize a risk-aware utility (profit with controlled volatility). I engineered a Supervised Autoencoder + MLP (AE-MLP) that learns denoised, predictive representations from noisy market features and converts them into high-precision trade actions. The model is trained jointly per time-series fold with a multi-task objective—reconstruction (MSE) plus auxiliary and main classification (BCE)—capturing stable structure while staying sharply aligned to directional returns.

To ensure robustness, the pipeline applies purged, grouped time-series cross-validation with a 31-day embargo, preventing temporal leakage and mirroring real-world deployment. Features are imputed causally (forward-fill, then zero), and training is economically weighted toward samples with larger absolute returns, focusing learning where P&L impact is highest.

At inference, I average multi-horizon predictions (and optionally multiple seeds) and apply a calibrated threshold to deliver stable, high-precision actions. Model selection uses validation AUC on the main head, directly supporting post-threshold decision quality under the competition’s utility metric.

**Tech stack:** TensorFlow/Keras, scikit-learn, pandas/numpy, datatable; time-aware CV and ML ops patterns suitable for productionization.

## Dataset Description

**What a row represents.** Each row is a **trading opportunity** for which the model decides whether to **take** the trade (`action = 1`) or **skip** it (`action = 0`).

**Core fields**
- **Anonymized features:** `feature_0 … feature_129` — 130 numeric signals derived from real market data.  
- **Targets / returns (train only):** `resp`, `resp_1`, `resp_2`, `resp_3`, `resp_4` — realized returns at different horizons.  
- **Scoring weight:** `weight` — row importance in the competition’s utility. Rows with `weight = 0` are included for completeness but **do not** affect the score.  
- **Time columns:**  
  - `date` — trading-day index (used to group rows by day).  
  - `ts_id` — within-day order (ensures causal predictions).

**Scale.** ~**1 million** training rows (historical data). The hidden test set was served via a time-series API and updated during the live phase.

### Schema at a Glance

| Column(s)                       | Type  | Train | Test | Description |
|---|---|:--:|:--:|---|
| `feature_0 … feature_129`      | float | ✓ | ✓ | Anonymized market features used as model inputs. |
| `date`                         | int   | ✓ | ✓ | Trading-day index; used for grouped, time-aware validation and day-level scoring. |
| `ts_id`                        | int   | ✓ | ✓ | Within-day order to preserve causality. |
| `weight`                       | float | ✓ | ✓ | Row importance in the utility score; `weight = 0` rows don’t contribute to scoring. |
| `resp`                         | float | ✓ |     | Primary realized return (training signal). |
| `resp_1, resp_2, resp_3, resp_4` | float | ✓ |     | Returns at alternative horizons (auxiliary training signals). |


## Methodology & Approach

I designed and implemented an end-to-end, production-style pipeline that converts anonymized market features into calibrated trade decisions. The approach emphasizes **leakage control**, **economic alignment**, and **stability**—properties that matter in real trading as much as they do on a leaderboard.

---

### Utility Metric

For each trading date \(i\):
\[
p_i=\sum_{j}\big(\text{weight}_{ij}\cdot \text{resp}_{ij}\cdot \text{action}_{ij}\big),\qquad
t=\frac{\sum_i p_i}{\sqrt{\sum_i p_i^{2}}}\cdot\sqrt{\frac{250}{|i|}},\qquad
u=\min(\max(t,0),6)\cdot\sum_i p_i.
\]

**Objective:** maximize aggregate profit \(\sum_i p_i\) while maintaining low day-to-day variance (high \(t\)).

---

### Architecture

flowchart TB
    %% Inputs & normalization
    X["Inputs: feature_0..129"] --> BN0["BatchNorm (x0)"]

    %% Autoencoder (trained jointly per fold)
    subgraph AE["Autoencoder Path"]
        BN0 --> GN["GaussianNoise (sigma ~= 0.0353)"]
        GN --> ENC["Encoder<br/>Dense 96 -> BN -> Swish"]
        ENC --> DEC["Decoder<br/>Dense -> Reconstruct X (130-d)"]
        DEC --> AECLS["AE Classifier<br/>Dense 96 -> BN -> Swish -> Dropout -> Dense 5 (Sigmoid)"]
    end

    %% Main classifier (concat raw + latent)
    ENC -->|latent z (96)| CAT(("Concat"))
    BN0 -->|x0 (130)| CAT
    CAT --> H1["Dense 896 -> BN -> Swish -> Dropout"]
    H1 --> H2["Dense 448 -> BN -> Swish -> Dropout"]
    H2 --> H3["Dense 448 -> BN -> Swish -> Dropout"]
    H3 --> H4["Dense 256 -> BN -> Swish -> Dropout"]
    H4 --> MAIN["Main Classifier<br/>Dense 5 (Sigmoid)"]

    %% Losses & selection
    DEC --> LREC["Reconstruction: MSE, metric: MAE"]
    AECLS --> LAE["AE Head: BCE, metric: AUC"]
    MAIN --> LMAIN["Main Head: BCE, metric: AUC, monitored for ES/CKPT"]

    classDef block fill:#f8fafc,stroke:#334155,stroke-width:1px;
    classDef loss  fill:#eef2ff,stroke:#1e40af,stroke-width:1px;
    class BN0,GN,ENC,DEC,AECLS,H1,H2,H3,H4,MAIN,CAT block;
    class LREC,LAE,LMAIN loss;



