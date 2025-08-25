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

## Utility Metric

For each date $i$, define daily profit:

$$
p_i = \sum_{j}\left(\mathrm{weight}_{ij}\cdot \mathrm{resp}_{ij}\cdot \mathrm{action}_{ij}\right).
$$

Annualized Sharpe-like factor:

$$
t = \frac{\sum_{i} p_i}{\sqrt{\sum_{i} p_i^{2}}}\,\sqrt{\frac{250}{|i|}} \, .
$$

Utility:

$$
u = \min\!\big(\max(t,0),6\big)\,\sum_{i} p_i \, .
$$

**Objective:** maximize $\sum_{i} p_i$ while maintaining low day-to-day variance (high $t$).




---

### Architecture



![AE-MLP Architecture](assets/architecture.png)

## Process

### 1) Load & filter
- Load `train.csv`; keep `date > 85` (remove early-regime shift) and `weight > 0` (rows that actually move the utility).
- Extract `feature_0..feature_129`, time columns (`date`, `ts_id`), and returns (`resp`, `resp_1..resp_4`).

### 2) Causal imputation
- Forward-fill each feature using only past rows (preserves causality).
- Fill residual missing values with `0` to ensure dense, consistent tensors for train and inference.

### 3) Supervision (labels)
- Build per-horizon binary labels: \( y_k = \mathbf{1}[\mathrm{resp}_k > 0] \), for \( k \in \{\mathrm{resp}, \mathrm{resp}_1, \mathrm{resp}_2, \mathrm{resp}_3, \mathrm{resp}_4\} \).
- *(Optional, analysis)* Define a strict composite label equal to \(1\) iff **all** horizons are positive.

### 4) Cross-validation design
- Use **5-fold Purged Group TimeSeriesSplit** (group = `date`) with a **31-day embargo** between train/validation windows.
- Generate **OOF predictions** per fold to support threshold tuning on held-out data.

### 5) Joint training (per fold)
- Train a **Supervised Autoencoder + MLP** within each fold so the encoder and classifier co-adapt **without cross-fold leakage**.
  - **Encoder:** Dense 96 → BatchNorm → Swish, preceded by GaussianNoise (\( \sigma \approx 0.0353 \)).
  - **Decoder:** reconstructs the 130-D input (**MSE** loss).
  - **AE auxiliary head:** Dense → BatchNorm → Swish → Dropout → **5 sigmoid** (**BCE**).
  - **Main classifier:** concat **BN(input)** with **latent (96-D)** → MLP \([896, 448, 448, 256]\) with BN/Swish/Dropout → **5 sigmoid** (**BCE**).
- **Optimization:** Adam (lr \(=10^{-3}\)), label smoothing \(=0\), batch size \(\approx 4096\), up to \(100\) epochs; **EarlyStopping** (patience \(\approx 10\)) and **ModelCheckpoint** on **val AUC (main head)**.
- **Independent problem-solving:** Identified potential label leakage from pretraining and **replaced it with joint per-fold training**, eliminating cross-fold information bleed.

### 6) Economics-aware sample weighting
- Compute per-row training weight: \( w_{\text{sample}} = \mathrm{mean}(|\mathrm{resp}_k|),\; k=1..5 \), and apply uniformly across outputs to emphasize **high-impact** observations.

### 7) Model selection & artifacts
- Select the **best epoch** by **validation AUC (main head)**; persist **best weights** per fold.
- *(Optional)* Prefer models from **later folds** or use **multi-seed** training to increase stability.

### 8) Inference & probability aggregation
- For each segment, output **five horizon probabilities** from the main head.
- **Average** across horizons; *(optional)* also average across **seeds/folds** to reduce variance.

### 9) Thresholding → actions
- Convert averaged probabilities to **`action ∈ {0,1}`** using a **calibrated cutoff**.
- **Threshold objective:** tune the decision threshold on **OOF data** to **maximize the utility \(u\)** (not only AUC).
  - Sweep \( \tau \in [0,1] \); compute day-wise \( p_i \), \( t \), and \( u \) on OOF predictions; select \( \tau^\* = \arg\max_{\tau} u \); **lock \( \tau^\* \)** for test.







