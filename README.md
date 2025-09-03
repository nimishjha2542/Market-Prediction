# Project Overview

This project builds a **binary trade-acceptance model** that, for each time-ordered opportunity, predicts whether to **execute (1)** or **pass (0)**.

In theory, perfectly efficient markets leave no systematic profit. In practice, markets exhibit micro-inefficiencies. A model that consistently identifies favorable trades can both generate returns and nudge prices toward fair value.

## Why this is hard

- Very low signal-to-noise in financial features  
- Strong feature correlation and redundancy  
- Non-stationarity and regime shifts that challenge generalization

## What we do

We train on historical, anonymized data from a major global exchange to learn a decision rule aimed at **maximizing risk-adjusted returns**. Models are evaluated **out of sample** on strictly time-ordered data using a streaming, leak-free harness that prevents look-ahead, with performance measured against subsequent market returns.

## Dataset

The project uses **historical, anonymized trading data** from a major global stock exchange. Each row corresponds to a trading opportunity at a given time.

### Key dataset elements
- **Features** (`feature_0 … feature_129`): 130 anonymized numeric predictors derived from market data.  
- **Returns** (`resp`, `resp_1 … resp_4`): realized returns over different future horizons.  
- **Weight**: importance of each trade; samples with `weight = 0` do not contribute to evaluation.  
- **Date & Time IDs**: `date` (integer day index) and `ts_id` (strictly increasing identifier) enforce the true time order of events.

### Preprocessing steps
- Dropped the earliest **85 days** to mitigate distributional shifts observed in feature variance.  
- Removed rows with `weight = 0` to focus learning on economically relevant trades.  
- Forward-filled missing values to maintain causality (no future leakage), then filled any remaining NAs with **0**.  
- Defined a **strict training-time label**: `action = 1` if *all* horizon returns (`resp` and `resp_1–4`) are positive, otherwise `0`. This conservative choice increases purity of positive labels in a noisy financial setting.  
- Applied **sample weighting** proportional to the average absolute returns across horizons, emphasizing economically impactful observations.


## Methodology

We combine **representation learning** and **classification** in a single neural network to address noisy, redundant financial features.

### Architecture (see figure below)

- **Autoencoder branch**
  - **Encoder**: learns a denoised latent representation from 130 input features.
  - **Decoder**: reconstructs inputs (**MSE loss**) to regularize the latent space.
  - **Supervised AE head**: auxiliary 5-unit classifier (sigmoids) predicting the sign of `resp`, `resp_1…resp_4`, injecting label signal into the representation (**BCE loss**).

- **MLP branch**
  - Consumes `[normalized raw inputs | encoder output]`.
  - Produces a 5-unit multi-label output (sigmoids, **BCE loss**), used for model selection and reporting.

### Regularization & optimization
- `GaussianNoise`, `BatchNorm`, `Dropout`, and **Swish** activations for stability and robustness.
- Adam optimizer with early stopping and per-fold checkpoints.
- Training **sample weights** proportional to the mean absolute return across horizons, emphasizing economically impactful observations.

### Cross-validation
- **5-fold Purged Group Time Series CV**.
- Groups = trading days, with a **31-day embargo** to prevent look-ahead.
- Validation monitored on **AUC** of the final classifier output.

### Decision thresholding & utility alignment
- Final trade decisions are derived by thresholding calibrated probabilities to maximize a risk-adjusted utility proxy on validation folds.


<p align="center">
  <img src="assets/architecture.png" alt="Model Architecture" width="600"/>
</p>

## Results

Model performance was evaluated using **5-fold Purged Group Time-Series Cross-Validation** with a **31-day embargo** to prevent look-ahead bias. Validation was monitored on the **AUC** of the final classifier output.

### Cross-validation performance (per fold AUC)
- Fold 1: **0.587**
- Fold 2: **0.594**
- Fold 3: **0.601**
- Fold 4: **0.598**
- Fold 5: **0.603**

### Summary
- **Mean AUC:** **0.597 ± 0.006** across folds  
- **Out-of-sample validation:** **AUC 0.605** on unseen financial data

These results demonstrate consistent performance across folds with low variance, suggesting the model generalizes well under temporal splits. The out-of-sample score slightly exceeds the cross-validation mean, reinforcing robustness against non-stationarity and confirming that the architecture captures signal in a low signal-to-noise environment.

