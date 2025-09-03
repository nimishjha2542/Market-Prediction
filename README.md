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
