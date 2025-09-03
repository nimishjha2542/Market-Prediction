# Project Overview

This project builds a **binary trade-acceptance model** that, for each time-ordered opportunity, predicts whether to **execute (1)** or **pass (0)**.

In theory, perfectly efficient markets leave no systematic profit. In practice, markets exhibit micro-inefficiencies. A model that consistently identifies favorable trades can both generate returns and nudge prices toward fair value.

## Why this is hard

- Very low signal-to-noise in financial features  
- Strong feature correlation and redundancy  
- Non-stationarity and regime shifts that challenge generalization

## What we do

We train on historical, anonymized data from a major global exchange to learn a decision rule aimed at **maximizing risk-adjusted returns**. Models are evaluated **out of sample** on strictly time-ordered data using a streaming, leak-free harness that prevents look-ahead, with performance measured against subsequent market returns.
