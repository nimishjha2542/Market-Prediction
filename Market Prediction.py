# =========================
# DATA LOADING & SETUP
# =========================

import warnings
warnings.filterwarnings('ignore')  

import os, gc
import pandas as pd
import numpy as np
import janestreet  

from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope

# Metrics and CV helpers (AUC is used as validation metric in this implementation).
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold

import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from joblib import dump, load
import datatable as dtable  # Faster CSV reader for large training file.

import tensorflow as tf
tf.random.set_seed(42)  # Fix TF RNG for partial reproducibility (deep nets remain somewhat stochastic).

import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

TEST = False  # Toggle for quick sanity runs (small sample) vs. full training run.

# -------------------------------------------------------------
# Fold-weighted averaging per Donate et al. (2012).
# Rationale:
# - Later folds typically contain more recent data, which may be more relevant
#   under covariate shift; a geometric weighting gives slightly more influence
#   to later folds when summarizing validation performance.
# - This is purely for *reporting* the CV AUC across folds.
# -------------------------------------------------------------
def weighted_average(a):
    w = []
    n = len(a)
    for j in range(1, n + 1):
        j = 2 if j == 1 else j  # Keep first two weights equal to avoid overweighting the very first fold.
        w.append(1 / (2**(n + 1 - j)))
    return np.average(a, weights = w)


# ===========================================
# FEATURE ENGINEERING & PREPROCESSING
# ===========================================
# Goals:
# - Load train.csv efficiently.
# - Define the feature set (anonymized feature_0..feature_129).
# - Apply competition-specific preprocessing:
#   * Drop early dates (domain shift in variance mentioned by organizers/community).
#   * Drop rows with weight == 0 (they do not affect the leaderboard utility).
#   * Forward-fill missing values to maintain causality (no look-ahead).
#   * Construct multi-label targets from resp and resp_{1..4} for supervised AE.
# - Prepare numpy arrays used by the model and CV, including sample weights (sw).


# ------- Preprocessing / Data Read -------
# Define a base path for data to avoid hardcoding in multiple places
DATA_PATH = '/Users/nimish/data/jane-street-market-prediction/'

if TEST:
    # In TEST mode we only read a tiny slice to speed up iteration.
    train = pd.read_csv(f'{DATA_PATH}train.csv', nrows=100)
    features = [c for c in train.columns if 'feature' in c]  # The anonymized numerical features used as inputs.
else:
    print('Loading...')
    # datatable is much faster than pandas.read_csv for ~1M rows; converts to pandas after read.
    train = dtable.fread(f'{DATA_PATH}train.csv').to_pandas()
    features = [c for c in train.columns if 'feature' in c]


    print('Filling...')
    # Remove first 85 days:
    # Rationale: early period exhibits different feature variance (documented by community),
    # which can destabilize training and inflate leakage risk across splits.
    train = train.query('date > 85').reset_index(drop = True) 

    # Remove rows with weight == 0:
    # Rationale: such rows do not contribute to the competition utility; excluding them
    # from training focuses the model on economically relevant samples.
    train = train.query('weight > 0').reset_index(drop = True)

    # Forward-fill missing values (per column) then remaining NAs to 0:
    # Causality-safe because we only use past observed values within each column.
    # The final .fillna(0) is a guard for leading NAs at the start of the series.
    train[features] = train[features].fillna(method = 'ffill').fillna(0)

    # Build a composite binary "action" label (train-time) by requiring all horizons positive:
    # action = 1 iff resp and resp_{1..4} are all > 0.
    # Rationale: strict positive agreement across horizons increases target purity in a low-SNR regime.
    train['action'] = ((train['resp_1'] > 0) & (train['resp_2'] > 0) & (train['resp_3'] > 0) & (train['resp_4'] > 0) & (train['resp'] > 0)).astype('int')

    # Multi-horizon return columns available in train (not available at test time).
    resp_cols = ['resp', 'resp_1', 'resp_2', 'resp_3', 'resp_4']

    # ==== Arrays for the model ====
    X = train[features].values  # Model inputs: anonymized numeric features.
    # y is a 5-label multi-output (one binary label per horizon) for the supervised AE and MLP heads.
    y = np.stack([(train[c] > 0).astype('int') for c in resp_cols]).T
    date = train['date'].values     # Used for grouped time-series CV (group = day).
    weight = train['weight'].values # Used for utility calculation (evaluation metric uses weight*resp).
    resp = train['resp'].values     # Primary realized return.
    # Sample weights emphasize rows with larger absolute returns across horizons:
    # Rationale: rows with bigger |returns| carry higher economic impact; emphasizing them
    # encourages the model to improve on the utility-aligned regions of the label space.
    sw = np.mean(np.abs(train[resp_cols].values), axis = 1)

# =========================================
# MODEL DEVELOPMENT & TRAINING
# =========================================
# Strategy:
# - Use a supervised autoencoder (AE) + MLP in a single Keras model.
#   * The AE reconstructs inputs (denoising representation learning).
#   * A supervised head ("ae_action") injects label signal into the learned representation.
#   * The downstream MLP consumes [original inputs || encoder output] and predicts actions.
# - Train the whole network jointly within each CV split to avoid leakage from pretraining.
# - Activation: 'swish' (smoother gradients than ReLU; helps in low SNR).
# - Regularization: BatchNorm + Dropout + GaussianNoise (data augmentation regularization).

# Cross-validation configuration (described in your write-up):
n_splits = 5   # 5-fold CV (time-ordered, grouped by date).
group_gap = 31 # 31-day embargo between train and validation groups to prevent look-ahead.

# ------- Model factory: AE + MLP -------
def create_ae_mlp(num_columns, num_labels, hidden_units, dropout_rates, ls = 1e-2, lr = 1e-3):
    # Inputs are the 130 anonymized features.
    inp = tf.keras.layers.Input(shape = (num_columns, ))

    # Initial normalization helps stabilize optimization across features with different scales.
    x0 = tf.keras.layers.BatchNormalization()(inp)
    
    # -------- Encoder (representation learning) --------
    # GaussianNoise acts as input-level augmentation to encourage robust, denoised features.
    encoder = tf.keras.layers.GaussianNoise(dropout_rates[0])(x0)
    encoder = tf.keras.layers.Dense(hidden_units[0])(encoder)      # First hidden layer for latent representation.
    encoder = tf.keras.layers.BatchNormalization()(encoder)
    encoder = tf.keras.layers.Activation('swish')(encoder)
    
    # -------- Decoder (reconstruction head) --------
    # Dropout regularizes the latent before reconstruction.
    decoder = tf.keras.layers.Dropout(dropout_rates[1])(encoder)
    # Reconstruct the original feature vector (MSE loss). Name: 'decoder' for Keras multi-output bookkeeping.
    decoder = tf.keras.layers.Dense(num_columns, name = 'decoder')(decoder)

    # -------- Supervised AE head (auxiliary classifier) --------
    # A small MLP branch taking the decoder output and learning to classify horizons.
    # This injects label supervision into the representation pathway to make the latent more predictive.
    x_ae = tf.keras.layers.Dense(hidden_units[1])(decoder)
    x_ae = tf.keras.layers.BatchNormalization()(x_ae)
    x_ae = tf.keras.layers.Activation('swish')(x_ae)
    x_ae = tf.keras.layers.Dropout(dropout_rates[2])(x_ae)

    # Multi-label output: 5 logistic units with sigmoid (one per horizon resp{,1..4}).
    out_ae = tf.keras.layers.Dense(num_labels, activation = 'sigmoid', name = 'ae_action')(x_ae)
    
    # -------- Downstream MLP (main classifier used for early stopping/selection) --------
    # Concatenate normalized raw inputs (x0) with encoder output to give the classifier both
    # low-level signals and learned denoised features.
    x = tf.keras.layers.Concatenate()([x0, encoder])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_rates[3])(x)
    
    # Stacked Dense -> BN -> Swish -> Dropout blocks (as defined by hidden_units/dropout_rates):
    # Rationale: deeper capacity to capture non-linear interactions under heavy regularization.
    for i in range(2, len(hidden_units)):
        x = tf.keras.layers.Dense(hidden_units[i])(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('swish')(x)
        x = tf.keras.layers.Dropout(dropout_rates[i + 2])(x)
        
    # Final multi-label output (same 5 horizons) for the main classifier.
    out = tf.keras.layers.Dense(num_labels, activation = 'sigmoid', name = 'action')(x)
    
    # -------- Multi-output Keras model --------
    # Outputs:
    #   - decoder: reconstruction target (MSE) aligns with AE objective.
    #   - ae_action: auxiliary classification head to supervise the AE path.
    #   - action: main classification head used for model selection (AUC).
    model = tf.keras.models.Model(inputs = inp, outputs = [decoder, out_ae, out])

    # -------- Losses, metrics, and optimizer --------
    # - Adam: adaptive LR works well for tabular deep nets with BN/Swish.
    # - MSE for decoder (reconstruction), BCE for both classifiers with optional label smoothing (ls).
    # - Metrics: MAE (decoder) and AUC (both classifiers). AUC is aligned with ranking quality,
    #   which is useful because we'll later threshold probabilities to decide actions.
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = lr),
                  loss = {'decoder': tf.keras.losses.MeanSquaredError(), 
                          'ae_action': tf.keras.losses.BinaryCrossentropy(label_smoothing = ls),
                          'action': tf.keras.losses.BinaryCrossentropy(label_smoothing = ls), 
                         },
                  metrics = {'decoder': tf.keras.metrics.MeanAbsoluteError(name = 'MAE'), 
                             'ae_action': tf.keras.metrics.AUC(name = 'AUC'), 
                             'action': tf.keras.metrics.AUC(name = 'AUC'), 
                            }, 
                 )
    
    return model

# Hyperparameters (from prior tuning/experience):
# - hidden_units: depth/width chosen to balance capacity and regularization in low SNR data.
# - dropout_rates: includes GaussianNoise sigma at index 0 and dropouts for subsequent blocks.
# - lr: 1e-3 is a standard Adam starting point; ls=0 disables label smoothing here.
params = {'num_columns': len(features), 
          'num_labels': 5, 
          'hidden_units': [96, 96, 896, 448, 448, 256], 
          'dropout_rates': [0.03527936123679956, 0.038424974585075086, 0.42409238408801436, 0.10431484318345882, 0.49230389137187497, 0.32024444956111164, 0.2716856145683449, 0.4379233941604448], 
          'ls': 0, 
          'lr':1e-3, 
         }


# =========================================
# EVALUATION & VALIDATION
# =========================================
# Cross-Validation approach:
# - PurgedGroupTimeSeriesSplit with n_splits=5 and group_gap=31.
#   * Group = date (ensures all rows from same day go to the same side).
#   * Embargo gap prevents leakage from temporal adjacency.
# - EarlyStopping/ModelCheckpoint monitored on 'val_action_AUC':
#   * We select models that rank positives better on the validation set, which
#     correlates with ability to pick profitable trades after thresholding.


if not TEST:
    scores = []                         # Collect per-fold validation AUCs for the main head.
    batch_size = 4096                   # Large batch for efficient GPU utilization on tabular data.

   
    gkf = PurgedGroupTimeSeriesSplit(n_splits = n_splits, group_gap = group_gap)

    # Iterate over time-ordered folds (train/validation indices per embargoed split).
    for fold, (tr, te) in enumerate(gkf.split(train['action'].values, train['action'].values, train['date'].values)):
        ckp_path = f'JSModel_{fold}.hdf5'  # Per-fold checkpoint path (weights-only) for the best val AUC.

        model = create_ae_mlp(**params)    # Fresh model per fold to avoid cross-fold contamination.

        # Save the best epoch by validation AUC on the main classifier head ('action').
        ckp = ModelCheckpoint(ckp_path, monitor = 'val_action_AUC', verbose = 0, 
                              save_best_only = True, save_weights_only = True, mode = 'max')

        # Early stop when validation AUC plateaus; patience=10 balances compute vs. overfitting risk.
        es = EarlyStopping(monitor = 'val_action_AUC', min_delta = 1e-4, patience = 10, mode = 'max', 
                           baseline = None, restore_best_weights = True, verbose = 0)

        # ----------------- TRAINING LOOP (per fold) -----------------
        # Inputs/targets:
        #   - X[...] goes to the AE (decoder) and both classifiers.
        #   - y[...] (5 binary labels) supervise 'ae_action' and 'action' outputs.
        #   - sample_weight=sw[tr]: emphasize economically impactful samples (proxy for utility).
        # Validation:
        #   - Validation data mirrors the multi-output structure.
        #   - Callbacks enforce early stopping and checkpointing on val_action_AUC.
        history = model.fit(X[tr], [X[tr], y[tr], y[tr]], 
                            validation_data = (X[te], [X[te], y[te], y[te]]), 
                            sample_weight = sw[tr], 
                            epochs = 100, batch_size = batch_size, callbacks = [ckp, es], verbose = 0)

        # Extract the best validation AUC for the main head during this fold.
        hist = pd.DataFrame(history.history)
        score = hist['val_action_AUC'].max()
        print(f'Fold {fold} ROC AUC:\t', score)
        scores.append(score)

        # Free up GPU/CPU memory between folds; keep environment clean.
        K.clear_session()
        del model
        rubbish = gc.collect()
    
    # Summarize CV performance with geometric-style weighting (later folds slightly favored).
    print('Weighted Average CV Score:', weighted_average(scores))
