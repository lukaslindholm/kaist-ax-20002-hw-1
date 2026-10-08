import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

# AI Usage Declaration for KAIST AX20002:
# Parts of this code were generated with AI assistance.

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TASKS = ['A', 'B', 'C', 'D']

def pinball_loss(y_true, y_pred, q):
    error = y_true - y_pred
    return np.mean(np.maximum(q * error, (q - 1) * error))

print("=== FINAL LOCAL VALIDATION BENCH (TRUE BEST PIPELINES) ===")

for TASK in TASKS:
    print(f"\n{'-'*40}\n--- Evaluating Task {TASK} ---\n{'-'*40}")
    
    train = pd.read_csv(f'data/train_{TASK}.csv')
    y = train['y'].to_numpy(dtype=float)
    X_df = pd.get_dummies(train.drop(columns='y'), drop_first=True, dtype=float)

    # 1. Feature Engineering (Innan split)
    if TASK == 'A':
        X_df['lat_lon_grid'] = np.round(X_df['x07']) * np.round(X_df['x08'])
        X_df['x04_is_1'] = (np.abs(X_df['x04'] - 1.0) < 1e-4).astype(float)
        X_df['x05_inv'] = 1.0 / (X_df['x05'] + 0.001)

    X = X_df.to_numpy()
    X_train_outer, X_val_outer, y_train_outer, y_val_outer = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------------------------------------------------------
    # TASKS A, B, C (Meta-features)
    # ---------------------------------------------------------
    if TASK in ['A', 'B', 'C']:
        # OLS Baseline meta-feature
        ols = LinearRegression()
        ols.fit(X_train_outer, y_train_outer)
        X_train_aug = np.hstack([X_train_outer, ols.predict(X_train_outer).reshape(-1, 1)])
        X_val_aug = np.hstack([X_val_outer, ols.predict(X_val_outer).reshape(-1, 1)])

        # Logistic Regression meta-feature enbart för B
        if TASK == 'B':
            y_bin = (y_train_outer > np.median(y_train_outer)).astype(int)
            logreg = LogisticRegression(max_iter=1000, random_state=42)
            logreg.fit(X_train_outer, y_bin)
            X_train_aug = np.hstack([X_train_aug, logreg.predict_proba(X_train_outer)[:, 1].reshape(-1, 1)])
            X_val_aug = np.hstack([X_val_aug, logreg.predict_proba(X_val_outer)[:, 1].reshape(-1, 1)])

        if TASK == 'A':
            leaves, n_est, lr = 100, 1000, 0.05
        else:
            leaves, n_est, lr = 31, 500, 0.1

        task_preds = np.zeros((X_val_aug.shape[0], len(QUANTILES)))
        for j, q in enumerate(QUANTILES):
            model = LGBMRegressor(
                objective='quantile', alpha=q, n_estimators=n_est,
                num_leaves=leaves, learning_rate=lr, random_state=42, verbose=-1
            )
            model.fit(
                X_train_aug, y_train_outer,
                eval_set=[(X_val_aug, y_val_outer)],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
            task_preds[:, j] = model.predict(X_val_aug)

        if TASK == 'A':
            task_preds = np.clip(task_preds, a_min=None, a_max=5.0)
        task_preds = np.sort(task_preds, axis=1)

    # ---------------------------------------------------------
    # TASK D (Mixture Model Architecture)
    # ---------------------------------------------------------
    elif TASK == 'D':
        THRESHOLD = 6.9
        DENSE_QUANTILES = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        
        y_class_outer = (y_train_outer > THRESHOLD).astype(int)
        clf = LGBMClassifier(n_estimators=150, num_leaves=31, random_state=42, verbose=-1)
        calibrated_clf = CalibratedClassifierCV(clf, method='sigmoid', cv=5)
        calibrated_clf.fit(X_train_outer, y_class_outer)
        
        p_high = calibrated_clf.predict_proba(X_val_outer)[:, 1].reshape(-1, 1)
        
        mask_low = y_train_outer <= THRESHOLD
        mask_high = y_train_outer > THRESHOLD
        X_low, y_low = X_train_outer[mask_low], y_train_outer[mask_low]
        X_high, y_high = X_train_outer[mask_high], y_train_outer[mask_high]
        
        low_preds_dense = np.zeros((X_val_outer.shape[0], len(DENSE_QUANTILES)))
        high_preds_dense = np.zeros((X_val_outer.shape[0], len(DENSE_QUANTILES)))
        
        for j, q in enumerate(DENSE_QUANTILES):
            model_low = LGBMRegressor(objective='quantile', alpha=q, n_estimators=700, 
                                      num_leaves=31, learning_rate=0.03, random_state=42, verbose=-1)
            model_low.fit(X_low, y_low)
            low_preds_dense[:, j] = model_low.predict(X_val_outer)
            
        for j, q in enumerate(DENSE_QUANTILES):
            model_high = LGBMRegressor(objective='quantile', alpha=q, n_estimators=1200, 
                                       num_leaves=127, learning_rate=0.02, min_child_samples=50, 
                                       reg_lambda=3.0, reg_alpha=0.1, random_state=42, verbose=-1)
            model_high.fit(X_high, y_high)
            high_preds_dense[:, j] = model_high.predict(X_val_outer)
            
        low_preds_dense.sort(axis=1)
        high_preds_dense.sort(axis=1)
        
        task_preds = np.zeros((X_val_outer.shape[0], len(QUANTILES)))
        y_grid = np.linspace(0, 21, 2100)
        
        for i in range(X_val_outer.shape[0]):
            cdf_low = np.interp(y_grid, low_preds_dense[i, :], DENSE_QUANTILES)
            cdf_high = np.interp(y_grid, high_preds_dense[i, :], DENSE_QUANTILES)
            cdf_mix = (1 - p_high[i, 0]) * cdf_low + p_high[i, 0] * cdf_high
            task_preds[i, :] = np.interp(QUANTILES, cdf_mix, y_grid)
            
        task_preds = np.clip(task_preds, a_min=0.0, a_max=21.0)
        task_preds = np.sort(task_preds, axis=1)

    # ---------------------------------------------------------
    # EVALUATION
    # ---------------------------------------------------------
    baseline_q_values = np.quantile(y_train_outer, QUANTILES)
    total_baseline_loss = sum(pinball_loss(y_val_outer, np.full_like(y_val_outer, baseline_q_values[j]), q) for j, q in enumerate(QUANTILES))
    total_my_loss = sum(pinball_loss(y_val_outer, task_preds[:, j], q) for j, q in enumerate(QUANTILES))
        
    avg_baseline_loss = total_baseline_loss / len(QUANTILES)
    avg_my_loss = total_my_loss / len(QUANTILES)
    d2_score = 1 - (avg_my_loss / avg_baseline_loss)
    
    print(f"  -> Baseline PL: {avg_baseline_loss:.4f} | Your PL: {avg_my_loss:.4f}")
    print(f"  -> Local D2 Score for {TASK}: {d2_score:.4f}")