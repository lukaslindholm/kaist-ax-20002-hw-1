import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# AI Usage Declaration for KAIST AX20002:
# Parts of this code were generated with AI assistance.

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TASKS = ['A', 'B', 'C', 'D']

def pinball_loss(y_true, y_pred, q):
    error = y_true - y_pred
    return np.mean(np.maximum(q * error, (q - 1) * error))

print("=== FINAL LOCAL VALIDATION BENCH ===")

for TASK in TASKS:
    print(f"\n{'-'*40}\n--- Evaluating Task {TASK} ---\n{'-'*40}")
    
    train = pd.read_csv(f'data/train_{TASK}.csv')
    y = train['y'].to_numpy(dtype=float)
    X_df = train.drop(columns='y')

    # 1. Base Feature Engineering
    if TASK == 'A':
        X_df['x04_is_1'] = (np.abs(X_df['x04'] - 1.0) < 1e-4).astype(float)
        X_df['x05_inv'] = 1.0 / (X_df['x05'] + 0.001)

    X_df = pd.get_dummies(X_df, drop_first=True, dtype=float)
    
    # Outer split for evaluation (80/20)
    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )

    # 2. Strict Leak-Free Spatial Clustering (Task A)
    if TASK == 'A':
        kmeans = KMeans(n_clusters=50, random_state=42, n_init='auto')
        dist_tr = kmeans.fit_transform(X_train_df[['x07', 'x08']])
        dist_va = kmeans.transform(X_val_df[['x07', 'x08']])
        
        X_train_df = X_train_df.drop(columns=['x07', 'x08'])
        X_val_df = X_val_df.drop(columns=['x07', 'x08'])
        
        X_train_outer = np.hstack([X_train_df.to_numpy(), dist_tr])
        X_val_outer = np.hstack([X_val_df.to_numpy(), dist_va])
    else:
        X_train_outer = X_train_df.to_numpy()
        X_val_outer = X_val_df.to_numpy()

    # ---------------------------------------------------------
    # TASKS A, B, C (OLS, LogReg, Optuna)
    # ---------------------------------------------------------
    if TASK in ['A', 'B', 'C']:
        ols = LinearRegression()
        ols.fit(X_train_outer, y_train)
        X_train_aug = np.hstack([X_train_outer, ols.predict(X_train_outer).reshape(-1, 1)])
        X_val_aug = np.hstack([X_val_outer, ols.predict(X_val_outer).reshape(-1, 1)])
        
        if TASK == 'B':
            y_bin = (y_train > np.median(y_train)).astype(int)
            logreg = LogisticRegression(max_iter=1000, random_state=42)
            logreg.fit(X_train_outer, y_bin)
            X_train_aug = np.hstack([X_train_aug, logreg.predict_proba(X_train_outer)[:, 1].reshape(-1, 1)])
            X_val_aug = np.hstack([X_val_aug, logreg.predict_proba(X_val_outer)[:, 1].reshape(-1, 1)])

        task_preds = np.zeros((X_val_aug.shape[0], len(QUANTILES)))
        
        if TASK == 'A':
            params = {'n_estimators': 1401, 'learning_rate': 0.0974, 'num_leaves': 36, 'min_child_samples': 70, 'subsample': 0.7065, 'colsample_bytree': 0.6229}
            for j, q in enumerate(QUANTILES):
                model = LGBMRegressor(objective='quantile', alpha=q, random_state=42, verbose=-1, **params)
                model.fit(X_train_aug, y_train)
                task_preds[:, j] = model.predict(X_val_aug)
            task_preds = np.clip(task_preds, a_min=None, a_max=5.0)
            
        elif TASK == 'B':
            params = {'objective': 'reg:quantileerror', 'n_estimators': 1096, 'learning_rate': 0.0941, 'max_depth': 4, 'min_child_weight': 5, 'subsample': 0.8622, 'colsample_bytree': 0.8361, 'reg_lambda': 0.5890, 'n_jobs': -1, 'random_state': 42}
            for j, q in enumerate(QUANTILES):
                model = xgb.XGBRegressor(quantile_alpha=q, **params)
                model.fit(X_train_aug, y_train)
                task_preds[:, j] = model.predict(X_val_aug)
                
        elif TASK == 'C':
            params = {'n_estimators': 1326, 'learning_rate': 0.0886, 'num_leaves': 52, 'min_child_samples': 14, 'subsample': 0.6738, 'colsample_bytree': 0.9322}
            for j, q in enumerate(QUANTILES):
                model = LGBMRegressor(objective='quantile', alpha=q, random_state=42, verbose=-1, **params)
                model.fit(X_train_aug, y_train)
                task_preds[:, j] = model.predict(X_val_aug)

        task_preds = np.sort(task_preds, axis=1)

    # ---------------------------------------------------------
    # TASK D (Optimized Mixture Model)
    # ---------------------------------------------------------
    elif TASK == 'D':
        THRESHOLD = 6.9
        DENSE_QUANTILES = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        
        y_class = (y_train > THRESHOLD).astype(int)
        clf = LGBMClassifier(n_estimators=150, num_leaves=31, random_state=42, verbose=-1)
        calibrated_clf = CalibratedClassifierCV(clf, method='sigmoid', cv=5)
        calibrated_clf.fit(X_train_outer, y_class)
        
        p_high = calibrated_clf.predict_proba(X_val_outer)[:, 1].reshape(-1, 1)
        
        mask_low = y_train <= THRESHOLD
        mask_high = y_train > THRESHOLD
        X_low, y_low = X_train_outer[mask_low], y_train[mask_low]
        X_high, y_high = X_train_outer[mask_high], y_train[mask_high]
        
        low_preds_dense = np.zeros((X_val_outer.shape[0], len(DENSE_QUANTILES)))
        high_preds_dense = np.zeros((X_val_outer.shape[0], len(DENSE_QUANTILES)))
        
        for j, q in enumerate(DENSE_QUANTILES):
            model_low = LGBMRegressor(objective='quantile', alpha=q, n_estimators=861, num_leaves=61, learning_rate=0.08685, random_state=42, verbose=-1)
            model_low.fit(X_low, y_low)
            low_preds_dense[:, j] = model_low.predict(X_val_outer)
            
            model_high = LGBMRegressor(objective='quantile', alpha=q, n_estimators=1144, num_leaves=220, learning_rate=0.01161, min_child_samples=50, random_state=42, verbose=-1)
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
    baseline_q_values = np.quantile(y_train, QUANTILES)
    total_baseline_loss = sum(pinball_loss(y_val, np.full_like(y_val, baseline_q_values[j]), q) for j, q in enumerate(QUANTILES))
    total_my_loss = sum(pinball_loss(y_val, task_preds[:, j], q) for j, q in enumerate(QUANTILES))
        
    avg_baseline_loss = total_baseline_loss / len(QUANTILES)
    avg_my_loss = total_my_loss / len(QUANTILES)
    d2_score = 1 - (avg_my_loss / avg_baseline_loss)
    
    print(f"  -> Baseline PL: {avg_baseline_loss:.4f} | Your PL: {avg_my_loss:.4f}")
    print(f"  -> Local D2 Score for {TASK}: {d2_score:.4f}")