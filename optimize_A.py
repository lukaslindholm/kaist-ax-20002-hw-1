import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import optuna
import warnings
import datetime
warnings.filterwarnings('ignore')

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
N_TRIALS = 30 

def pinball_loss(y_true, y_pred, q):
    error = y_true - y_pred
    return np.mean(np.maximum(q * error, (q - 1) * error))

print("=== OPTIMIZING TASK A WITH SPATIAL DISTANCES (K=50) ===")

# 1. Load Data
train = pd.read_csv('data/train_A.csv')
y = train['y'].to_numpy(dtype=float)
X_df = train.drop(columns=['y'])

# 2. Advanced Spatial Features (k=50)
print("[*] Generating spatial distance features...")
coords = X_df[['x07', 'x08']]
kmeans = KMeans(n_clusters=50, random_state=42, n_init='auto')
dist_features = kmeans.fit_transform(coords)
dist_df = pd.DataFrame(dist_features, columns=[f'dist_c_{i}' for i in range(50)])

# DROP raw coordinates, KEEP distances (Empirically proven better)
X_df = pd.concat([X_df.drop(columns=['x07', 'x08']), dist_df], axis=1)

X_df['x04_is_1'] = (np.abs(X_df['x04'] - 1.0) < 1e-4).astype(float)
X_df['x05_inv'] = 1.0 / (X_df['x05'] + 0.001)

X_df = pd.get_dummies(X_df, drop_first=True, dtype=float)
X = X_df.to_numpy()

# 3. Optuna Objective
def objective(trial):
    params = {
        'objective': 'quantile',
        'n_estimators': trial.suggest_int('n_estimators', 800, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.9), # Tweakad sökrymd pga fler kolumner
        'random_state': 42,
        'verbose': -1
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_d2_scores = []
    
    for train_idx, val_idx in kf.split(X):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        
        # OLS Baseline Meta-feature
        ols = LinearRegression()
        ols.fit(X_tr, y_tr)
        X_tr_aug = np.hstack([X_tr, ols.predict(X_tr).reshape(-1, 1)])
        X_va_aug = np.hstack([X_va, ols.predict(X_va).reshape(-1, 1)])
        
        val_preds = np.zeros((X_va_aug.shape[0], len(QUANTILES)))
        
        for j, q in enumerate(QUANTILES):
            model = lgb.LGBMRegressor(alpha=q, **params)
            model.fit(X_tr_aug, y_tr)
            val_preds[:, j] = model.predict(X_va_aug)
            
        val_preds = np.clip(val_preds, a_min=None, a_max=5.0)
        val_preds = np.sort(val_preds, axis=1)
        
        baseline_q_values = np.quantile(y_tr, QUANTILES)
        total_baseline_loss = sum(pinball_loss(y_va, np.full_like(y_va, baseline_q_values[j]), q) for j, q in enumerate(QUANTILES))
        total_my_loss = sum(pinball_loss(y_va, val_preds[:, j], q) for j, q in enumerate(QUANTILES))
        
        avg_baseline_loss = total_baseline_loss / len(QUANTILES)
        avg_my_loss = total_my_loss / len(QUANTILES)
        
        fold_d2 = 1 - (avg_my_loss / avg_baseline_loss)
        fold_d2_scores.append(fold_d2)
        
    return np.mean(fold_d2_scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=N_TRIALS)

print(f"\nBEST TASK A D2 SCORE: {study.best_value:.4f}")
print(f"BEST PARAMS: {study.best_params}")