import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.model_selection import KFold
from sklearn.calibration import CalibratedClassifierCV
import optuna
import warnings
import datetime
warnings.filterwarnings('ignore')

# AI Usage Declaration for KAIST AX20002:
# Parts of this code were generated with AI assistance.

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DENSE_QUANTILES = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
THRESHOLD = 6.9
N_TRIALS = 30 

def pinball_loss(y_true, y_pred, q):
    error = y_true - y_pred
    return np.mean(np.maximum(q * error, (q - 1) * error))

print("=== OPTIMIZING MIXTURE EXPERTS FOR TASK D ===")
print("[!] VARNING: Detta är ett tungt skript. Varje trial tränar 130 modeller.\n")

# 1. Load Data
train = pd.read_csv('data/train_D.csv')
y = train['y'].to_numpy(dtype=float)
X_df = pd.get_dummies(train.drop(columns='y'), drop_first=True, dtype=float)
X = X_df.to_numpy()

# 2. Optuna Objective
def objective(trial):
    # Definiera separata parametrar för Low-regimen och High-regimen
    low_leaves = trial.suggest_int('low_leaves', 15, 63)
    low_lr = trial.suggest_float('low_lr', 0.01, 0.1, log=True)
    low_est = trial.suggest_int('low_est', 300, 1000)
    
    high_leaves = trial.suggest_int('high_leaves', 63, 255)
    high_lr = trial.suggest_float('high_lr', 0.01, 0.1, log=True)
    high_est = trial.suggest_int('high_est', 500, 1500)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_d2_scores = []
    
    for train_idx, val_idx in kf.split(X):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        
        # A. Gating Network (Hardcoded, snabbt att träna)
        y_class_tr = (y_tr > THRESHOLD).astype(int)
        clf = LGBMClassifier(n_estimators=150, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
        calibrated_clf = CalibratedClassifierCV(clf, method='sigmoid', cv=3)
        calibrated_clf.fit(X_tr, y_class_tr)
        p_high = calibrated_clf.predict_proba(X_va)[:, 1].reshape(-1, 1)
        
        # B. Segmentera Träningsdatan
        mask_low = y_tr <= THRESHOLD
        mask_high = y_tr > THRESHOLD
        X_low, y_low = X_tr[mask_low], y_tr[mask_low]
        X_high, y_high = X_tr[mask_high], y_tr[mask_high]
        
        # Matriser för valideringsprediktioner
        low_preds_dense = np.zeros((X_va.shape[0], len(DENSE_QUANTILES)))
        high_preds_dense = np.zeros((X_va.shape[0], len(DENSE_QUANTILES)))
        
        # C. Träna experter (De tunga stegen)
        for j, q in enumerate(DENSE_QUANTILES):
            # Low Expert
            m_low = LGBMRegressor(objective='quantile', alpha=q, n_estimators=low_est, 
                                  num_leaves=low_leaves, learning_rate=low_lr, random_state=42, verbose=-1, n_jobs=2)
            m_low.fit(X_low, y_low)
            low_preds_dense[:, j] = m_low.predict(X_va)
            
            # High Expert
            m_high = LGBMRegressor(objective='quantile', alpha=q, n_estimators=high_est, 
                                   num_leaves=high_leaves, learning_rate=high_lr, min_child_samples=50, random_state=42, verbose=-1, n_jobs=2)
            m_high.fit(X_high, y_high)
            high_preds_dense[:, j] = m_high.predict(X_va)
            
        low_preds_dense.sort(axis=1)
        high_preds_dense.sort(axis=1)
        
        # D. CDF Invertering
        task_preds = np.zeros((X_va.shape[0], len(QUANTILES)))
        y_grid = np.linspace(0, 21, 2100)
        
        for i in range(X_va.shape[0]):
            cdf_low = np.interp(y_grid, low_preds_dense[i, :], DENSE_QUANTILES)
            cdf_high = np.interp(y_grid, high_preds_dense[i, :], DENSE_QUANTILES)
            cdf_mix = (1 - p_high[i, 0]) * cdf_low + p_high[i, 0] * cdf_high
            task_preds[i, :] = np.interp(QUANTILES, cdf_mix, y_grid)
            
        task_preds = np.clip(task_preds, a_min=0.0, a_max=21.0)
        task_preds = np.sort(task_preds, axis=1)
        
        # E. Utvärdera
        baseline_q_values = np.quantile(y_tr, QUANTILES)
        total_baseline_loss = sum(pinball_loss(y_va, np.full_like(y_va, baseline_q_values[j]), q) for j, q in enumerate(QUANTILES))
        total_my_loss = sum(pinball_loss(y_va, task_preds[:, j], q) for j, q in enumerate(QUANTILES))
        
        fold_d2 = 1 - (total_my_loss / total_baseline_loss)
        fold_d2_scores.append(fold_d2)
        
    return np.mean(fold_d2_scores)

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_TRIALS)

    print(f"\nBEST TASK D SCORE: {study.best_value:.4f}")
    print(f"BEST PARAMS: {study.best_params}")