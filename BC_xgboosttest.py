import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
import optuna
import warnings
warnings.filterwarnings('ignore')

# AI Usage Declaration for KAIST AX20002:
# Parts of this code were generated with AI assistance.

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TASKS_TO_OPTIMIZE = ['B', 'C']
N_TRIALS = 30

def pinball_loss(y_true, y_pred, q):
    error = y_true - y_pred
    return np.mean(np.maximum(q * error, (q - 1) * error))

def load_and_prep_data(task):
    train = pd.read_csv(f'data/train_{task}.csv')
    y = train['y'].to_numpy(dtype=float)
    X_df = pd.get_dummies(train.drop(columns='y'), drop_first=True, dtype=float)
    X = X_df.to_numpy()
    return X, y

def create_objective(task, X, y):
    def objective(trial):
        params = {
            'objective': 'reg:quantileerror',
            'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True), # L2 Regularization (stark för små dataset)
            'random_state': 42,
            'n_jobs': -1
        }
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_d2_scores = []
        
        for train_idx, val_idx in kf.split(X):
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]
            
            # OLS Baseline
            ols = LinearRegression()
            ols.fit(X_tr, y_tr)
            X_tr_aug = np.hstack([X_tr, ols.predict(X_tr).reshape(-1, 1)])
            X_va_aug = np.hstack([X_va, ols.predict(X_va).reshape(-1, 1)])
            
            # LogReg (Enbart Task B)
            if task == 'B':
                y_bin = (y_tr > np.median(y_tr)).astype(int)
                logreg = LogisticRegression(max_iter=1000, random_state=42)
                logreg.fit(X_tr, y_bin)
                X_tr_aug = np.hstack([X_tr_aug, logreg.predict_proba(X_tr)[:, 1].reshape(-1, 1)])
                X_va_aug = np.hstack([X_va_aug, logreg.predict_proba(X_va)[:, 1].reshape(-1, 1)])
            
            val_preds = np.zeros((X_va_aug.shape[0], len(QUANTILES)))
            
            for j, q in enumerate(QUANTILES):
                model = xgb.XGBRegressor(quantile_alpha=q, **params)
                model.fit(X_tr_aug, y_tr)
                val_preds[:, j] = model.predict(X_va_aug)
                
            val_preds = np.sort(val_preds, axis=1)
            
            baseline_q_values = np.quantile(y_tr, QUANTILES)
            total_baseline_loss = sum(pinball_loss(y_va, np.full_like(y_va, baseline_q_values[j]), q) for j, q in enumerate(QUANTILES))
            total_my_loss = sum(pinball_loss(y_va, val_preds[:, j], q) for j, q in enumerate(QUANTILES))
            
            avg_baseline_loss = total_baseline_loss / len(QUANTILES)
            avg_my_loss = total_my_loss / len(QUANTILES)
            
            fold_d2 = 0 if avg_baseline_loss == 0 else 1 - (avg_my_loss / avg_baseline_loss)
            fold_d2_scores.append(fold_d2)
            
        return np.mean(fold_d2_scores)
    return objective

if __name__ == '__main__':
    for task in TASKS_TO_OPTIMIZE:
        print(f"\n=== OPTIMIZING XGBOOST FOR TASK {task} ===")
        try:
            X, y = load_and_prep_data(task)
            objective = create_objective(task, X, y)
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=N_TRIALS)
            
            print(f"\nTASK {task} XGBOOST FINISHED")
            print(f"Best D2 Score (5-Fold CV): {study.best_value:.4f}")
            print(f"Best Parameters: {study.best_params}")
            print("-" * 50)
        except Exception as e:
            print(f"Krasch på Task {task}. Kontrollera att din XGBoost-version är >= 2.0. Fel: {e}")