import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# AI Usage Declaration for KAIST AX20002:
# Parts of this code were generated with AI assistance.

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def pinball_loss(y_true, y_pred, q):
    error = y_true - y_pred
    return np.mean(np.maximum(q * error, (q - 1) * error))

def run_task_a_xgboost():
    print("=== SPATIAL DISTANCE & XGBOOST EXPERIMENT FOR TASK A ===")
    
    train = pd.read_csv('data/train_A.csv')
    y = train['y'].to_numpy(dtype=float)
    X_df = train.drop(columns=['y'])
    
    # 1. OPTIMIZED SPATIAL CLUSTERING (Distance-based instead of One-Hot)
    print("[*] Creating high-res spatial distance map (k=50)...")
    coords = X_df[['x07', 'x08']]
    kmeans = KMeans(n_clusters=50, random_state=42, n_init='auto')
    
    # Instead of cluster IDs, we get the distance to ALL 50 cluster centers.
    # This gives tree models continuous gradients to split on.
    dist_features = kmeans.fit_transform(coords)
    dist_df = pd.DataFrame(dist_features, columns=[f'dist_to_cluster_{i}' for i in range(50)])
    
    X_df = pd.concat([X_df.drop(columns=['x07', 'x08']), dist_df], axis=1)
    
    X_df['x04_is_1'] = (np.abs(X_df['x04'] - 1.0) < 1e-4).astype(float)
    X_df['x05_inv'] = 1.0 / (X_df['x05'] + 0.001)

    X_df = pd.get_dummies(X_df, drop_first=True, dtype=float)
    X = X_df.to_numpy()
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. OLS Baseline meta-feature
    ols = LinearRegression()
    ols.fit(X_train, y_train)
    X_train_aug = np.hstack([X_train, ols.predict(X_train).reshape(-1, 1)])
    X_val_aug = np.hstack([X_val, ols.predict(X_val).reshape(-1, 1)])

    # 3. Train LightGBM (With Optuna params)
    print("[*] Training LightGBM...")
    lgbm_preds = np.zeros((X_val_aug.shape[0], len(QUANTILES)))
    
    lgb_params = {'n_estimators': 1401, 'learning_rate': 0.0974, 'num_leaves': 36, 
                  'min_child_samples': 70, 'subsample': 0.7065, 'colsample_bytree': 0.6229}
                  
    for j, q in enumerate(QUANTILES):
        model = lgb.LGBMRegressor(objective='quantile', alpha=q, random_state=42, verbose=-1, **lgb_params)
        model.fit(X_train_aug, y_train)
        lgbm_preds[:, j] = model.predict(X_val_aug)

    # 4. Train XGBoost
    print("[*] Training XGBoost...")
    xgb_preds = np.zeros((X_val_aug.shape[0], len(QUANTILES)))
    
    # XGBoost requires 'reg:quantileerror' objective (available in XGBoost >= 2.0)
    xgb_params = {
        'objective': 'reg:quantileerror',
        'n_estimators': 800,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    }
    
    try:
        for j, q in enumerate(QUANTILES):
            model = xgb.XGBRegressor(quantile_alpha=q, **xgb_params)
            model.fit(X_train_aug, y_train)
            xgb_preds[:, j] = model.predict(X_val_aug)
    except Exception as e:
        print(f"\n[!] XGBoost quantile error failed (requires XGBoost >= 2.0). Skipping XGBoost. Error: {e}")
        xgb_preds = lgbm_preds.copy() # Fallback

    # 5. Ensemble & Post-processing
    print("[*] Blending predictions...")
    ensemble_preds = (lgbm_preds + xgb_preds) / 2.0
    
    lgbm_preds = np.clip(np.sort(lgbm_preds, axis=1), a_min=None, a_max=5.0)
    xgb_preds = np.clip(np.sort(xgb_preds, axis=1), a_min=None, a_max=5.0)
    ensemble_preds = np.clip(np.sort(ensemble_preds, axis=1), a_min=None, a_max=5.0)

    # 6. Evaluation
    baseline_q_values = np.quantile(y_train, QUANTILES)
    baseline_loss = sum(pinball_loss(y_val, np.full_like(y_val, baseline_q_values[j]), q) for j, q in enumerate(QUANTILES)) / len(QUANTILES)
    
    lgbm_loss = sum(pinball_loss(y_val, lgbm_preds[:, j], q) for j, q in enumerate(QUANTILES)) / len(QUANTILES)
    xgb_loss = sum(pinball_loss(y_val, xgb_preds[:, j], q) for j, q in enumerate(QUANTILES)) / len(QUANTILES)
    ens_loss = sum(pinball_loss(y_val, ensemble_preds[:, j], q) for j, q in enumerate(QUANTILES)) / len(QUANTILES)

    print("-" * 50)
    print(f"LGBM (Distances) D2: {1 - (lgbm_loss / baseline_loss):.4f}")
    print(f"XGBoost (Distances) D2: {1 - (xgb_loss / baseline_loss):.4f}")
    print(f"ENSEMBLE D2: {1 - (ens_loss / baseline_loss):.4f}")

if __name__ == '__main__':
    run_task_a_xgboost()