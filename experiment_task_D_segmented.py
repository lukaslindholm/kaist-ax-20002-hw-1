import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# AI Usage Declaration for KAIST AX20002:
# Parts of this code were generated with AI assistance.

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def pinball_loss(y_true, y_pred, q):
    error = y_true - y_pred
    return np.mean(np.maximum(q * error, (q - 1) * error))

def run_task_D_segmented():
    print("=== SEGMENTED EXPERIMENT FOR TASK D (Divide & Conquer) ===")
    
    # 1. Load data
    train = pd.read_csv('data/train_D.csv')
    
    y = train['y'].to_numpy(dtype=float)
    X_df = train.drop(columns=['y'])
    
    # We will segment based on x07. 
    # Visually, the variance explodes when x07 is roughly below 10,000.
    X = X_df.to_numpy()
    x07_idx = X_df.columns.get_loc('x07')
    
    # Create indices for the initial train/val split
    indices = np.arange(len(y))
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X, y, indices, test_size=0.2, random_state=42
    )

    # 2. Define segmentation logic (Split by x07 threshold)
    threshold = 10000.0
    print(f"[*] Segmenting data based on x07 < {threshold} vs x07 >= {threshold}")
    
    # Train masks
    mask_train_extreme = X_train[:, x07_idx] < threshold
    mask_train_normal = ~mask_train_extreme
    
    # Val masks
    mask_val_extreme = X_val[:, x07_idx] < threshold
    mask_val_normal = ~mask_val_extreme
    
    print(f"    -> Train instances: Extreme={mask_train_extreme.sum()}, Normal={mask_train_normal.sum()}")
    print(f"    -> Val instances: Extreme={mask_val_extreme.sum()}, Normal={mask_val_normal.sum()}")

    # 3. Prepare prediction array for validation set
    val_preds = np.zeros((X_val.shape[0], len(QUANTILES)))
    
    # 4. Train and predict on EXTREME segment
    print("\n[*] Training LightGBM on EXTREME segment...")
    for j, q in enumerate(QUANTILES):
        model_extreme = LGBMRegressor(
            objective='quantile',
            alpha=q,
            n_estimators=1000,
            num_leaves=63, # Slightly lower complexity since we have less data
            learning_rate=0.02,
            random_state=42,
            verbose=-1
        )
        model_extreme.fit(X_train[mask_train_extreme], y_train[mask_train_extreme])
        val_preds[mask_val_extreme, j] = model_extreme.predict(X_val[mask_val_extreme])
        
    # 5. Train and predict on NORMAL segment
    print("[*] Training LightGBM on NORMAL segment...")
    for j, q in enumerate(QUANTILES):
        model_normal = LGBMRegressor(
            objective='quantile',
            alpha=q,
            n_estimators=1000,
            num_leaves=127,
            learning_rate=0.02,
            random_state=42,
            verbose=-1
        )
        model_normal.fit(X_train[mask_train_normal], y_train[mask_train_normal])
        val_preds[mask_val_normal, j] = model_normal.predict(X_val[mask_val_normal])
        
    # 6. Post-process combined predictions
    print("\n[*] Clipping and sorting predictions...")
    val_preds = np.clip(val_preds, a_min=0.0, a_max=20.0)
    val_preds = np.sort(val_preds, axis=1)
    
    # 7. Evaluate global metric
    baseline_q_values = np.quantile(y_train, QUANTILES)
    total_baseline_loss = 0
    total_my_loss = 0
    
    for j, q in enumerate(QUANTILES):
        baseline_preds = np.full_like(y_val, baseline_q_values[j])
        total_baseline_loss += pinball_loss(y_val, baseline_preds, q)
        total_my_loss += pinball_loss(y_val, val_preds[:, j], q)
        
    avg_baseline_loss = total_baseline_loss / len(QUANTILES)
    avg_my_loss = total_my_loss / len(QUANTILES)
    d2_score = 1 - (avg_my_loss / avg_baseline_loss)
    
    print("-" * 50)
    print(f"Baseline PL: {avg_baseline_loss:.4f} | Your PL: {avg_my_loss:.4f}")
    print(f"--> Segmented D2 Score for Task D: {d2_score:.4f} <--")

if __name__ == '__main__':
    run_task_D_segmented()