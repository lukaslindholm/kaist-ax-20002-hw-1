import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

# AI Usage Declaration for KAIST AX20002:
# Parts of this code were generated with AI assistance.

TARGET_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DENSE_QUANTILES = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
THRESHOLD = 6.9

def pinball_loss(y_true, y_pred, q):
    error = y_true - y_pred
    return np.mean(np.maximum(q * error, (q - 1) * error))

def run_task_D_mixture():
    print("=== MIXTURE CDF EXPERIMENT FOR TASK D (Low/High Regime) ===")
    
    # 1. Ladda data (inga manuella feature transformations)
    train = pd.read_csv('data/train_D.csv')
    y = train['y'].to_numpy(dtype=float)
    X = train.drop(columns=['y']).to_numpy()
    
    # 2. Skapa Outer Split (Enbart för slutlig D2 score)
    X_train_outer, X_val_outer, y_train_outer, y_val_outer = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 3 & 4. Train and calibrate the Classifier using internal CV
    print("[*] Training and Platt-calibrating regime classifier (y > 6.9)...")
    y_class_outer = (y_train_outer > THRESHOLD).astype(int)
    
    clf = LGBMClassifier(n_estimators=150, num_leaves=31, random_state=42, verbose=-1)
    
    # cv=5 handles the calibration splits automatically, bypassing the 'prefit' version bug
    calibrated_clf = CalibratedClassifierCV(clf, method='sigmoid', cv=5)
    calibrated_clf.fit(X_train_outer, y_class_outer)
    
    # Predict P(high | x) for the validation set
    p_high = calibrated_clf.predict_proba(X_val_outer)[:, 1].reshape(-1, 1)
    
    # 5. Segment training data for the Experts
    # We now get to use 100% of the training data for the experts instead of losing 25% to calibration
    mask_low = y_train_outer <= THRESHOLD
    mask_high = y_train_outer > THRESHOLD
    
    X_low, y_low = X_train_outer[mask_low], y_train_outer[mask_low]
    X_high, y_high = X_train_outer[mask_high], y_train_outer[mask_high]
    
    # 6. Träna Dense Quantile Experts
    low_preds_dense = np.zeros((X_val_outer.shape[0], len(DENSE_QUANTILES)))
    high_preds_dense = np.zeros((X_val_outer.shape[0], len(DENSE_QUANTILES)))
    
    print("[*] Tränar Low-Regime Expert (700 est, 31 leaves)...")
    for j, q in enumerate(DENSE_QUANTILES):
        model_low = LGBMRegressor(objective='quantile', alpha=q, n_estimators=700, 
                                  num_leaves=31, learning_rate=0.03, random_state=42, verbose=-1)
        model_low.fit(X_low, y_low)
        low_preds_dense[:, j] = model_low.predict(X_val_outer)
        
    print("[*] Tränar High-Regime Expert (1200 est, 127 leaves)...")
    for j, q in enumerate(DENSE_QUANTILES):
        model_high = LGBMRegressor(objective='quantile', alpha=q, n_estimators=1200, 
                                   num_leaves=127, learning_rate=0.02, min_child_samples=50, 
                                   reg_lambda=3.0, reg_alpha=0.1, random_state=42, verbose=-1)
        model_high.fit(X_high, y_high)
        high_preds_dense[:, j] = model_high.predict(X_val_outer)
        
    # Säkerställ att kvantilerna är stigande (krävs för CDF-interpolation)
    low_preds_dense.sort(axis=1)
    high_preds_dense.sort(axis=1)
    
    # 7. Mixture CDF Invertering (Kombinera experterna)
    print("[*] Utför numerisk CDF-invertering...")
    val_preds_final = np.zeros((X_val_outer.shape[0], len(TARGET_QUANTILES)))
    y_grid = np.linspace(0, 21, 2100) # Fint grid för y-axeln
    
    for i in range(X_val_outer.shape[0]):
        # Beräkna interpolerad CDF för Low och High på y_grid
        # np.interp(x, xp, fp) -> vi vill veta CDF(y) givet (predikterat y, kvantil)
        cdf_low = np.interp(y_grid, low_preds_dense[i, :], DENSE_QUANTILES)
        cdf_high = np.interp(y_grid, high_preds_dense[i, :], DENSE_QUANTILES)
        
        # Blanda CDF:erna med P(high)
        cdf_mix = (1 - p_high[i, 0]) * cdf_low + p_high[i, 0] * cdf_high
        
        # Invertera CDF_mix för att hitta y vid våra TARGET_QUANTILES
        # Måste filtrera bort platta regioner i cdf_mix för att np.interp ska fungera smidigt
        val_preds_final[i, :] = np.interp(TARGET_QUANTILES, cdf_mix, y_grid)
        
    # 8. Post-processing: Klipp (0-21) och sortera
    val_preds_final = np.clip(val_preds_final, a_min=0.0, a_max=21.0)
    val_preds_final.sort(axis=1)
    
    # 9. Utvärdera Pinball Loss och D2 Score
    baseline_q_values = np.quantile(y_train_outer, TARGET_QUANTILES)
    total_baseline_loss = 0
    total_my_loss = 0
    
    for j, q in enumerate(TARGET_QUANTILES):
        baseline_preds = np.full_like(y_val_outer, baseline_q_values[j])
        total_baseline_loss += pinball_loss(y_val_outer, baseline_preds, q)
        total_my_loss += pinball_loss(y_val_outer, val_preds_final[:, j], q)
        
    avg_baseline_loss = total_baseline_loss / len(TARGET_QUANTILES)
    avg_my_loss = total_my_loss / len(TARGET_QUANTILES)
    d2_score = 1 - (avg_my_loss / avg_baseline_loss)
    
    print("-" * 50)
    print(f"Baseline PL: {avg_baseline_loss:.4f} | Your PL: {avg_my_loss:.4f}")
    print(f"--> Mixture D2 Score for Task D: {d2_score:.4f} <--")

if __name__ == '__main__':
    run_task_D_mixture()

    