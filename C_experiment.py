import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# AI Usage Declaration for KAIST AX20002:
# Parts of this code were generated with AI assistance.

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def pinball_loss(y_true, y_pred, q):
    error = y_true - y_pred
    return np.mean(np.maximum(q * error, (q - 1) * error))

def run_task_c_variance_experiment():
    print("=== VARIANCE META-FEATURE EXPERIMENT FOR TASK C ===")
    
    train = pd.read_csv('data/train_C.csv')
    y = train['y'].to_numpy(dtype=float)
    X_df = pd.get_dummies(train.drop(columns='y'), drop_first=True, dtype=float)
    X = X_df.to_numpy()
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. Steg 1: OLS Baseline (Hitta mitten)
    print("[*] Training OLS Baseline to find the mean...")
    ols = LinearRegression()
    ols.fit(X_train, y_train)
    
    y_hat_train = ols.predict(X_train)
    y_hat_val = ols.predict(X_val)
    
    # 2. Steg 2: Beräkna variansen (kvadrerade residualer)
    print("[*] Calculating squared residuals (variance)...")
    variance_train = (y_train - y_hat_train) ** 2
    
    # 3. Steg 3: Träna en Varians-estimator (Ridge Regression) för att prediktera tratten
    print("[*] Training Variance Estimator (RidgeCV)...")
    var_estimator = RidgeCV(alphas=np.logspace(-3, 3, 10))
    var_estimator.fit(X_train, variance_train)
    
    # Förutsäg förväntad varians
    expected_var_train = np.maximum(var_estimator.predict(X_train), 0) # Varians kan inte vara negativ
    expected_var_val = np.maximum(var_estimator.predict(X_val), 0)
    
    # 4. Steg 4: Konstruera den utökade feature-matrisen
    # Vi ger LightGBM både X, OLS-prediktionen (mitten) och Varians-prediktionen (spridningen)
    X_train_aug = np.hstack([X_train, y_hat_train.reshape(-1, 1), expected_var_train.reshape(-1, 1)])
    X_val_aug = np.hstack([X_val, y_hat_val.reshape(-1, 1), expected_var_val.reshape(-1, 1)])

    # Standard Feature matris utan varians (För jämförelse)
    X_train_standard = np.hstack([X_train, y_hat_train.reshape(-1, 1)])
    X_val_standard = np.hstack([X_val, y_hat_val.reshape(-1, 1)])

    # Dina Optuna parametrar för Task C
    params = {'n_estimators': 1326, 'learning_rate': 0.0886, 'num_leaves': 52, 
              'min_child_samples': 14, 'subsample': 0.6738, 'colsample_bytree': 0.9322}

    # --- Träna STANDARD modellen (Bara OLS) ---
    print("[*] Training Standard LGBM (OLS only)...")
    standard_preds = np.zeros((X_val.shape[0], len(QUANTILES)))
    for j, q in enumerate(QUANTILES):
        model = lgb.LGBMRegressor(objective='quantile', alpha=q, random_state=42, verbose=-1, **params)
        model.fit(X_train_standard, y_train)
        standard_preds[:, j] = model.predict(X_val_standard)

    # --- Träna VARIANCE modellen (OLS + Förväntad Varians) ---
    print("[*] Training Variance-Aware LGBM (OLS + Variance Meta-feature)...")
    variance_preds = np.zeros((X_val.shape[0], len(QUANTILES)))
    for j, q in enumerate(QUANTILES):
        model = lgb.LGBMRegressor(objective='quantile', alpha=q, random_state=42, verbose=-1, **params)
        model.fit(X_train_aug, y_train)
        variance_preds[:, j] = model.predict(X_val_aug)
        
    standard_preds = np.sort(standard_preds, axis=1)
    variance_preds = np.sort(variance_preds, axis=1)

    # --- Utvärdering ---
    baseline_q_values = np.quantile(y_train, QUANTILES)
    baseline_loss = sum(pinball_loss(y_val, np.full_like(y_val, baseline_q_values[j]), q) for j, q in enumerate(QUANTILES)) / len(QUANTILES)
    
    std_loss = sum(pinball_loss(y_val, standard_preds[:, j], q) for j, q in enumerate(QUANTILES)) / len(QUANTILES)
    var_loss = sum(pinball_loss(y_val, variance_preds[:, j], q) for j, q in enumerate(QUANTILES)) / len(QUANTILES)

    print("-" * 50)
    print(f"Standard D2 (Optuna):         {1 - (std_loss / baseline_loss):.4f}")
    print(f"Variance-Aware D2 (Optuna):   {1 - (var_loss / baseline_loss):.4f}")

if __name__ == '__main__':
    run_task_c_variance_experiment()