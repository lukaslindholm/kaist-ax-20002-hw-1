import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

# AI Usage Declaration for KAIST AX20002:
# Parts of this code were generated with AI assistance.

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
Q_LABELS = [10, 20, 30, 40, 50, 60, 70, 80, 90]
TASKS = ['A', 'B', 'C', 'D']

print("=== BUILDING KAGGLE SUBMISSION ===")

all_ids = []
all_values = []

for TASK in TASKS:
    print(f"--- Training Final Models for Task {TASK} ---")
    
    # Ladda Data
    train = pd.read_csv(f'data/train_{TASK}.csv')
    test = pd.read_csv(f'data/test_{TASK}.csv')
    
    y_train = train['y'].to_numpy(dtype=float)
    row_ids = test['row_id'].values
    
    both = pd.concat([train.drop(columns='y'), test.drop(columns='row_id')], keys=["tr", "te"])
    both_encoded = pd.get_dummies(both, drop_first=True, dtype=float)
    
    # ---------------------------------------------------------
    # Feature Engineering (Före OLS)
    # ---------------------------------------------------------
    if TASK == 'A':
        both_encoded['lat_lon_grid'] = np.round(both_encoded['x07']) * np.round(both_encoded['x08'])
        both_encoded['x04_is_1'] = (np.abs(both_encoded['x04'] - 1.0) < 1e-4).astype(float)
        both_encoded['x05_inv'] = 1.0 / (both_encoded['x05'] + 0.001)

    X_train = both_encoded.loc["tr"].to_numpy()
    X_test = both_encoded.loc["te"].to_numpy()

    # ---------------------------------------------------------
    # Tasks A, B, C (OLS, LogReg, Optuna)
    # ---------------------------------------------------------
    if TASK in ['A', 'B', 'C']:
        
        # 1. OLS Baseline meta-feature
        ols = LinearRegression()
        ols.fit(X_train, y_train)
        X_train_aug = np.hstack([X_train, ols.predict(X_train).reshape(-1, 1)])
        X_test_aug = np.hstack([X_test, ols.predict(X_test).reshape(-1, 1)])
        
        # 2. Logistic Regression meta-feature (Endast Task B)
        if TASK == 'B':
            y_bin = (y_train > np.median(y_train)).astype(int)
            logreg = LogisticRegression(max_iter=1000, random_state=42)
            logreg.fit(X_train, y_bin)
            X_train_aug = np.hstack([X_train_aug, logreg.predict_proba(X_train)[:, 1].reshape(-1, 1)])
            X_test_aug = np.hstack([X_test_aug, logreg.predict_proba(X_test)[:, 1].reshape(-1, 1)])

        # 3. Optuna Hyperparameters
        if TASK == 'A':
            params = {'n_estimators': 1401, 'learning_rate': 0.0974, 'num_leaves': 36, 'min_child_samples': 70, 'subsample': 0.7065, 'colsample_bytree': 0.6229}
        elif TASK == 'B':
            params = {'n_estimators': 1037, 'learning_rate': 0.0637, 'num_leaves': 53, 'min_child_samples': 28, 'subsample': 0.8014, 'colsample_bytree': 0.8498}
        elif TASK == 'C':
            params = {'n_estimators': 1326, 'learning_rate': 0.0886, 'num_leaves': 52, 'min_child_samples': 14, 'subsample': 0.6738, 'colsample_bytree': 0.9322}

        # 4. Träna Kvantiler (100% data, ingen early stopping)
        task_preds = np.zeros((X_test_aug.shape[0], len(QUANTILES)))
        for j, q in enumerate(QUANTILES):
            model = LGBMRegressor(
                objective='quantile', alpha=q, random_state=42, verbose=-1, **params
            )
            model.fit(X_train_aug, y_train)
            task_preds[:, j] = model.predict(X_test_aug)
            
        if TASK == 'A':
            task_preds = np.clip(task_preds, a_min=None, a_max=5.0)

    # ---------------------------------------------------------
    # Task D (Mixture Model)
    # ---------------------------------------------------------
    elif TASK == 'D':
        THRESHOLD = 6.9
        DENSE_QUANTILES = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        
        y_class = (y_train > THRESHOLD).astype(int)
        clf = LGBMClassifier(n_estimators=150, num_leaves=31, random_state=42, verbose=-1)
        calibrated_clf = CalibratedClassifierCV(clf, method='sigmoid', cv=5)
        calibrated_clf.fit(X_train, y_class)
        
        p_high = calibrated_clf.predict_proba(X_test)[:, 1].reshape(-1, 1)
        
        mask_low = y_train <= THRESHOLD
        mask_high = y_train > THRESHOLD
        X_low, y_low = X_train[mask_low], y_train[mask_low]
        X_high, y_high = X_train[mask_high], y_train[mask_high]
        
        low_preds_dense = np.zeros((X_test.shape[0], len(DENSE_QUANTILES)))
        high_preds_dense = np.zeros((X_test.shape[0], len(DENSE_QUANTILES)))
        
        for j, q in enumerate(DENSE_QUANTILES):
            model_low = LGBMRegressor(objective='quantile', alpha=q, n_estimators=700, 
                                      num_leaves=31, learning_rate=0.03, random_state=42, verbose=-1)
            model_low.fit(X_low, y_low)
            low_preds_dense[:, j] = model_low.predict(X_test)
            
        for j, q in enumerate(DENSE_QUANTILES):
            model_high = LGBMRegressor(objective='quantile', alpha=q, n_estimators=1200, 
                                       num_leaves=127, learning_rate=0.02, min_child_samples=50, 
                                       reg_lambda=3.0, reg_alpha=0.1, random_state=42, verbose=-1)
            model_high.fit(X_high, y_high)
            high_preds_dense[:, j] = model_high.predict(X_test)
            
        low_preds_dense.sort(axis=1)
        high_preds_dense.sort(axis=1)
        
        task_preds = np.zeros((X_test.shape[0], len(QUANTILES)))
        y_grid = np.linspace(0, 21, 2100)
        
        for i in range(X_test.shape[0]):
            cdf_low = np.interp(y_grid, low_preds_dense[i, :], DENSE_QUANTILES)
            cdf_high = np.interp(y_grid, high_preds_dense[i, :], DENSE_QUANTILES)
            cdf_mix = (1 - p_high[i, 0]) * cdf_low + p_high[i, 0] * cdf_high
            task_preds[i, :] = np.interp(QUANTILES, cdf_mix, y_grid)
            
        task_preds = np.clip(task_preds, a_min=0.0, a_max=21.0)

    # ---------------------------------------------------------
    # Format and Save (Monotonicity Check)
    # ---------------------------------------------------------
    task_preds = np.sort(task_preds, axis=1) # Ensure non-decreasing quantiles
    
    for i, r_id in enumerate(row_ids):
        for j, q_lab in enumerate(Q_LABELS):
            all_ids.append(f"{r_id}_q{q_lab}")
            all_values.append(task_preds[i, j])

print("--- Generating submission_final.csv ---")
df_out = pd.DataFrame({
    'id': all_ids,
    'value': all_values
})
df_out.to_csv("submission_final2.csv", index=False)
print("Done! File ready for Kaggle.")