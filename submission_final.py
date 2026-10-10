import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# AI Usage Declaration for KAIST AX20002:
# Parts of this code were generated with AI assistance.

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
Q_LABELS = [10, 20, 30, 40, 50, 60, 70, 80, 90]
TASKS = ['A', 'B', 'C', 'D']

print("=== BUILDING FINAL KAGGLE SUBMISSION ===")

all_ids = []
all_values = []

for TASK in TASKS:
    print(f"--- Training Final Models for Task {TASK} ---")
    
    train = pd.read_csv(f'data/train_{TASK}.csv')
    test = pd.read_csv(f'data/test_{TASK}.csv')
    
    y_train = train['y'].to_numpy(dtype=float)
    row_ids = test['row_id'].values
    
    both = pd.concat([train.drop(columns='y'), test.drop(columns='row_id')], keys=["tr", "te"])
    
    # ---------------------------------------------------------
    # FEATURE ENGINEERING
    # ---------------------------------------------------------
    if TASK == 'A':
        coords = both[['x07', 'x08']]
        kmeans = KMeans(n_clusters=50, random_state=42, n_init='auto')
        dist_features = kmeans.fit_transform(coords)
        dist_df = pd.DataFrame(dist_features, columns=[f'dist_c_{i}' for i in range(50)], index=both.index)
        
        both = pd.concat([both.drop(columns=['x07', 'x08']), dist_df], axis=1)
        both['x04_is_1'] = (np.abs(both['x04'] - 1.0) < 1e-4).astype(float)
        both['x05_inv'] = 1.0 / (both['x05'] + 0.001)

    both_encoded = pd.get_dummies(both, drop_first=True, dtype=float)
    X_train = both_encoded.loc["tr"].to_numpy()
    X_test = both_encoded.loc["te"].to_numpy()

    # ---------------------------------------------------------
    # TASKS A, B, C
    # ---------------------------------------------------------
    if TASK in ['A', 'B', 'C']:
        ols = LinearRegression()
        ols.fit(X_train, y_train)
        X_train_aug = np.hstack([X_train, ols.predict(X_train).reshape(-1, 1)])
        X_test_aug = np.hstack([X_test, ols.predict(X_test).reshape(-1, 1)])
        
        if TASK == 'B':
            y_bin = (y_train > np.median(y_train)).astype(int)
            logreg = LogisticRegression(max_iter=1000, random_state=42)
            logreg.fit(X_train, y_bin)
            X_train_aug = np.hstack([X_train_aug, logreg.predict_proba(X_train)[:, 1].reshape(-1, 1)])
            X_test_aug = np.hstack([X_test_aug, logreg.predict_proba(X_test)[:, 1].reshape(-1, 1)])

        task_preds = np.zeros((X_test_aug.shape[0], len(QUANTILES)))
        
        if TASK == 'A':
            params = {'n_estimators': 1401, 'learning_rate': 0.0974, 'num_leaves': 36, 'min_child_samples': 70, 'subsample': 0.7065, 'colsample_bytree': 0.6229}
            for j, q in enumerate(QUANTILES):
                model = LGBMRegressor(objective='quantile', alpha=q, random_state=42, verbose=-1, **params)
                model.fit(X_train_aug, y_train)
                task_preds[:, j] = model.predict(X_test_aug)
            task_preds = np.clip(task_preds, a_min=None, a_max=5.0)
            
        elif TASK == 'B':
            params = {'objective': 'reg:quantileerror', 'n_estimators': 1096, 'learning_rate': 0.0941, 'max_depth': 4, 'min_child_weight': 5, 'subsample': 0.8622, 'colsample_bytree': 0.8361, 'reg_lambda': 0.5890, 'n_jobs': -1, 'random_state': 42}
            for j, q in enumerate(QUANTILES):
                model = xgb.XGBRegressor(quantile_alpha=q, **params)
                model.fit(X_train_aug, y_train)
                task_preds[:, j] = model.predict(X_test_aug)
                
        elif TASK == 'C':
            params = {'n_estimators': 1326, 'learning_rate': 0.0886, 'num_leaves': 52, 'min_child_samples': 14, 'subsample': 0.6738, 'colsample_bytree': 0.9322}
            for j, q in enumerate(QUANTILES):
                model = LGBMRegressor(objective='quantile', alpha=q, random_state=42, verbose=-1, **params)
                model.fit(X_train_aug, y_train)
                task_preds[:, j] = model.predict(X_test_aug)

    # ---------------------------------------------------------
    # TASK D (Optimized Mixture Model)
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
            model_low = LGBMRegressor(objective='quantile', alpha=q, n_estimators=861, num_leaves=61, learning_rate=0.08685, random_state=42, verbose=-1)
            model_low.fit(X_low, y_low)
            low_preds_dense[:, j] = model_low.predict(X_test)
            
            model_high = LGBMRegressor(objective='quantile', alpha=q, n_estimators=1144, num_leaves=220, learning_rate=0.01161, min_child_samples=50, random_state=42, verbose=-1)
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
    # FORMAT & EXPORT
    # ---------------------------------------------------------
    task_preds = np.sort(task_preds, axis=1) 
    for i, r_id in enumerate(row_ids):
        for j, q_lab in enumerate(Q_LABELS):
            all_ids.append(f"{r_id}_q{q_lab}")
            all_values.append(task_preds[i, j])

print("--- Generating submission_final.csv ---")
df_out = pd.DataFrame({'id': all_ids, 'value': all_values})
df_out.to_csv("submission_final.csv", index=False)
print("Done! Ready for submission.")