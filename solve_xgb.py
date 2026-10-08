import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TASKS = ['A', 'B', 'C', 'D']

def load_and_engineer_data(task):
    train = pd.read_csv(f'data/train_{task}.csv')
    test = pd.read_csv(f'data/test_{task}.csv')
    
    y_train = train['y'].to_numpy(dtype=float)
    both = pd.concat([train.drop(columns='y'), test.drop(columns='row_id')], keys=["tr", "te"])
    both_encoded = pd.get_dummies(both, drop_first=True, dtype=float)
    
    X_train, X_test = both_encoded.loc["tr"].to_numpy(), both_encoded.loc["te"].to_numpy()
    
    # OLS-tricket
    ols = LinearRegression().fit(X_train, y_train)
    X_train = np.hstack([X_train, ols.predict(X_train).reshape(-1, 1)])
    X_test = np.hstack([X_test, ols.predict(X_test).reshape(-1, 1)])
    
    return X_train, y_train, X_test, test['row_id'].values

all_row_ids, all_predictions = [], []

for task in TASKS:
    print(f"Tränar XGBoost för Task {task}...")
    X_train, y_train, X_test, row_ids = load_and_engineer_data(task)
    task_preds = np.zeros((X_test.shape[0], len(QUANTILES)))
    
    # Tillämpa din uppdaterade logik för trädkomplexitet!
    depth = 5 if task in ['B', 'D'] else 3
    
    for j, q in enumerate(QUANTILES):
        model = xgb.XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=q,
            n_estimators=150,
            max_depth=depth,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        task_preds[:, j] = model.predict(X_test)
    
    # Klippning
    if task == 'A': task_preds = np.clip(task_preds, a_min=None, a_max=5.0)
    if task == 'D': task_preds = np.clip(task_preds, a_min=0.0, a_max=None)
        
    task_preds_sorted = np.sort(task_preds, axis=1)
    q_labels = [int(q*100) for q in QUANTILES]
    for i, r_id in enumerate(row_ids):
        for j, q_lab in enumerate(q_labels):
            all_row_ids.append(f"{r_id}_q{q_lab}")
            all_predictions.append(task_preds_sorted[i, j])

pd.DataFrame({'id': all_row_ids, 'value': all_predictions}).to_csv("submission_xgb.csv", index=False)
print("submission_xgb.csv är redo!")