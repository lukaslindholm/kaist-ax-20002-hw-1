import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import os

# ==========================================
# 1. SETUP
# ==========================================
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TASKS = ['A', 'B', 'C', 'D']

def load_data(task):
    print(f"Laddar data för Task {task}...")
    train = pd.read_csv(f'data/train_{task}.csv')
    test = pd.read_csv(f'data/test_{task}.csv')
    
    y_train = train['y'].to_numpy(dtype=float)
    
    # 1. Slå ihop X_train och X_test temporärt
    both = pd.concat([train.drop(columns='y'), test.drop(columns='row_id')], keys=["tr", "te"])
    
    # 2. Omvandla all text till numeriska kolumner (One-Hot Encoding)
    both_encoded = pd.get_dummies(both, drop_first=True, dtype=float)
    
    # 3. Dela upp dem igen
    X_train = both_encoded.loc["tr"].to_numpy()
    X_test = both_encoded.loc["te"].to_numpy()
    
    return X_train, y_train, X_test, test['row_id'].values

all_row_ids = []
all_predictions = []

# ==========================================
# 2. TRAIN MACHINE LEARNING MODELS
# ==========================================
for task in TASKS:
    X_train, y_train, X_test, row_ids = load_data(task)
    
    # Vi skapar en (N, 9) array för att spara förutsägelserna för denna uppgift
    task_preds = np.zeros((X_test.shape[0], len(QUANTILES)))
    
    # GBR tränar ett separat ensemble av träd för VARJE kvantil
    for j, q in enumerate(QUANTILES):
        print(f"  Tränar GBR för q={q}...")
        
        # Loss='quantile' är magin. Alpha sätter själva kvantilen.
        # Parametrarna (n_estimators, max_depth) kan justeras för att öka scoren ytterligare
        model = GradientBoostingRegressor(loss='quantile', alpha=q, n_estimators=100, max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        
        task_preds[:, j] = model.predict(X_test)
    
    # Tvinga kvantilerna att vara stigande (non-decreasing) enligt Kaggles krav
    task_preds_sorted = np.sort(task_preds, axis=1)
    
    # Spara ID och värden i listor
    q_labels = [int(q*100) for q in QUANTILES]
    for i, r_id in enumerate(row_ids):
        for j, q_lab in enumerate(q_labels):
            all_row_ids.append(f"{r_id}_q{q_lab}")
            all_predictions.append(task_preds_sorted[i, j])

# ==========================================
# 3. EXPORT SUBMISSION
# ==========================================
print("\nSkapar slutgiltig submission_gbr.csv...")
df_out = pd.DataFrame({
    'id': all_row_ids,
    'value': all_predictions
})

df_out.to_csv("submission_gbr.csv", index=False)
print("Klar! Ladda upp submission_gbr.csv till Kaggle.")