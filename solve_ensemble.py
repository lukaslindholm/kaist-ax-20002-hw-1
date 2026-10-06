import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. SETUP OCH DATA-LADDNING
# ==========================================
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TASKS = ['A', 'B', 'C', 'D']

def load_and_engineer_data(task):
    print(f"\n--- Förbereder data för Task {task} ---")
    train = pd.read_csv(f'data/train_{task}.csv')
    test = pd.read_csv(f'data/test_{task}.csv')
    
    y_train = train['y'].to_numpy(dtype=float)
    
    # One-Hot Encoding för att hantera textkolumner
    both = pd.concat([train.drop(columns='y'), test.drop(columns='row_id')], keys=["tr", "te"])
    both_encoded = pd.get_dummies(both, drop_first=True, dtype=float)
    
    X_train = both_encoded.loc["tr"].to_numpy()
    X_test = both_encoded.loc["te"].to_numpy()
    
    # FEATURE ENGINEERING: Lägg till OLS-gissningen som en kolumn
    ols = LinearRegression().fit(X_train, y_train)
    ols_train = ols.predict(X_train).reshape(-1, 1)
    ols_test = ols.predict(X_test).reshape(-1, 1)
    
    X_train = np.hstack([X_train, ols_train])
    X_test = np.hstack([X_test, ols_test])
    
    return X_train, y_train, X_test, test['row_id'].values

all_row_ids = []
all_predictions = []

# ==========================================
# 2. ENSEMBLE TRAINING (LGBM + GBR)
# ==========================================
for task in TASKS:
    X_train, y_train, X_test, row_ids = load_and_engineer_data(task)
    
    # Matriser för att spara resultat från båda modellerna
    preds_lgbm = np.zeros((X_test.shape[0], len(QUANTILES)))
    preds_gbr = np.zeros((X_test.shape[0], len(QUANTILES)))
    
    # Vi ökar trädens kapacitet en aning för att fånga upp svansarna
    lgbm_leaves = 31 if task in ['B', 'C'] else 15
    gbr_depth = 4 if task in ['B', 'C'] else 3
    
    for j, q in enumerate(QUANTILES):
        print(f"  Tränar Ensemble för q={q}...")
        
        # Modell 1: LightGBM
        model_lgbm = LGBMRegressor(objective='quantile', alpha=q, n_estimators=150, 
                                   num_leaves=lgbm_leaves, learning_rate=0.1, random_state=42, verbose=-1)
        model_lgbm.fit(X_train, y_train)
        preds_lgbm[:, j] = model_lgbm.predict(X_test)
        
        # Modell 2: Gradient Boosting (Fungerar som Scikit-learns XGBoost-motsvarighet)
        model_gbr = GradientBoostingRegressor(loss='quantile', alpha=q, n_estimators=150, 
                                              max_depth=gbr_depth, learning_rate=0.1, random_state=42)
        model_gbr.fit(X_train, y_train)
        preds_gbr[:, j] = model_gbr.predict(X_test)
    
    # MAGIN MED ENSEMBLING: Ta genomsnittet av båda modellerna
    task_preds = (preds_lgbm + preds_gbr) / 2.0
    
    # DOMÄNKUNSKAP: Applicera klippning baserat på din diagnostik
    if task == 'A':
        task_preds = np.clip(task_preds, a_min=None, a_max=5.0)
    elif task == 'D':
        task_preds = np.clip(task_preds, a_min=0.0, a_max=None)
        
    # KRAV: Sortera kvantilerna (non-decreasing)
    task_preds_sorted = np.sort(task_preds, axis=1)
    
    # Förbered för export
    q_labels = [int(q*100) for q in QUANTILES]
    for i, r_id in enumerate(row_ids):
        for j, q_lab in enumerate(q_labels):
            all_row_ids.append(f"{r_id}_q{q_lab}")
            all_predictions.append(task_preds_sorted[i, j])

# ==========================================
# 3. EXPORT
# ==========================================
print("\nSkapar slutgiltig submission_ensemble.csv...")
pd.DataFrame({'id': all_row_ids, 'value': all_predictions}).to_csv("submission_ensemble.csv", index=False)
print("Klar! Ladda upp submission_ensemble.csv till Kaggle.")