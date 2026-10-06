import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. SETUP OCH DATA-LADDNING
# ==========================================
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TASKS = ['A', 'B', 'C', 'D']

def load_and_engineer_data(task):
    print(f"\n--- Laddar och pre-processar data för Task {task} ---")
    train = pd.read_csv(f'data/train_{task}.csv')
    test = pd.read_csv(f'data/test_{task}.csv')
    
    y_train = train['y'].to_numpy(dtype=float)
    
    # Slå ihop för One-Hot Encoding av textkolumner (t.ex. 'c1', 'c2')
    both = pd.concat([train.drop(columns='y'), test.drop(columns='row_id')], keys=["tr", "te"])
    both_encoded = pd.get_dummies(both, drop_first=True, dtype=float)
    
    X_train = both_encoded.loc["tr"].to_numpy()
    X_test = both_encoded.loc["te"].to_numpy()
    
    # FUSK 1: Lägg till baslinjens (OLS) prediktion som en ny feature!
    # Detta hjälper LightGBM enormt med heteroskedasticiteten (Task C)
    ols = LinearRegression()
    ols.fit(X_train, y_train)
    ols_train = ols.predict(X_train).reshape(-1, 1)
    ols_test = ols.predict(X_test).reshape(-1, 1)
    
    # Lägg till OLS-kolumnen längst bak i datamatrisen
    X_train = np.hstack([X_train, ols_train])
    X_test = np.hstack([X_test, ols_test])
    
    return X_train, y_train, X_test, test['row_id'].values

all_row_ids = []
all_predictions = []

# ==========================================
# 2. TRÄNA LIGHTGBM FÖR VARJE KVANTIL OCH TASK
# ==========================================
for task in TASKS:
    X_train, y_train, X_test, row_ids = load_and_engineer_data(task)
    task_preds = np.zeros((X_test.shape[0], len(QUANTILES)))
    
    # Justera trädens komplexitet (Task B och C behöver mer kapacitet)
    n_est = 150 if task in ['B', 'C'] else 100
    leaves = 31 if task in ['B', 'C'] else 15
    
    for j, q in enumerate(QUANTILES):
        # Objektive 'quantile' optimerar direkt för tävlingens Pinball Loss
        model = LGBMRegressor(
            objective='quantile',
            alpha=q,
            n_estimators=n_est,
            num_leaves=leaves,
            learning_rate=0.1,
            random_state=42,
            verbose=-1 # Stänger av onödiga varningar
        )
        model.fit(X_train, y_train)
        task_preds[:, j] = model.predict(X_test)
    
    # FUSK 2: Tillämpa vår teoretiska domänkunskap (Censureringen)
    if task == 'A':
        print("  Applicerar hårt tak (max 5.0) för Task A...")
        task_preds = np.clip(task_preds, a_min=None, a_max=5.0)
    elif task == 'D':
        print("  Applicerar hårt golv (min 0.0) för Task D...")
        task_preds = np.clip(task_preds, a_min=0.0, a_max=None)
        
    # KRAV: Kvantilerna måste vara stigande (non-decreasing)
    task_preds_sorted = np.sort(task_preds, axis=1)
    
    # Spara resultaten i rätt format
    q_labels = [int(q*100) for q in QUANTILES]
    for i, r_id in enumerate(row_ids):
        for j, q_lab in enumerate(q_labels):
            all_row_ids.append(f"{r_id}_q{q_lab}")
            all_predictions.append(task_preds_sorted[i, j])

# ==========================================
# 3. EXPORTERA TILL KAGGLE
# ==========================================
print("\nSkapar slutgiltig submission_lgbm.csv...")
df_out = pd.DataFrame({
    'id': all_row_ids,
    'value': all_predictions
})

df_out.to_csv("submission_lgbm.csv", index=False)
print("Klar! Ladda upp submission_lgbm.csv till Kaggle.")